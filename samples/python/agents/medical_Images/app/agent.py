import base64
import os
# Set PyTorch allocator configuration to prevent memory fragmentation on low-VRAM GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import time
import asyncio
import uuid
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, TypedDict, Annotated, AsyncIterable
import operator
import torch
import gc
import warnings
from PIL import Image, ImageEnhance

# Qdrant
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, MultiVectorConfig, MultiVectorComparator, Filter, FieldCondition, HasIdCondition

# MUVERA from fastembed
from fastembed.postprocess import Muvera

# ColPali - Visual document embeddings
from colpali_engine.models import ColPali as ColPaliModel
from colpali_engine.models import ColPaliProcessor

# LangChain / LangGraph / Tavily
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools import TavilySearchResults

# PDFs and Images
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# System settings (loaded dynamically to avoid import-time lifecycle issues)

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración del sistema ColPali Puro + MUVERA"""
    # Ruta absoluta basada en la ubicación del archivo
    BASE_DIR = Path(__file__).resolve().parent.parent / "histopatologia_data"
    EMBEDDINGS_DIR = BASE_DIR / "embeddings"
    ONTOLOGY_DIR = BASE_DIR / "ontologia"
    CACHE_DIR = BASE_DIR / "cache"

    ONTOLOGY_FILE = ONTOLOGY_DIR / "ontologia_histopatologia.json"

    # Dimensiones de embeddings (SOLO ColPali)
    COLPALI_EMBEDDING_DIM = 128  # ColPali dimensión por vector
    FDE_DIM = 20480              # MUVERA FDE dimension (64 clusters * 16 dim_proj * 20 reps)

    # Parámetros de procesamiento
    TEXT_CHUNK_SIZE = 1000
    TEXT_CHUNK_OVERLAP = 100
    IMAGE_DPI = 150
    MAX_IMAGE_SIZE = (448, 448)

    # Parámetros de memoria
    BATCH_SIZE = 8
    CLEAR_CACHE_AFTER_PROCESS = True

    # Mejoras visuales
    ENHANCE_CONTRAST = True
    ENHANCE_BRIGHTNESS = True
    CONTRAST_FACTOR = 1.2
    BRIGHTNESS_FACTOR = 1.1

    # Parámetros de búsqueda
    SEARCH_PREFETCH_MULTIPLIER = 20

    SEARCH_SCORE_THRESHOLD = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.0"))
    NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

    # Cuantización: 8 = mejor precisión en scores (~870+), 4 = menos VRAM (~800 scores)
    QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "8"))

    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.EMBEDDINGS_DIR, cls.ONTOLOGY_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

def cleanup_memory():
    """Liberar memoria GPU/CPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ============================================================================
# EXTRACTOR DE ONTOLOGÍA
# ============================================================================

class ExtractorOntologia:
    """Extrae ontología histopatológica usando Groq"""

    def __init__(self, api_key: str):
        if not api_key:
            print("⚠️ API Key de Groq no proporcionada para ExtractorOntologia")
            self.model = None
            return
        from groq import Groq as GroqClient
        self._groq_client = GroqClient(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    @staticmethod
    def extraer_caption_imagen(page_fitz, img_bbox, texto_pagina_completo: str) -> str:
        """Extrae la etiqueta 'Imagen X.X' / 'Fig X.X' + TODO el texto debajo de la imagen."""
        caption = ""
        try:
            page_rect = page_fitz.rect
            margen_overlap = 10
            area_expandida = fitz.Rect(0, max(0, img_bbox[3] - margen_overlap), page_rect.width, page_rect.height)
            texto_expandido = page_fitz.get_text("text", clip=area_expandida).strip()
            
            if texto_expandido:
                caption = texto_expandido
            else:
                area_abajo = fitz.Rect(0, img_bbox[3], page_rect.width, page_rect.height)
                caption = page_fitz.get_text("text", clip=area_abajo).strip()
        except Exception:
            pass
        
        if caption:
            caption = re.sub(r'\n\s*\d{1,3}\s*$', '', caption).strip()
            return caption
        return texto_pagina_completo[:500] if texto_pagina_completo else ""

    def extraer_ontologia_completa(self, contenido: str, num_imagenes: int) -> Dict:
        """Extrae ontología completa del documento"""
        if not self.model:
            return {"sistemas_anatomicos": [], "metadata": {"tipo": "default"}}

        print(f"\n🔬 Extrayendo ontología de {len(contenido)} caracteres...")
        
        prompt = f"""Analiza este atlas de histopatología y extrae una ontología completa.

CONTENIDO TEXTUAL (muestra):
{contenido[:8000]}...

IMÁGENES: {num_imagenes} figuras

EXTRAE:
1. SISTEMAS ANATÓMICOS: órganos, tejidos, estructuras
2. TERMINOLOGÍA HISTOLÓGICA: tipos celulares, componentes tisulares
3. TÉCNICAS Y TINCIONES: métodos de procesamiento, coloraciones
4. FIGURAS: numeración y descripciones breves
5. PATOLOGÍAS: alteraciones, lesiones comunes

Responde SOLAMENTE con un JSON válido, sin texto adicional ni explicaciones."""

        for intento in range(2):
            try:
                prompt_actual = prompt if intento == 0 else \
                    f"Extrae una ontología en formato JSON puro (sin markdown) del siguiente texto de histopatología:\n{contenido[:5000]}"
                response = self._groq_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt_actual}],
                    temperature=0
                )
                ontologia_texto = response.choices[0].message.content.strip()
                if '```' in ontologia_texto:
                    bloques = ontologia_texto.split('```')
                    for bloque in bloques:
                        bloque_limpio = bloque.strip()
                        if bloque_limpio.startswith('json'):
                            bloque_limpio = bloque_limpio[4:].strip()
                        if bloque_limpio.startswith('{'):
                            ontologia_texto = bloque_limpio
                            break
                
                ontologia = json.loads(ontologia_texto)
                ontologia["metadata"] = {
                    "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "modelo": self.model,
                    "num_imagenes": num_imagenes
                }

                with open(Config.ONTOLOGY_FILE, 'w', encoding='utf-8') as f:
                    json.dump(ontologia, f, indent=2, ensure_ascii=False)

                print(f"✅ Ontología extraída: {len(ontologia)} categorías")
                return ontologia

            except json.JSONDecodeError as e:
                print(f"⚠️ Intento {intento+1}/2 - Error parsing JSON ontología: {e}")
                if intento == 0:
                    continue
            except Exception as e:
                print(f"⚠️ Error ontología (intento {intento+1}): {e}")
                break

        print("⚠️ No se pudo extraer ontología. Continuando sin ella.")
        return {"sistemas_anatomicos": [], "metadata": {"tipo": "default"}}

    def cargar_ontologia(self) -> Optional[Dict]:
        """Cargar ontología desde archivo"""
        if Config.ONTOLOGY_FILE.exists():
            try:
                with open(Config.ONTOLOGY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def buscar_en_ontologia(self, termino: str, ontologia: Dict) -> List[str]:
        """Buscar términos relevantes en ontología"""
        resultados = []
        termino_lower = termino.lower()

        def buscar_recursivo(obj, ruta=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    nueva_ruta = f"{ruta}/{k}" if ruta else k
                    if termino_lower in k.lower():
                        resultados.append(f"{nueva_ruta}: {str(v)[:100]}")
                    buscar_recursivo(v, nueva_ruta)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and termino_lower in item.lower():
                        resultados.append(f"{ruta}: {item}")

        buscar_recursivo(ontologia)
        return resultados[:5]

# ============================================================================
# PROCESADOR COLPALI PURO + MUVERA
# ============================================================================

class ProcesadorColPaliPuro:
    """Procesador simplificado usando SOLO ColPali para texto e imágenes"""

    def __init__(self):
        print("\n🖼️ Inicializando ColPali Puro + MUVERA...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bits = Config.QUANTIZATION_BITS
        print(f"   📚 Cargando ColPali v1.2 ({bits}-bit, texto + imágenes)...")
        
        try:
            from transformers import BitsAndBytesConfig
            cleanup_memory()

            if torch.cuda.is_available():
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cudnn.enabled = False

            quantization_config = None
            if bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
            elif bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            
            kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            else:
                kwargs["torch_dtype"] = torch.bfloat16

            self.colpali_model = ColPaliModel.from_pretrained(
                "vidore/colpali-v1.2",
                **kwargs
            )
            self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            self.colpali_model.eval()
            print(f"   ✅ ColPali cargado ({bits}-bit, {Config.COLPALI_EMBEDDING_DIM}D multi-vector)")
        except Exception as e:
            if bits == 8:
                print(f"   ⚠️ Error con 8-bit, intentando fallback a 4-bit: {e}")
                try:
                    if hasattr(self, 'colpali_model') and self.colpali_model is not None:
                        del self.colpali_model
                        self.colpali_model = None
                    cleanup_memory()

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )
                    self.colpali_model = ColPaliModel.from_pretrained(
                        "vidore/colpali-v1.2",
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
                    self.colpali_model.eval()
                    print(f"   ✅ ColPali cargado (4-bit fallback, {Config.COLPALI_EMBEDDING_DIM}D multi-vector)")
                except Exception as e2:
                    print(f"   ❌ Error cargando ColPali: {e2}")
                    self.colpali_model = None
                    self.colpali_processor = None
            else:
                print(f"   ❌ Error cargando ColPali: {e}")
                self.colpali_model = None
                self.colpali_processor = None

        print("   🚀 Inicializando MUVERA...")
        self.muvera = Muvera(
            dim=128,
            k_sim=6,
            dim_proj=16,
            r_reps=20,
            random_seed=42,
        )
        print(f"   ✅ MUVERA inicializado (FDE: {Config.FDE_DIM}D)")

    def __del__(self):
        cleanup_memory()

    def extraer_imagenes_pdf(self, pdf_path: str) -> List[Dict]:
        """Extrae imágenes del PDF usando PyMuPDF"""
        print(f"📄 Extrayendo imágenes de {pdf_path}...")
        imagenes = []
        nombre_base = Path(pdf_path).stem
        
        if fitz is not None:
            try:
                doc = fitz.open(pdf_path)
                image_count = 0
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    valid_images_this_page = []
                    texto_pagina_completo = page.get_text("text").strip()
                    
                    img_info_list = page.get_image_info(xrefs=True)
                    page_xrefs = [info["xref"] for info in img_info_list if info.get("xref")]
                    page_y_positions = {info["xref"]: info.get("bbox", (0,0,0,0))[1] for info in img_info_list if info.get("xref")}
                    page_bboxes = {info["xref"]: info.get("bbox", (0,0,0,0)) for info in img_info_list if info.get("xref")}
                    
                    if page_xrefs:
                        for xref in page_xrefs:
                            try:
                                base_image = doc.extract_image(xref)
                                if not base_image:
                                    continue
                                
                                image_bytes = base_image["image"]
                                ext = base_image["ext"]
                                y_position = page_y_positions.get(xref, 0.0)
                                bbox = page_bboxes.get(xref, (0,0,0,0))
                                
                                caption_extraido = ExtractorOntologia.extraer_caption_imagen(page, bbox, texto_pagina_completo)
                                
                                image_count += 1
                                img_path = Config.EMBEDDINGS_DIR / f"{nombre_base}_p{page_num+1}_img{image_count}.{ext}"
                                with open(img_path, "wb") as f:
                                    f.write(image_bytes)
                                    
                                img_pil = Image.open(img_path)
                                width, height = img_pil.size
                                area = width * height
                                
                                if width >= 150 and height >= 150:
                                    if img_pil.mode != "RGB":
                                        img_pil = img_pil.convert("RGB")
                                    
                                    target_size = 448
                                    if max(width, height) > target_size:
                                        img_pil.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
                                        width, height = img_pil.size
                                        area = width * height
                                    
                                    img_path_rgb = Config.EMBEDDINGS_DIR / f"{nombre_base}_p{page_num+1}_img{image_count}.jpg"
                                    img_pil.save(img_path_rgb, "JPEG")
                                    
                                    if str(img_path) != str(img_path_rgb):
                                        os.remove(img_path)
                                        
                                    img_path = img_path_rgb

                                    valid_images_this_page.append({
                                        "page": page_num + 1,
                                        "path": str(img_path),
                                        "type": "extracted_figure",
                                        "size": (width, height),
                                        "area": area,
                                        "y_position": y_position,
                                        "caption": caption_extraido
                                    })
                                else:
                                    os.remove(img_path)
                            except Exception as e:
                                print(f"⚠️ Error procesando xref {xref} en página {page_num+1}: {e}")
                    
                    if valid_images_this_page:
                        largest_image = max(valid_images_this_page, key=lambda x: x["area"])
                        for img_data in valid_images_this_page:
                            if img_data["path"] != largest_image["path"]:
                                try:
                                    os.remove(img_data["path"])
                                except OSError:
                                    pass
                        largest_image["img_index_in_page"] = 0
                        largest_image["total_images_in_page"] = 1
                        imagenes.append(largest_image)

                if len(imagenes) > 0:
                    print(f"✅ {len(imagenes)} figuras extraídas del PDF.")
                    return imagenes
            except Exception as e:
                print(f"⚠️ Error extracción: {e}")
        return imagenes

    def _preprocesar_imagen(self, imagen_path: str) -> Image.Image:
        """Preprocesamiento para histopatología"""
        image = Image.open(imagen_path).convert("RGB")
        target_size = 448
        if max(image.size) > target_size:
            image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        if Config.ENHANCE_CONTRAST:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(Config.CONTRAST_FACTOR)

        if Config.ENHANCE_BRIGHTNESS:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(Config.BRIGHTNESS_FACTOR)

        return image

    def generar_embedding_imagen(self, imagen_path: str) -> Optional[np.ndarray]:
        """Genera embedding ColPali para imagen"""
        if self.colpali_model is None:
            return None
        cleanup_memory()
        try:
            image = self._preprocesar_imagen(imagen_path)
            batch_images = self.colpali_processor.process_images([image])
            batch_images = {k: v.to(self.colpali_model.device) for k, v in batch_images.items()}

            with torch.no_grad():
                if "cuda" in str(self.colpali_model.device):
                    with torch.amp.autocast('cuda'):
                        image_embeddings = self.colpali_model(**batch_images)
                else:
                    image_embeddings = self.colpali_model(**batch_images)

            multivector = image_embeddings[0].cpu().float().numpy()
            
            # Liberar explícitamente tensores intermedios
            del batch_images
            del image_embeddings
            cleanup_memory()

            if Config.NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(multivector, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                multivector = multivector / norms

            return multivector
        except Exception as e:
            print(f"❌ Error generando embedding imagen: {e}")
            cleanup_memory()
            return None

    def generar_embedding_texto(self, texto: str) -> Optional[np.ndarray]:
        """Genera embedding ColPali para TEXTO"""
        if self.colpali_model is None:
            return None
        cleanup_memory()
        try:
            batch_queries = self.colpali_processor.process_queries([texto])
            batch_queries = {k: v.to(self.colpali_model.device) for k, v in batch_queries.items()}
            
            with torch.no_grad():
                if "cuda" in str(self.colpali_model.device):
                    with torch.amp.autocast('cuda'):
                        text_embeddings = self.colpali_model(**batch_queries)
                else:
                    text_embeddings = self.colpali_model(**batch_queries)
            
            multivector = text_embeddings[0].cpu().float().numpy()
            
            # Liberar explícitamente tensores intermedios
            del batch_queries
            del text_embeddings
            cleanup_memory()

            if Config.NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(multivector, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                multivector = multivector / norms

            return multivector
        except Exception as e:
            print(f"❌ Error embedding texto: {e}")
            cleanup_memory()
            return None

    def generar_fde_muvera(self, multivectors: np.ndarray) -> np.ndarray:
        mv = np.array(multivectors, dtype=np.float32)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)
        return self.muvera.process_document(mv)

    def generar_query_muvera(self, query_multivectors: np.ndarray) -> np.ndarray:
        mv = np.array(query_multivectors, dtype=np.float32)
        if mv.ndim == 1:
            mv = mv.reshape(1, -1)
        return self.muvera.process_query(mv)

# ============================================================================
# GESTOR DE QDRANT CON MUVERA
# ============================================================================

class GestorQdrantMuvera:
    """Gestor de Qdrant con arquitectura dual MUVERA"""

    def __init__(self, url: str, api_key: str, collection_base: str):
        self.url = url
        self.api_key = api_key
        self.collection_base = collection_base
        self._client = None
        self.content_mv_collection = f"{collection_base}_content_mv"
        self.content_fde_collection = f"{collection_base}_content_fde"

    @property
    def client(self):
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=120,
                prefer_grpc=False
            )
            print("🔗 Cliente Qdrant conectado")
        return self._client

    async def crear_colecciones(self):
        print("\n📦 Creando colecciones Qdrant...")
        client = self.client
        
        for col, size_dim, is_mv in [
            (self.content_mv_collection, Config.COLPALI_EMBEDDING_DIM, True),
            (self.content_fde_collection, Config.FDE_DIM, False)
        ]:
            try:
                await client.get_collection(col)
            except Exception:
                if is_mv:
                    await client.create_collection(
                        collection_name=col,
                        vectors_config=VectorParams(
                            size=size_dim,
                            distance=Distance.COSINE,
                            multivector_config=MultiVectorConfig(
                                comparator=MultiVectorComparator.MAX_SIM
                            )
                        )
                    )
                else:
                    await client.create_collection(
                        collection_name=col,
                        vectors_config=VectorParams(
                            size=size_dim,
                            distance=Distance.COSINE
                        )
                    )

        try:
            from qdrant_client.models import PayloadSchemaType
            for col in [self.content_mv_collection, self.content_fde_collection]:
                for field, schema in [
                    ("tipo", PayloadSchemaType.KEYWORD),
                    ("numero_pagina", PayloadSchemaType.INTEGER),
                    ("nombre_archivo", PayloadSchemaType.KEYWORD)
                ]:
                    try:
                        await client.create_payload_index(
                            collection_name=col,
                            field_name=field,
                            field_schema=schema
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    async def insertar_batch_muvera(self, points_mv: List[PointStruct], points_fde: List[PointStruct]):
        client = self.client
        await client.upsert(collection_name=self.content_mv_collection, points=points_mv, wait=True)
        await client.upsert(collection_name=self.content_fde_collection, points=points_fde, wait=True)

    async def buscar_muvera_2stage(
        self,
        query_multivector: np.ndarray,
        query_fde: np.ndarray,
        top_k: int = 5,
        prefetch_multiplier: int = Config.SEARCH_PREFETCH_MULTIPLIER,
        min_score: float = 0.0,
        figuras_filtro: List[str] = None,
        filtro_tipo: str = None,
        filtro_paginas: List[int] = None
    ) -> Tuple[List[Dict], bool]:
        client = self.client
        has_rejected = False

        try:
            # STAGE 1: Fast FDE search
            must_conditions = []
            should_conditions = []
            
            if filtro_tipo:
                from qdrant_client.models import MatchValue
                must_conditions.append(FieldCondition(key="tipo", match=MatchValue(value=filtro_tipo)))
            if figuras_filtro:
                from qdrant_client.models import MatchAny
                should_conditions.append(FieldCondition(key="figuras", match=MatchAny(any=figuras_filtro)))
            if filtro_paginas:
                from qdrant_client.models import MatchAny
                must_conditions.append(FieldCondition(key="numero_pagina", match=MatchAny(any=filtro_paginas)))
                
            qdrant_filter = None
            if must_conditions or should_conditions:
                qdrant_filter = Filter(
                    must=must_conditions if must_conditions else None,
                    should=should_conditions if should_conditions else None
                )

            fde_response = await client.query_points(
                collection_name=self.content_fde_collection,
                query=query_fde.tolist(),
                query_filter=qdrant_filter,
                limit=top_k * prefetch_multiplier,
                with_payload=False
            )
            
            if not fde_response.points:
                return [], False
            
            candidate_ids = [point.id for point in fde_response.points]

            # STAGE 2: Precise multi-vector reranking
            mv_response = await client.query_points(
                collection_name=self.content_mv_collection,
                query=query_multivector.tolist(),
                query_filter=Filter(
                    must=[HasIdCondition(has_id=candidate_ids)]
                ),
                limit=top_k * 2
            )

            todos_candidatos = []
            for r in mv_response.points:
                score = float(r.score)
                tiene_figura_exacta = False
                if figuras_filtro and r.payload and "figuras" in r.payload:
                    if any(f in r.payload["figuras"] for f in figuras_filtro):
                        tiene_figura_exacta = True
                todos_candidatos.append({
                    "id": r.id,
                    "score": score,
                    "payload": r.payload or {},
                    "tiene_figura_exacta": tiene_figura_exacta
                })

            if not todos_candidatos:
                return [], False

            todos_candidatos.sort(key=lambda x: x['score'], reverse=True)
            mejor_score = todos_candidatos[0]['score']
            umbral = min_score

            resultados = []
            descartados = 0
            for c in todos_candidatos:
                if c['score'] >= umbral or c['tiene_figura_exacta']:
                    resultados.append({
                        "id": c['id'],
                        "score": c['score'],
                        "payload": c['payload']
                    })
                else:
                    descartados += 1

            if descartados > 0:
                print(f"      🗑️ {descartados} candidatos rechazados (score < {umbral:.4f})")

            resultados = resultados[:top_k]
            if not resultados:
                has_rejected = umbral > 0.0

            return resultados, has_rejected

        except Exception as e:
            print(f"❌ Error búsqueda MUVERA: {e}")
            return [], False

# ============================================================================
# MEMORIA DE CHAT SQLITE
# ============================================================================

import sqlite3
import datetime

class MemoriaSQLite:
    """Gestor de memoria lineal usando SQLite y resúmenes con LLM"""
    def __init__(self, db_path: str = "./chat_memory.sqlite"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT,
                summary TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def add_interaction_summary(self, session_id: str, user_query: str, summary: str):
        if not user_query.strip() or not summary.strip():
            return
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO interactions (session_id, timestamp, user_query, summary) VALUES (?, ?, ?, ?)",
                (session_id, datetime.datetime.now(), user_query, summary)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"   ⚠️ Error guardando en SQLite: {e}")
            
    def get_relevant_history(self, session_id: str, n_results: int = 5) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "SELECT summary FROM interactions WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", 
                (session_id, n_results)
            )
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                return ""
            
            rows.reverse()
            history = "\n---\n".join([row[0] for row in rows])
            return history
        except Exception as e:
            print(f"   ⚠️ Error recuperando memoria SQLite: {e}")
            return ""

# ============================================================================
# COMPLEMENTOS
# ============================================================================

_KEYWORDS_IMAGEN = [
    "mostrá", "mostrar", "muéstrame",
    "enseñame", "quiero ver una", "ver imagen",
    "ver foto", "ver figura", "dame una imagen",
    "buscá una imagen", "buscar una imagen"
]
_PATRON_IMAGEN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _KEYWORDS_IMAGEN) + r")\b",
    re.IGNORECASE,
)

def detectar_intencion_imagen(texto: str) -> bool:
    return bool(_PATRON_IMAGEN.search(texto))

def filtrar_resultados_busqueda(resultados: List[Dict], requiere_imagen: bool, tiene_imagen_adjunta: bool) -> Tuple[List[Dict], List[str]]:
    if not requiere_imagen and not tiene_imagen_adjunta:
        filtrados = [r for r in resultados if r.get("payload", {}).get("tipo") != "imagen"]
        return filtrados, []

    filtrados = []
    imagenes_relevantes = []
    for r in resultados:
        payload = r.get("payload", {})
        if payload.get("tipo") == "imagen":
            filtrados.append(r)
            ruta = payload.get("imagen_path")
            if ruta:
                imagenes_relevantes.append(ruta)
        else:
            filtrados.append(r)
    return filtrados, imagenes_relevantes

def rerank_imagenes_por_caption(query_embedding: np.ndarray, candidatas: List[Dict], umbral: float = 0.45) -> List[Dict]:
    if len(candidatas) == 0:
        return []

    q = np.asarray(query_embedding, dtype=np.float64)
    scored = []
    for cand in candidatas:
        emb = cand.get("caption_embedding")
        if emb is None:
            continue
        c = np.asarray(emb, dtype=np.float64)
        
        if q.ndim == 2 and c.ndim == 2:
            sim_matrix = np.dot(q, c.T)
            sim = float(np.sum(np.max(sim_matrix, axis=1)))
        else:
            q_mean = q.mean(axis=0) if q.ndim == 2 else q
            c_mean = c.mean(axis=0) if c.ndim == 2 else c
            q_norm = np.linalg.norm(q_mean)
            c_norm = np.linalg.norm(c_mean)
            q_mean = q_mean / q_norm if q_norm > 0 else q_mean
            c_mean = c_mean / c_norm if c_norm > 0 else c_mean
            sim = float(np.dot(q_mean, c_mean))

        if sim >= umbral:
            scored.append((sim, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        top_sim = scored[0][0]
        rel_cutoff = top_sim * 0.75
        scored = [(s, c) for s, c in scored if s >= rel_cutoff]
    
    return [cand for _, cand in scored]

# ============================================================================
# ESTADO DEL GRAFO
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    consulta_usuario: str
    consulta_resuelta: str
    imagen_consulta: Optional[str]
    contexto_memoria: str
    ontologia: Dict
    contexto_ontologico: str
    clasificacion: str
    requiere_imagen: bool
    consulta_optimizada: str
    filtros_ontologia: List[str]
    resultados_busqueda: List[Dict[str, Any]]
    contexto_documentos: str
    imagenes_relevantes: List[Any]
    respuesta_final: str
    trayectoria: Annotated[List[Dict[str, Any]], operator.add]
    imagen_base64: Optional[str]
    user_id: str
    tiempo_inicio: float
    abortar_reset: bool

# ============================================================================
# CLASE MEDICAL AGENT CON COLPALI Y FALLBACK
# ============================================================================

class MedicalAgent:
    """Agente médico con Colpali local y fallback inteligente a búsquedas web"""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    SYSTEM_INSTRUCTION = (
        'Eres un médico especialista experimentado que analiza consultas médicas, '
        'imágenes histológicas y proporciona análisis profesionales.'
    )

    def __init__(self):
        Config.setup_directories()
        
        # LLM principal (Llama-4 Scout de Groq)
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Directorio temporal de uploads
        self.uploads_dir = Config.BASE_DIR / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Tavily Search Tool
        self.tavily_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )

        self.procesador = None
        self.gestor_qdrant = None
        self.extractor_ontologia = None
        self.ontologia = None
        self.memoria = None

    def inicializar_componentes(self):
        """Cargar modelos y base de datos (lazy initialization para uvicorn)"""
        if self.procesador is not None:
            return

        print("\n🏥 Inicializando modelos y DB para MedicalAgent...")
        self.procesador = ProcesadorColPaliPuro()
        self.memoria = MemoriaSQLite(db_path=str(Config.BASE_DIR / 'chat_memory.sqlite'))
        
        self.gestor_qdrant = GestorQdrantMuvera(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_KEY", ""),
            collection_base="histopatologia"
        )
        
        self.extractor_ontologia = ExtractorOntologia(os.getenv("GROQ_API_KEY"))
        self.ontologia = self.extractor_ontologia.cargar_ontologia()
        
        # Inicializar LangGraph interno
        self._inicializar_langgraph()
        cleanup_memory()

    @property
    def qdrant_client(self):
        self.inicializar_componentes()
        return self.gestor_qdrant.client

    def _inicializar_langgraph(self):
        """Define la arquitectura del flujo del agente"""
        graph = StateGraph(AgentState)

        graph.add_node("recepcionar_consulta", self._nodo_recepcionar_consulta)
        graph.add_node("inicializar", self._nodo_inicializar)
        graph.add_node("analizar_ontologia", self._nodo_analizar_ontologia)
        graph.add_node("clasificar", self._nodo_clasificar)
        graph.add_node("optimizar_consulta", self._nodo_optimizar_consulta)
        graph.add_node("buscar", self._nodo_buscar)
        graph.add_node("generar_respuesta", self._nodo_generar_respuesta)
        graph.add_node("fallback_internet", self._nodo_fallback_internet)
        graph.add_node("reset", self._nodo_reset)
        graph.add_node("finalizar", self._nodo_finalizar)

        graph.add_edge(START, "recepcionar_consulta")
        graph.add_edge("recepcionar_consulta", "inicializar")
        graph.add_edge("inicializar", "analizar_ontologia")
        graph.add_edge("analizar_ontologia", "clasificar")
        graph.add_edge("clasificar", "optimizar_consulta")
        graph.add_edge("optimizar_consulta", "buscar")
        
        graph.add_conditional_edges(
            "buscar",
            self._decidir_camino_tras_busqueda,
            {
                "generar": "generar_respuesta",
                "fallback": "fallback_internet"
            }
        )
        
        graph.add_edge("generar_respuesta", "finalizar")
        graph.add_edge("fallback_internet", "finalizar")
        graph.add_edge("reset", "finalizar")
        graph.add_edge("finalizar", END)

        self.compiled_graph = graph.compile()

    # ========== NODOS DEL GRAFO ==========

    async def _nodo_recepcionar_consulta(self, state: AgentState) -> AgentState:
        state["trayectoria"] = [{"nodo": "recepcionar_consulta", "timestamp": time.time()}]
        if state.get("imagen_base64"):
            try:
                image_data = base64.b64decode(state["imagen_base64"])
                filename = f"query_image_{uuid.uuid4().hex}.jpg"
                filepath = self.uploads_dir / filename
                with open(filepath, "wb") as f:
                    f.write(image_data)
                state["imagen_consulta"] = str(filepath)
                print(f"📸 Imagen de consulta guardada en: {filepath}")
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                state["imagen_consulta"] = None
        return state

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        state["ontologia"] = self.ontologia or {}
        state["tiempo_inicio"] = time.time()
        
        history = self.memoria.get_relevant_history(session_id=state["user_id"], n_results=5)
        state["contexto_memoria"] = history
            
        consulta_resuelta = state["consulta_usuario"]
        if history and history.strip():
            try:
                prompt_resolucion = f"""Eres un experto en histopatología y lingüística. Resuelve referencias en la consulta actual del usuario utilizando el historial.
Historial:
{history}
Consulta actual: {state["consulta_usuario"]}
Responde ÚNICAMENTE con la consulta reescrita sin explicaciones."""

                messages = [
                    SystemMessage(content="Eres un asistente que resuelve correferencias."),
                    HumanMessage(content=prompt_resolucion)
                ]
                resolucion_response = await self.llm.ainvoke(messages)
                resolved = resolucion_response.content.strip()
                if resolved:
                    consulta_resuelta = resolved
            except Exception as e:
                print(f"⚠️ Error resolviendo referencias: {e}")
        
        state["consulta_resuelta"] = consulta_resuelta
        state["trayectoria"].append({"nodo": "inicializar", "timestamp": time.time()})
        return state

    async def _nodo_analizar_ontologia(self, state: AgentState) -> AgentState:
        if not state["ontologia"]:
            state["contexto_ontologico"] = "No disponible"
            state["filtros_ontologia"] = []
        else:
            terminos = self.extractor_ontologia.buscar_en_ontologia(state["consulta_resuelta"], state["ontologia"])
            state["contexto_ontologico"] = "\n".join(terminos)
            state["filtros_ontologia"] = [t.split(":")[1].strip() for t in terminos[:3]] if terminos else []

        state["trayectoria"].append({"nodo": "analizar_ontologia", "timestamp": time.time()})
        return state

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        imagen_upload = bool(state.get('imagen_consulta') and os.path.exists(state['imagen_consulta']))
        info_imagen = "\nImagen adjunta: Sí" if state.get('imagen_consulta') else "\nImagen adjunta: No"
        
        messages = [
            SystemMessage(content="""Clasifica si el usuario pide ver una imagen de forma explícita.
Termina tu respuesta EXACTAMENTE con la línea "REQUIERE_IMAGEN: TRUE" o "REQUIERE_IMAGEN: FALSE"."""),
            HumanMessage(content=f"CONSULTA: {state['consulta_usuario']}{info_imagen}\nCONTEXTO ONTOLÓGICO:\n{state['contexto_ontologico']}")
        ]
        response = await self.llm.ainvoke(messages)
        state["clasificacion"] = response.content

        if imagen_upload:
            state["requiere_imagen"] = True
        elif "REQUIERE_IMAGEN: TRUE" in response.content.upper():
            state["requiere_imagen"] = True
        elif "REQUIERE_IMAGEN: FALSE" in response.content.upper():
            state["requiere_imagen"] = False
        else:
            state["requiere_imagen"] = detectar_intencion_imagen(state['consulta_usuario'])

        state["trayectoria"].append({"nodo": "clasificar", "timestamp": time.time()})
        return state

    async def _nodo_optimizar_consulta(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content="""Optimiza la consulta para búsqueda en atlas RAG. Responde SOLO los términos optimizados."""),
            HumanMessage(content=f"CONSULTA: {state['consulta_resuelta']}\nCONTEXTO: {state['contexto_ontologico'][:500]}")
        ]
        response = await self.llm.ainvoke(messages)
        state["consulta_optimizada"] = response.content.strip()
        state["trayectoria"].append({"nodo": "optimizar_consulta", "timestamp": time.time()})
        return state

    def _calcular_dhash(self, image_path: str, hash_size: int = 16) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("L")
            img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.float32)
            return (pixels[:, 1:] > pixels[:, :-1]).flatten()
        except Exception:
            return None

    def _verificar_match_visual(self, query_path: str, match_path: str) -> float:
        hash1 = self._calcular_dhash(query_path)
        hash2 = self._calcular_dhash(match_path)
        if hash1 is None or hash2 is None:
            return 1.0
        hamming_distance = np.sum(hash1 != hash2)
        return 1.0 - (hamming_distance / len(hash1))

    def _extraer_figuras_de_texto(self, texto: str) -> List[str]:
        patrones_figura = [
            r'[Ff]igura\s+(\d+[\-\.·]\d+)',
            r'[Ff]i[gG~][\.\s]*\s*(\d+[\-\.·\s]\d+)',
        ]
        patrones_imagen = [
            r'[Ii]magen\s+(\d+[\-\.·]\d+)',
            r'[Ii]magen\s+(\d+)(?=\s*:)',
        ]
        figuras = set()
        for patron in patrones_figura:
            matches = re.findall(patron, texto)
            for m in matches:
                figuras.add(f"Figura {re.sub(r'[·\.\s]', '-', m)}")
        for patron in patrones_imagen:
            matches = re.findall(patron, texto)
            for m in matches:
                figuras.add(f"Imagen {re.sub(r'[·\.\s]', '-', m)}")
        return sorted(list(figuras))

    async def _nodo_buscar(self, state: AgentState) -> AgentState:
        resultados = []
        has_rejected = False
        state["abortar_reset"] = False

        requiere_imagen = state.get('requiere_imagen', False)
        tiene_imagen_adjunta = bool(state.get('imagen_consulta') and os.path.exists(state['imagen_consulta']))

        # CASE 1: Query contains image upload
        if tiene_imagen_adjunta:
            print("   🔍 Búsqueda local por imagen...")
            query_mv = self.procesador.generar_embedding_imagen(state['imagen_consulta'])
            if query_mv is not None:
                query_fde = self.procesador.generar_query_muvera(query_mv)
                figuras_filtro = self._extraer_figuras_de_texto(state['consulta_optimizada'])

                resultados, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                    query_mv, query_fde, min_score=0.0, figuras_filtro=figuras_filtro, filtro_tipo="imagen"
                )
                resultados, _ = filtrar_resultados_busqueda(resultados, requiere_imagen=True, tiene_imagen_adjunta=True)

                # Verificación por dHash y MaxSim
                UMBRAL_VERIFICACION = float(os.getenv("VERIFICATION_THRESHOLD", "830"))
                ids_rechazados = set()
                
                for img_res in [r for r in resultados if r.get('payload', {}).get('tipo') == 'imagen']:
                    match_path = img_res['payload'].get('imagen_path', '')
                    if not match_path or not os.path.exists(match_path):
                        continue
                    
                    match_mv = self.procesador.generar_embedding_imagen(match_path)
                    if match_mv is None:
                        continue
                    
                    sim_matrix = np.dot(query_mv, match_mv.T)
                    maxsim = float(np.sum(np.max(sim_matrix, axis=1)))
                    
                    if maxsim < UMBRAL_VERIFICACION:
                        ids_rechazados.add(img_res['id'])
                    else:
                        dhash_sim = self._verificar_match_visual(state['imagen_consulta'], match_path)
                        if dhash_sim < 0.80:
                            ids_rechazados.add(img_res['id'])

                if ids_rechazados:
                    resultados = [r for r in resultados if r.get('id') not in ids_rechazados]
                    if len([r for r in resultados if r.get('payload', {}).get('tipo') == 'imagen']) == 0:
                        has_rejected = True
                
                if not resultados:
                    has_rejected = True

        # CASE 2: Text query requesting image
        elif requiere_imagen:
            print("   🔍 Búsqueda local por descripción de imagen...")
            query_mv = self.procesador.generar_embedding_texto(state['consulta_optimizada'])
            if query_mv is not None:
                query_fde = self.procesador.generar_query_muvera(query_mv)
                figuras_filtro = self._extraer_figuras_de_texto(state['consulta_optimizada'])

                resultados_texto, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                    query_mv, query_fde, min_score=0.0, figuras_filtro=figuras_filtro, filtro_tipo="texto"
                )

                # Identificar documento principal y buscar imágenes asociadas
                doc_scores = {}
                for r in resultados_texto:
                    doc = r.get('payload', {}).get('nombre_archivo', '')
                    if doc:
                        doc_scores[doc] = doc_scores.get(doc, 0.0) + r.get('score', 0.0)
                
                doc_principal = max(doc_scores, key=doc_scores.get) if doc_scores else ''
                imagenes_encontradas = []

                if doc_principal:
                    try:
                        client = self.gestor_qdrant.client
                        scroll_res = await client.scroll(
                            collection_name=self.gestor_qdrant.content_mv_collection,
                            scroll_filter=Filter(
                                must=[
                                    FieldCondition(key="tipo", match={"value": "texto"}),
                                    FieldCondition(key="nombre_archivo", match={"value": doc_principal})
                                ]
                            ),
                            limit=1000
                        )
                        chunks_doc = scroll_res[0] if scroll_res else []
                        paginas_con_etiqueta = {}
                        
                        for chunk in chunks_doc:
                            cp = chunk.payload or {}
                            texto_chunk = cp.get('texto', '')
                            pg_chunk = cp.get('numero_pagina')
                            matches = re.findall(r'[Ii]magen\s+(\d+(?:[\.\-·]\d+)?)\s*:\s*([^\n]{1,100})', texto_chunk)
                            for num, desc in matches:
                                if pg_chunk is not None:
                                    paginas_con_etiqueta.setdefault(pg_chunk, []).append({'numero': num, 'descripcion': desc.strip()})

                        # Rerank y asociar con imágenes
                        scroll_img = await client.scroll(
                            collection_name=self.gestor_qdrant.content_mv_collection,
                            scroll_filter=Filter(must=[FieldCondition(key="tipo", match={"value": "imagen"})]),
                            limit=1000
                        )
                        all_images = scroll_img[0] if scroll_img else []

                        paginas_usadas = set()
                        for pg, ets in paginas_con_etiqueta.items():
                            for et in ets:
                                for img_point in all_images:
                                    payload = img_point.payload or {}
                                    if payload.get('nombre_archivo') == doc_principal and payload.get('numero_pagina') == pg:
                                        img_path = payload.get('imagen_path', '')
                                        if img_path and os.path.exists(img_path) and pg not in paginas_usadas:
                                            imagenes_encontradas.append({
                                                "path": img_path,
                                                "descripcion": f"Imagen {et['numero']}: {et['descripcion']}",
                                                "caption_completo": payload.get('texto', '')
                                            })
                                            paginas_usadas.add(pg)
                    except Exception as e:
                        print(f"⚠️ Error buscando etiquetas: {e}")

                # Rerank: ordenar todas las imágenes candidatas por relevancia a la consulta del usuario
                if len(imagenes_encontradas) > 1:
                    consulta_lower = state['consulta_usuario'].lower()
                    consulta_tokens = set(consulta_lower.split())
                    
                    def _score_imagen(img):
                        """Score de similitud textual simple entre caption y consulta"""
                        desc = (img.get('descripcion', '') + ' ' + img.get('caption_completo', '')).lower()
                        # Coincidencia exacta de la consulta en la descripción
                        if consulta_lower in desc:
                            return 100
                        # Coincidencia de tokens individuales
                        desc_tokens = set(desc.split())
                        overlap = len(consulta_tokens & desc_tokens)
                        return overlap
                    
                    imagenes_encontradas.sort(key=_score_imagen, reverse=True)
                    print(f"   📊 Reranked {len(imagenes_encontradas)} imágenes candidatas:")
                    for idx, img in enumerate(imagenes_encontradas):
                        print(f"      {idx+1}. score={_score_imagen(img)} | {img['descripcion'][:80]}")

                if imagenes_encontradas:
                    state["imagenes_relevantes"] = imagenes_encontradas[:1]
                else:
                    # Fallback: buscar directamente imágenes por embedding si no hubo match por etiquetas
                    print("   🔍 No se encontraron imágenes por etiquetas, buscando directamente por embedding de imagen...")
                    resultados_img, _ = await self.gestor_qdrant.buscar_muvera_2stage(
                        query_mv, query_fde, min_score=0.0, filtro_tipo="imagen"
                    )
                    for r in resultados_img:
                        p = r.get('payload', {})
                        if p.get('tipo') == 'imagen' and p.get('imagen_path'):
                            img_path = p['imagen_path']
                            if os.path.exists(img_path):
                                caption = p.get('texto', '') or p.get('contexto_texto', '')[:300]
                                imagenes_encontradas.append({"path": img_path, "descripcion": caption})
                    if imagenes_encontradas:
                        state["imagenes_relevantes"] = imagenes_encontradas[:1]
                        print(f"   ✅ Encontrada imagen por embedding directo: {imagenes_encontradas[0]['path']}")
                    else:
                        state["imagenes_relevantes"] = []
                resultados = resultados_texto

        # CASE 3: Text query
        else:
            print("   🔍 Búsqueda local por texto...")
            query_mv = self.procesador.generar_embedding_texto(state['consulta_optimizada'])
            if query_mv is not None:
                query_fde = self.procesador.generar_query_muvera(query_mv)
                resultados, has_rejected = await self.gestor_qdrant.buscar_muvera_2stage(
                    query_mv, query_fde, min_score=0.0
                )
                resultados, _ = filtrar_resultados_busqueda(resultados, requiere_imagen=False, tiene_imagen_adjunta=False)

        state["resultados_busqueda"] = resultados
        state["abortar_reset"] = has_rejected

        if has_rejected or not resultados:
            state["contexto_documentos"] = ""
            state["imagenes_relevantes"] = []
        else:
            contextos = []
            imagenes = []
            for i, r in enumerate(resultados):
                score = r.get('score', 0.0)
                tipo = r['payload'].get('tipo', 'unknown')
                pdf_name = r['payload'].get('pdf_name', 'desconocido')
                page_num = r['payload'].get('numero_pagina', '?')

                if tipo == 'texto':
                    texto = r['payload'].get('texto', '')
                    contextos.append(f"[RESULTADO {i+1} - TEXTO - Score: {score:.2f} - Fuente: {pdf_name} (Pg {page_num})]\n{texto[:800]}")
                elif tipo == 'imagen':
                    img_path = r['payload'].get('imagen_path')
                    contexto_texto = r['payload'].get('contexto_texto', '')
                    caption = r['payload'].get('texto', '')
                    if img_path:
                        imagenes.append({"path": img_path, "descripcion": caption or contexto_texto[:300]})
                        contextos.append(f"[RESULTADO - IMAGEN - Score: {score:.2f} - Fuente: {pdf_name} (Pg {page_num})]\nArchivo: {os.path.basename(img_path)}\nTexto asociado: {contexto_texto[:600]}")

            state["contexto_documentos"] = "\n\n---\n\n".join(contextos)
            if not state.get("imagenes_relevantes"):
                state["imagenes_relevantes"] = imagenes[:1]

        state["trayectoria"].append({"nodo": "buscar", "timestamp": time.time()})
        return state

    def _decidir_camino_tras_busqueda(self, state: AgentState) -> str:
        requiere_imagen = state.get("requiere_imagen", False)
        imagenes_encontradas = len(state.get("imagenes_relevantes", [])) > 0
        
        # Conmutar a fallback si falló la verificación o no hay coincidencia visual/textual local
        if state.get("abortar_reset", False) or not state.get("resultados_busqueda") or (requiere_imagen and not imagenes_encontradas):
            return "fallback"
        return "generar"

    async def _nodo_reset(self, state: AgentState) -> AgentState:
        state["respuesta_final"] = "No está en la base de datos."
        state["imagenes_relevantes"] = []
        state["contexto_documentos"] = ""
        state["trayectoria"].append({"nodo": "reset", "timestamp": time.time()})
        return state

    async def _nodo_fallback_internet(self, state: AgentState) -> AgentState:
        """Nodo de Fallback: búsqueda en internet con Tavily al no encontrar coincidencias locales"""
        print("🌐 FALLBACK: Buscando en internet...")
        
        # 1. Analizar imagen visualmente con Llama-4-Scout si existe
        hallazgos_visuales = ""
        imagen_path = state.get("imagen_consulta")
        
        if imagen_path and os.path.exists(imagen_path):
            print("   📸 Extrayendo hallazgos de imagen para consulta web...")
            try:
                with open(imagen_path, "rb") as img_f:
                    img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
                
                content = [
                    {
                        "type": "text",
                        "text": "Describe detalladamente los hallazgos de esta imagen médica: tejidos, anomalías o patrones de interés clínico para buscar información en la web."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
                res = await self.llm.ainvoke([HumanMessage(content=content)])
                hallazgos_visuales = res.content
            except Exception as e:
                print(f"   ⚠️ Error de visión en fallback: {e}")
                hallazgos_visuales = "Error en análisis visual de la imagen."

        # 2. Generar query para Tavily
        system_query_prompt = """Eres un experto en búsqueda médica. Genera una consulta de búsqueda precisa y corta en inglés o español basada en los hallazgos de la imagen y la consulta. Responde SOLO con la query."""
        user_query_prompt = f"Consulta del usuario: {state['consulta_usuario']}\nHallazgos visuales: {hallazgos_visuales}"
        
        try:
            messages = [
                SystemMessage(content=system_query_prompt),
                HumanMessage(content=user_query_prompt)
            ]
            res = await self.llm.ainvoke(messages)
            search_query = res.content.strip()
        except Exception:
            search_query = state.get("consulta_optimizada") or state["consulta_usuario"]

        # 3. Realizar búsqueda en internet
        print(f"   🔎 Buscando en Tavily: '{search_query}'")
        search_info = ""
        try:
            result = await asyncio.to_thread(self.tavily_tool.invoke, {"query": search_query})
            if isinstance(result, list) and len(result) > 0:
                parts = []
                for idx, r in enumerate(result[:3]):
                    ans = r.get('answer') or r.get('content') or ''
                    url = r.get('url') or ''
                    parts.append(f"[{idx+1}] Fuente ({url}): {ans}")
                search_info = "\n\n".join(parts)
            else:
                search_info = "No se encontraron resultados relevantes en la web."
        except Exception as e:
            search_info = f"Error de búsqueda web: {e}"

        # 4. Generar respuesta final de fallback
        system_response_prompt = """Eres un médico especialista que responde consultas.
Dado que la consulta o la imagen NO coinciden con los manuales locales de histopatología, realizaste una búsqueda en la web.
1. Aclara amablemente al usuario que no se encontró el caso en la base de datos local y que buscaste en internet.
2. Proporciona una explicación estructurada basada en los hallazgos visuales (si aplica) y la búsqueda web.
3. Agrega disclaimers obligatorios aclarando que esto no sustituye una consulta médica real.
4. Cita las fuentes (URLs) encontradas."""

        user_response_prompt = f"""CONSULTA DEL USUARIO: {state['consulta_usuario']}
HISTORIAL: {state.get('contexto_memoria', '')}
HALLAZGOS VISUALES: {hallazgos_visuales}
RESULTADOS DE BÚSQUEDA WEB:
{search_info}"""

        try:
            res = await self.llm.ainvoke([
                SystemMessage(content=system_response_prompt),
                HumanMessage(content=user_response_prompt)
            ])
            state["respuesta_final"] = res.content
        except Exception as e:
            state["respuesta_final"] = f"Error generando respuesta de fallback: {e}"

        state["imagenes_relevantes"] = []
        state["contexto_documentos"] = f"[Búsqueda en Internet]\n{search_info}"
        state["trayectoria"].append({"nodo": "fallback_internet", "timestamp": time.time()})
        return state

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        """Nodo de generación para RAG local"""
        print("   📝 Generando respuesta basada en RAG local...")
        
        system_prompt = """Eres un profesor experto en histopatología.
Responde basándote EXCLUSIVAMENTE en el contexto textual e imágenes recuperadas de la base de datos local.
Usa terminología precisa y agrega disclaimers médicos al final.
Si el usuario subió una imagen, la imagen recuperada es la mejor coincidencia del manual."""

        # Adjuntar imágenes locales en base64 si existen
        user_content = []
        historial = f"\nHISTORIAL:\n{state.get('contexto_memoria', '')}\n" if state.get("contexto_memoria") else ""
        
        texto_mensaje = f"""{historial}CONSULTA DEL USUARIO: {state["consulta_usuario"]}
========================================
CONTEXTO LOCAL RECUPERADO:
{state["contexto_documentos"][:10000]}
========================================
Responde basándote en el contexto anterior."""

        user_content.append({"type": "text", "text": texto_mensaje})

        # Adjuntar las imágenes recuperadas del manual
        for i, img_item in enumerate(state.get("imagenes_relevantes", [])[:2]):
            try:
                img_path = img_item["path"]
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_f:
                        img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    
                    user_content.append({
                        "type": "text",
                        "text": f"\n[IMAGEN DEL MANUAL RECUPERADA {i+1}: {os.path.basename(img_path)}]"
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
            except Exception as e:
                print(f"⚠️ Error cargando imagen para prompt: {e}")

        try:
            res = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            state["respuesta_final"] = res.content
        except Exception as e:
            state["respuesta_final"] = f"Error en generación local: {e}"

        state["trayectoria"].append({"nodo": "generar_respuesta", "timestamp": time.time()})
        return state

    async def _nodo_finalizar(self, state: AgentState) -> AgentState:
        resp = state.get("respuesta_final", "")
        query = state.get("consulta_usuario", "")
        
        if self.memoria is not None and resp and query:
            try:
                messages = [
                    SystemMessage(content="Resume la consulta y la respuesta en 1-2 oraciones breves para la memoria."),
                    HumanMessage(content=f"USUARIO: {query}\nASISTENTE: {resp[:1000]}")
                ]
                res = await self.llm.ainvoke(messages)
                summary = res.content
                self.memoria.add_interaction_summary(
                    session_id=state["user_id"],
                    user_query=query,
                    summary=f"Consulta: {query} | Resumen: {summary}"
                )
            except Exception as e:
                print(f"⚠️ Error memoria: {e}")
        state["trayectoria"].append({"nodo": "finalizar", "timestamp": time.time()})
        return state

    # ========== MÉTODOS DE PROCESAMIENTO PÚBLICOS ==========

    async def procesar_pdfs(self, archivos: List[str]):
        """Procesa PDFs y los indexa en Qdrant"""
        self.inicializar_componentes()
        await self.gestor_qdrant.crear_colecciones()
        
        for archivo in archivos:
            if not os.path.exists(archivo):
                continue
            
            print(f"📦 Indexando PDF: {archivo}")
            reader = PdfReader(archivo)
            paginas_info = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    paginas_info.append({"texto": txt, "pagina": i + 1})

            chunks_info = []
            size = Config.TEXT_CHUNK_SIZE
            overlap = Config.TEXT_CHUNK_OVERLAP
            for pag in paginas_info:
                texto = pag["texto"]
                num_pag = pag["pagina"]
                if len(texto) < size:
                    chunks_info.append({"texto": texto, "pagina": num_pag})
                else:
                    for idx in range(0, len(texto), size - overlap):
                        chunks_info.append({"texto": texto[idx:idx + size], "pagina": num_pag})

            texto_completo = "\n".join([p["texto"] for p in paginas_info])
            imagenes = self.procesador.extraer_imagenes_pdf(archivo)
            
            if not self.ontologia:
                self.ontologia = self.extractor_ontologia.extraer_ontologia_completa(texto_completo, len(imagenes))

            # Subir texto
            await self._procesar_contenido_batch(chunks_info, None, archivo, tipo="texto")
            # Subir imágenes
            await self._procesar_contenido_batch(chunks_info, imagenes, archivo, tipo="imagen")
            cleanup_memory()

    async def _procesar_contenido_batch(self, chunks_info, imagenes, pdf_name, tipo="texto"):
        items = chunks_info if tipo == "texto" else imagenes
        if not items:
            return
            
        batch_mv, batch_fde = [], []
        for i, item in enumerate(items):
            if tipo == "texto":
                contenido = item["texto"]
                page_num = item["pagina"]
                mv_embedding = self.procesador.generar_embedding_texto(contenido)
                payload = {
                    "pdf_name": str(pdf_name),
                    "tipo": "texto",
                    "texto": contenido[:500],
                    "numero_pagina": page_num,
                    "figuras": self._extraer_figuras_de_texto(contenido),
                    "nombre_archivo": Path(pdf_name).stem
                }
            else:
                contenido = item["path"]
                page_num = item["page"]
                mv_embedding = self.procesador.generar_embedding_imagen(contenido)
                
                contexto_texto = ""
                figura_asignada = []
                if chunks_info:
                    chunks_pag = [c["texto"] for c in chunks_info if c["pagina"] == page_num]
                    if chunks_pag:
                        contexto_texto = " ".join(chunks_pag)
                    figuras_en_pagina = self._extraer_figuras_de_texto(contexto_texto)
                    img_idx = item.get("img_index_in_page", 0)
                    total_imgs = item.get("total_images_in_page", 1)
                    
                    if figuras_en_pagina and total_imgs > 1:
                        figuras_ordenadas = sorted(figuras_en_pagina)
                        if img_idx < len(figuras_ordenadas):
                            figura_asignada = [figuras_ordenadas[img_idx]]
                        else:
                            figura_asignada = figuras_en_pagina
                    else:
                        figura_asignada = figuras_en_pagina

                payload = {
                    "pdf_name": str(pdf_name),
                    "tipo": "imagen",
                    "texto": item.get("caption", ""),
                    "imagen_path": contenido,
                    "contexto_texto": contexto_texto[:1000],
                    "numero_pagina": page_num,
                    "figuras": figura_asignada,
                    "nombre_archivo": Path(pdf_name).stem
                }

            if mv_embedding is None:
                continue
            
            fde_embedding = self.procesador.generar_fde_muvera(mv_embedding)
            seed_id = f"{Path(pdf_name).stem}_{tipo}_{page_num}_{i}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_id))

            batch_mv.append(PointStruct(id=point_id, vector=mv_embedding.tolist(), payload=payload))
            batch_fde.append(PointStruct(id=point_id, vector=fde_embedding.tolist(), payload=payload))

            if len(batch_mv) >= Config.BATCH_SIZE:
                await self.gestor_qdrant.insertar_batch_muvera(batch_mv, batch_fde)
                batch_mv, batch_fde = [], []
                cleanup_memory()

        if batch_mv:
            await self.gestor_qdrant.insertar_batch_muvera(batch_mv, batch_fde)

    # ========== MÉTODOS DE INTEGRACIÓN CON AGENT SDK ==========

    async def invoke(self, query: str, context_id: str, images: list[dict] = None) -> str:
        """Compatibilidad con ejecución directa"""
        final_text = ""
        async for chunk in self.stream(query, context_id, images):
            if chunk.get('is_task_complete'):
                final_text = chunk.get('content', '')
        return final_text

    async def stream(self, query: str, context_id: str, images: list[dict] = None) -> AsyncIterable[dict[str, Any]]:
        """Streaming de eventos y respuestas para el Agent SDK UI"""
        self.inicializar_componentes()

        # Extraer imagen base64 si viene de la parte de entrada
        imagen_base64 = None
        if images and len(images) > 0:
            img = images[0]
            img_data = img.get('data') or img.get('bytes')
            if isinstance(img_data, bytes):
                imagen_base64 = base64.b64encode(img_data).decode('utf-8')
            elif isinstance(img_data, str):
                imagen_base64 = img_data

        initial_state = AgentState(
            messages=[],
            consulta_usuario=query,
            consulta_resuelta="",
            imagen_consulta=None,
            imagen_base64=imagen_base64,
            contexto_memoria="",
            ontologia=self.ontologia or {},
            contexto_ontologico="",
            clasificacion="",
            requiere_imagen=False,
            consulta_optimizada="",
            filtros_ontologia=[],
            resultados_busqueda=[],
            contexto_documentos="",
            imagenes_relevantes=[],
            respuesta_final="",
            trayectoria=[],
            user_id=context_id,
            tiempo_inicio=time.time(),
            abortar_reset=False
        )

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🏥 Recepcionando consulta médica...',
            'status': 'analyzing_images'
        }

        # 1. Recepcionar
        state = await self._nodo_recepcionar_consulta(initial_state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🧠 Inicializando memoria y contexto...',
            'status': 'analyzing_images'
        }

        # 2. Inicializar
        state = await self._nodo_inicializar(state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔬 Analizando ontología histológica...',
            'status': 'classifying'
        }

        # 3. Analizar Ontología
        state = await self._nodo_analizar_ontologia(state)

        # 4. Clasificar consulta
        state = await self._nodo_clasificar(state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔎 Optimizando consulta para búsqueda...',
            'status': 'searching'
        }

        # 5. Optimizar
        state = await self._nodo_optimizar_consulta(state)

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔍 Buscando en base de datos local (ColPali)...',
            'status': 'searching'
        }

        # 6. Buscar localmente
        state = await self._nodo_buscar(state)

        # 7. Decidir camino: Local RAG o Fallback
        camino = self._decidir_camino_tras_busqueda(state)

        if camino == "generar":
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '📝 Generando respuesta basada en el manual local...',
                'status': 'generating_response'
            }
            state = await self._nodo_generar_respuesta(state)
        else:
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🌐 Sin coincidencia local confiable. Conmutando a fallback de internet...',
                'status': 'searching'
            }
            state = await self._nodo_fallback_internet(state)

        # 8. Finalizar
        state = await self._nodo_finalizar(state)

        # Generar respuesta final con Markdown
        final_answer = state["respuesta_final"]
        
        # Si se recuperaron imágenes locales válidas, podemos añadir su enlace al final del Markdown para visualización
        if camino == "generar" and state.get("imagenes_relevantes"):
            final_answer += "\n\n### Micrografías del Manual Relacionadas:\n"
            for i, img in enumerate(state["imagenes_relevantes"]):
                # Servir la imagen mediante la ruta montada
                img_name = os.path.basename(img["path"])
                img_url = f"http://localhost:10002/histopatologia_data/embeddings/{img_name}"
                desc = img.get("descripcion", f"Figura {i+1}")
                final_answer += f"![{desc}]({img_url})\n\n*{desc}*\n"

        cleanup_memory()

        yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': final_answer,
            'status': 'completed',
            'imagenes_relevantes': state.get("imagenes_relevantes", [])
        }
