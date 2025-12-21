# samples/python/agents/palimodal/app/agent.py
# Agente de Física usando ColPali + MUVERA para embeddings multi-vector

import asyncio
import base64
import glob
import os
from collections.abc import AsyncIterable
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    PointStruct,
    VectorParams,
)

# ==================== CONFIGURACIÓN ====================

GEMINI_MODEL = "gemini-2.5-flash"
COLPALI_MODEL = "vidore/colqwen2-v1.0"  # Modelo ColPali recomendado
MUVERA_TARGET_DIM = 2048  # Dimensión del vector MUVERA FDE
COLPALI_EMBEDDING_DIM = 128  # Dimensión de cada patch embedding

# ==================== MUVERA IMPLEMENTATION ====================

class MuveraProcessor:
    """
    Implementación de MUVERA para comprimir multi-vectors a un vector fijo.
    
    MUVERA (Multi-Vector Retrieval Algorithm) convierte embeddings multi-vector
    en representaciones de dimensión fija (FDE) para búsqueda rápida.
    """
    
    def __init__(self, target_dim: int = 2048, num_projections: int = 16):
        self.target_dim = target_dim
        self.num_projections = num_projections
        self.bucket_size = target_dim // num_projections
        self._projection_matrix = None
        
    def _get_projection_matrix(self, input_dim: int) -> np.ndarray:
        """Genera matriz de proyección aleatoria (determinística por seed)."""
        if self._projection_matrix is None or self._projection_matrix.shape[1] != input_dim:
            np.random.seed(42)  # Seed fija para reproducibilidad
            self._projection_matrix = np.random.randn(self.num_projections, input_dim)
            # Normalizar
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=1, keepdims=True)
        return self._projection_matrix
    
    def compress(self, multi_vectors: np.ndarray) -> List[float]:
        """
        Comprime multi-vectors a un vector FDE de dimensión fija.
        
        Args:
            multi_vectors: Array de shape (num_vectors, embedding_dim)
            
        Returns:
            Vector FDE de longitud target_dim
        """
        if len(multi_vectors.shape) == 1:
            multi_vectors = multi_vectors.reshape(1, -1)
            
        num_vectors, embedding_dim = multi_vectors.shape
        proj_matrix = self._get_projection_matrix(embedding_dim)
        
        # Inicializar FDE con ceros
        fde = np.zeros(self.target_dim)
        
        # Para cada vector en el multi-vector
        for vec in multi_vectors:
            # Calcular bucket assignments usando proyección
            projections = proj_matrix @ vec
            bucket_indices = np.argmax(projections)
            
            # Agregar el vector al bucket correspondiente
            start_idx = bucket_indices * self.bucket_size
            end_idx = start_idx + self.bucket_size
            
            # Reducir dimensión del vector para caber en el bucket
            if embedding_dim > self.bucket_size:
                reduced = vec[:self.bucket_size]
            else:
                reduced = np.pad(vec, (0, self.bucket_size - embedding_dim))
            
            # Acumular en el bucket (MaxPool style)
            fde[start_idx:end_idx] = np.maximum(fde[start_idx:end_idx], reduced)
        
        # Normalizar
        norm = np.linalg.norm(fde)
        if norm > 0:
            fde = fde / norm
            
        return fde.tolist()


# ==================== MEMORIA SEMÁNTICA ====================

class SemanticMemory:
    """Memoria conversacional simplificada."""
    
    def __init__(self, llm=None, max_entries: int = 10):
        self.conversations = []
        self.max_entries = max_entries
        self.summary = ""
        self.direct_history = ""
    
    def add_interaction(self, query: str, response: str):
        """Guardar interacción en memoria."""
        self.conversations.append({"query": query, "response": response})
        
        if len(self.conversations) > self.max_entries:
            self.conversations.pop(0)
        
        self.direct_history += f"\nUsuario: {query}\nAsistente: {response}\n"
        
        if len(self.conversations) > 3:
            recent = self.conversations[-3:]
            self.direct_history = ""
            for conv in recent:
                self.direct_history += f"\nUsuario: {conv['query']}\nAsistente: {conv['response']}\n"
        
        self.update_summary()
    
    def update_summary(self):
        """Actualizar resumen."""
        if self.conversations:
            recent_context = "\n".join([
                f"Q: {c['query']}\nA: {c['response']}"
                for c in self.conversations[-5:]
            ])
            self.summary = f"Resumen:\n{recent_context}"
    
    def get_context(self) -> str:
        """Obtener contexto completo."""
        return f"{self.summary}\n\nHistorial:\n{self.direct_history}"
    
    def clear(self):
        """Limpiar memoria."""
        self.conversations = []
        self.summary = ""
        self.direct_history = ""


# ==================== AGENTE PALIMODAL ====================

class PhysicsPalimodalAgent:
    """Agente de física con ColPali multi-vector embeddings y MUVERA."""
    
    SYSTEM_INSTRUCTION = (
        'Eres un profesor experto en Física I de la UBA. '
        'Analizas consultas, imágenes de experimentos y proporciona '
        'explicaciones claras y didácticas.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        """Inicializar el agente de física con ColPali."""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=4096,
        )
        
        # Qdrant
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_KEY", "")
        
        # Colecciones Palimodal
        self.muvera_collection = "palimodal_muvera"      # FDEs para búsqueda rápida
        self.multivector_collection = "palimodal_mv"     # Multi-vectors para re-ranking
        
        # MUVERA processor
        self.muvera = MuveraProcessor(target_dim=MUVERA_TARGET_DIM)
        
        # Modelo ColPali (carga lazy)
        self._colpali_model = None
        self._colpali_processor = None
        self._device = None
        
        # Memoria conversacional
        self.memories = {}
        self.visual_findings = {}
        self.temario = ""

        # Cliente Qdrant persistente
        self._qdrant_client = None
        
        print("✅ PhysicsPalimodalAgent inicializado (ColPali se cargará cuando sea necesario)")

    @property
    def qdrant_client(self) -> AsyncQdrantClient:
        """Cliente Qdrant singleton lazy."""
        if self._qdrant_client is None:
            self._qdrant_client = AsyncQdrantClient(
                url=self.qdrant_url, 
                api_key=self.qdrant_api_key,
                # Supress warning for minor version mismatch
                check_compatibility=False
            )
        return self._qdrant_client
    
    def _load_colpali(self):
        """Carga el modelo ColPali de forma lazy."""
        if self._colpali_model is None:
            print(f"🔄 Cargando modelo ColPali: {COLPALI_MODEL}...")
            
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            
            # Forzar CPU para evitar errores de compatibilidad CUDA
            # Si quieres usar GPU, asegúrate de tener la versión correcta de PyTorch+CUDA
            use_cuda = os.getenv("USE_CUDA", "false").lower() == "true"
            
            if use_cuda and torch.cuda.is_available():
                self._device = "cuda:0"
                dtype = torch.bfloat16
                print(f"⚠️ Usando CUDA - si hay errores, usa USE_CUDA=false")
            else:
                self._device = "cpu"
                dtype = torch.float32
                print(f"💻 Usando CPU (más lento pero compatible)")
            
            try:
                self._colpali_model = ColQwen2.from_pretrained(
                    COLPALI_MODEL,
                    torch_dtype=dtype,
                    device_map=self._device,
                ).eval()
                
                self._colpali_processor = ColQwen2Processor.from_pretrained(COLPALI_MODEL)
                
                print(f"✅ ColPali cargado en {self._device}")
            except Exception as e:
                print(f"❌ Error cargando ColPali: {e}")
                if "CUDA" in str(e) and self._device != "cpu":
                    print("🔄 Reintentando con CPU...")
                    self._device = "cpu"
                    self._colpali_model = ColQwen2.from_pretrained(
                        COLPALI_MODEL,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                    ).eval()
                    self._colpali_processor = ColQwen2Processor.from_pretrained(COLPALI_MODEL)
                    print(f"✅ ColPali cargado en CPU (fallback)")
                else:
                    raise
    
    @property
    def colpali_model(self):
        self._load_colpali()
        return self._colpali_model
    
    @property
    def colpali_processor(self):
        self._load_colpali()
        return self._colpali_processor
    
    @property
    def device(self):
        self._load_colpali()
        return self._device
    
    # ==================== GENERACIÓN DE EMBEDDINGS ====================
    
    def generate_document_embedding(self, image: Image.Image) -> tuple[np.ndarray, List[float]]:
        """
        Genera embeddings para un documento/imagen.
        
        Returns:
            Tuple de (multi_vectors, muvera_fde)
            - multi_vectors: Array de shape (num_patches, 128)
            - muvera_fde: Vector comprimido de dimensión MUVERA_TARGET_DIM
        """
        batch = self.colpali_processor.process_images([image]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.colpali_model(**batch)
        
        # Convertir a numpy
        multi_vectors = embeddings[0].cpu().numpy()  # Shape: (num_patches, 128)
        
        # Comprimir con MUVERA
        muvera_fde = self.muvera.compress(multi_vectors)
        
        return multi_vectors, muvera_fde
    
    def generate_query_embedding(self, query: str) -> tuple[np.ndarray, List[float]]:
        """
        Genera embeddings para una query.
        
        Returns:
            Tuple de (multi_vectors, muvera_fde)
        """
        batch = self.colpali_processor.process_queries([query]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.colpali_model(**batch)
        
        multi_vectors = embeddings[0].cpu().numpy()
        muvera_fde = self.muvera.compress(multi_vectors)
        
        return multi_vectors, muvera_fde
    
    # ==================== PROCESAMIENTO DE PDFs ====================
    
    def leer_pdf(self, archivo: str) -> str:
        """Leer texto de un PDF."""
        try:
            reader = PdfReader(archivo)
            return "".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")
            return ""
    
    def extraer_imagenes_pdf(self, pdf_path: str, output_folder: str = "extracted_images") -> List[str]:
        """Extraer imágenes de un PDF."""
        os.makedirs(output_folder, exist_ok=True)
        imagenes = []
        
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(pdf_path, dpi=150)
            
            for page_num, page in enumerate(pages):
                img_path = os.path.join(
                    output_folder,
                    f"{Path(pdf_path).stem}_page{page_num}.png"
                )
                page.save(img_path, 'PNG')
                imagenes.append(img_path)
            
            print(f"✅ Extraídas {len(imagenes)} páginas de {Path(pdf_path).name}")
            return imagenes
        except Exception as e:
            print(f"❌ Error extrayendo imágenes: {e}")
            return []
    
    async def setup_collections(self):
        """Configura las colecciones de Qdrant para ColPali + MUVERA."""
        # client = self.qdrant_client  <-- Accedido via property
        
        # Colección MUVERA (FDE para búsqueda rápida)
        try:
            await self.qdrant_client.get_collection(self.muvera_collection)
            print(f"📦 Colección '{self.muvera_collection}' ya existe")
        except Exception:
            await self.qdrant_client.create_collection(
                collection_name=self.muvera_collection,
                vectors_config=VectorParams(
                    size=MUVERA_TARGET_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"✨ Colección '{self.muvera_collection}' creada")
        
        # Colección Multi-Vector (para re-ranking con MaxSim)
        try:
            await self.qdrant_client.get_collection(self.multivector_collection)
            print(f"📦 Colección '{self.multivector_collection}' ya existe")
        except Exception:
            await self.qdrant_client.create_collection(
                collection_name=self.multivector_collection,
                vectors_config=VectorParams(
                    size=COLPALI_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    )
                )
            )
            print(f"✨ Colección '{self.multivector_collection}' creada (MaxSim)")
    
    async def store_document(self, doc_id: int, image: Image.Image, metadata: dict):
        """Almacena un documento en ambas colecciones."""
        # client = self.qdrant_client
        
        # Generar embeddings
        multi_vectors, muvera_fde = self.generate_document_embedding(image)
        
        # Almacenar FDE en colección MUVERA
        await self.qdrant_client.upsert(
            collection_name=self.muvera_collection,
            points=[PointStruct(
                id=doc_id,
                vector=muvera_fde,
                payload=metadata
            )],
            wait=True
        )
        
        # Almacenar multi-vectors
        await self.qdrant_client.upsert(
            collection_name=self.multivector_collection,
            points=[PointStruct(
                id=doc_id,
                vector=multi_vectors.tolist(),  # Lista de vectores
                payload=metadata
            )],
            wait=True
        )
    
    async def extraer_temario(self, contenido_completo: str) -> str:
        """Extraer temario con Gemini."""
        print("🤖 Extrayendo temario...")
        contenido_limitado = contenido_completo[:15000]
        
        system_message = f"""Eres un profesor de Física I de la UBA.
Extrae el TEMARIO COMPLETO del contenido.

Formato:
TEMA 1: [Título]
- Subtema 1.1: [Descripción]

Contenido:
---
{contenido_limitado}
---
"""
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content="Extrae el temario estructurado."),
        ]
        
        try:
            response = self.llm.invoke(messages)
            print(f"✅ Temario extraído ({len(response.content)} caracteres)")
            return response.content
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return "Temario no disponible."
    
    async def procesar_y_almacenar_pdfs(self, pdf_files: List[str]) -> str:
        """Procesar PDFs y almacenar en Qdrant con ColPali + MUVERA."""
        print("\n" + "="*80)
        print("📚 PROCESANDO PDFs con ColPali + MUVERA")
        print("="*80)
        
        # Setup colecciones
        await self.setup_collections()
        
        global_id_counter = 0
        contenido_completo_texto = ""
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
                continue
            
            print(f"\n📄 Procesando: {Path(pdf_file).name}")
            
            # Extraer texto para temario
            text = self.leer_pdf(pdf_file)
            if text:
                contenido_completo_texto += f"\n--- {Path(pdf_file).name} ---\n{text}"
            
            # Extraer imágenes de páginas
            imagenes = self.extraer_imagenes_pdf(pdf_file)
            print(f"   🖼️ {len(imagenes)} páginas")
            
            for img_path in imagenes:
                try:
                    image = Image.open(img_path).convert("RGB")
                    
                    await self.store_document(
                        doc_id=global_id_counter,
                        image=image,
                        metadata={
                            "pdf_name": pdf_file,
                            "page_image": img_path,
                            "type": "page"
                        }
                    )
                    global_id_counter += 1
                    
                except Exception as e:
                    print(f"   ❌ Error procesando {img_path}: {e}")
        
        # Extraer temario
        temario = await self.extraer_temario(contenido_completo_texto)
        
        print("\n✅ PROCESAMIENTO COMPLETADO")
        print(f"   📄 Documentos indexados: {global_id_counter}")
        
        self.temario = temario
        return temario
    
    # ==================== BÚSQUEDA ====================
    
    async def search_colpali(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Búsqueda usando MUVERA para candidatos + MaxSim para re-ranking.
        
        1. Búsqueda rápida con MUVERA FDE (top-100 candidatos)
        2. Re-ranking con multi-vectors originales usando MaxSim
        """
        from qdrant_client.models import QueryRequest
        
        # client = self.qdrant_client
        
        # Generar query embeddings
        query_mv, query_fde = self.generate_query_embedding(query)
        
        # Paso 1: Búsqueda rápida con MUVERA
        print(f"🔎 Búsqueda MUVERA (Top-100 candidatos)...")
        candidates_response = await self.qdrant_client.query_points(
            collection_name=self.muvera_collection,
            query=query_fde,
            limit=100
        )
        candidates = candidates_response.points
        
        if not candidates:
            print("⚠️ No se encontraron candidatos")
            return []
        
        # Paso 2: Re-ranking con multi-vectors
        print(f"🎯 Re-ranking con MaxSim (Top-{top_k})...")
        candidate_ids = [c.id for c in candidates]
        
        # Búsqueda en colección multi-vector
        final_response = await self.qdrant_client.query_points(
            collection_name=self.multivector_collection,
            query=query_mv.tolist(),
            limit=top_k
        )
        final_results = final_response.points
        
        results = [{
            "id": r.id,
            "score": round(r.score, 4),
            "payload": r.payload
        } for r in final_results]
        
        print(f"✅ {len(results)} resultados encontrados")
        return results
    
    # ==================== ANÁLISIS DE IMÁGENES ====================
    
    def encode_image(self, image_data: bytes) -> str:
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_mime_type(self, content_type: str) -> str:
        mapping = {
            'image/jpeg': 'image/jpeg',
            'image/png': 'image/png',
            'image/webp': 'image/webp',
        }
        return mapping.get(content_type, 'image/png')
    
    async def analyze_physics_image(self, images: List[dict]) -> str:
        """Analiza imágenes de física."""
        if not images:
            return "No se proporcionaron imágenes."
        
        content = [{
            "type": "text",
            "text": f"""Analiza estas {len(images)} imágenes de física:

1. FENÓMENO FÍSICO observado
2. PRINCIPIOS FÍSICOS aplicables
3. ECUACIONES RELEVANTES
4. DESCRIPCIÓN DETALLADA

Sé técnico y preciso."""
        }]
        
        for idx, img in enumerate(images):
            try:
                image_data_raw = img.get('data') or img.get('bytes')
                if isinstance(image_data_raw, bytes):
                    image_data_b64 = self.encode_image(image_data_raw)
                elif isinstance(image_data_raw, str):
                    image_data_b64 = image_data_raw
                else:
                    continue
                
                mime_type = self.get_mime_type(img.get('mime_type', 'image/png'))
                content.append({
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_data_b64}"
                })
            except Exception as e:
                print(f"❌ Error imagen {idx}: {e}")
        
        try:
            response = self.llm.invoke([HumanMessage(content=content)])
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ==================== MEMORIA ====================
    
    def _get_or_create_memory(self, context_id: str) -> SemanticMemory:
        if context_id not in self.memories:
            self.memories[context_id] = SemanticMemory(llm=self.llm)
        return self.memories[context_id]
    
    def _get_memory_context(self, context_id: str) -> str:
        memory = self._get_or_create_memory(context_id)
        return memory.get_context()
    
    def _save_to_memory(self, context_id: str, query: str, response: str):
        memory = self._get_or_create_memory(context_id)
        memory.add_interaction(query, response)
    
    # ==================== CLASIFICACIÓN Y GENERACIÓN ====================
    
    async def classify_query(self, query: str, context: str, visual_findings: str) -> str:
        """Clasifica la consulta."""
        system_prompt = f"""Profesor de Física I.

TEMARIO:
{self.temario}

Identifica:
1. Tema del temario
2. Subtemas relevantes
3. Palabras clave
4. Tipo de contenido (texto/imagen)

Formato:
TEMA: [número y título]
SUBTEMAS: [lista]
KEYWORDS: [palabras clave]
TIPO_CONTENIDO: [texto/imagen/ambos]
"""
        
        user_prompt = f"""
HALLAZGOS VISUALES:
{visual_findings}

CONTEXTO:
{context}

CONSULTA:
{query}

Clasifica según el temario."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def generate_search_query(self, classification: str, visual_findings: str, 
                                   original_query: str) -> str:
        """Genera consulta de búsqueda."""
        system_prompt = """Experto en búsqueda física.
Genera consulta precisa y técnica.
SOLO la consulta, sin explicaciones."""
        
        user_prompt = f"""
CLASIFICACIÓN:
{classification}

HALLAZGOS:
{visual_findings}

CONSULTA:
{original_query}

Genera consulta optimizada."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def generate_physics_response(
        self, 
        query: str, 
        context: str, 
        classification: str, 
        visual_findings: str,
        document_context: str
    ) -> str:
        """Genera respuesta final."""
        system_prompt = f"""Profesor de Física I UBA.

TEMARIO:
{self.temario}

Estructura:
1. CONTEXTO DEL TEMA
2. EXPLICACIÓN TEÓRICA
3. ANÁLISIS DE IMÁGENES
4. ECUACIONES
5. EJEMPLOS
6. RESUMEN

Reglas:
- Técnico pero claro
- Relacionar con temario
- Conectar imágenes con teoría
- Incluir ecuaciones
"""
        
        user_prompt = f"""
CONSULTA:
{query}

CONTEXTO:
{context}

CLASIFICACIÓN:
{classification}

HALLAZGOS:
{visual_findings}

DOCUMENTOS RELEVANTES:
{document_context}

Explicación completa."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ==================== MÉTODOS PRINCIPALES ====================
    
    async def invoke(self, query: str, context_id: str, 
                    images: List[dict] = None) -> str:
        """Procesa consulta completa."""
        print(f"\n{'='*80}")
        print(f"📚 Consulta de física (ColPali + MUVERA)")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        try:
            memory_context = self._get_memory_context(context_id)
            
            # Analizar imágenes
            visual_findings = ""
            
            if images and len(images) > 0:
                print(f"🖼️ Analizando imágenes...")
                visual_findings = await self.analyze_physics_image(images)
                self.visual_findings[context_id] = visual_findings
            else:
                visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
            
            # Clasificar
            print(f"🔍 Clasificando...")
            classification = await self.classify_query(query, memory_context, visual_findings)
            
            # Buscar con ColPali + MUVERA
            print(f"🔎 Buscando con ColPali...")
            search_query = await self.generate_search_query(
                classification, visual_findings, query
            )
            search_results = await self.search_colpali(search_query, top_k=5)
            
            # Contexto de documentos
            document_context = "\n".join([
                f"--- Documento {i+1} (score: {r['score']}) ---\n{r['payload']}"
                for i, r in enumerate(search_results)
            ])
            
            # Respuesta
            print(f"📝 Generando respuesta...")
            final_response = await self.generate_physics_response(
                query, memory_context, classification, visual_findings,
                document_context
            )
            
            self._save_to_memory(context_id, query, final_response)
            print(f"✅ Completado\n")
            
            return final_response
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"
    
    async def stream(self, query: str, context_id: str, 
                    images: List[dict] = None) -> AsyncIterable[dict[str, Any]]:
        """
        Streaming del agente.
        Yielda diccionarios con:
        - 'is_task_complete': bool
        - 'require_user_input': bool
        - 'content': str
        - 'status': str
        """
        print(f"\n{'='*80}")
        print(f"📚 Consulta (streaming) ColPali + MUVERA")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        memory_context = self._get_memory_context(context_id)
        
        # Analizar imágenes
        visual_findings = ""
        
        if images and len(images) > 0:
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f'🖼️ Analizando {len(images)} imagen(es)...',
                'status': 'analyzing_images'
            }
            
            visual_findings = await self.analyze_physics_image(images)
            self.visual_findings[context_id] = visual_findings
            
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '✅ Fenómenos físicos identificados.',
                'status': 'analyzing_images'
            }
        else:
            visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
        
        # Clasificar
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '📚 Clasificando según el temario...',
            'status': 'classifying'
        }
        
        classification = await self.classify_query(query, memory_context, visual_findings)
        
        # Buscar con ColPali
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔎 Buscando con ColPali + MUVERA...',
            'status': 'searching_documents'
        }
        
        search_query = await self.generate_search_query(
            classification, visual_findings, query
        )
        search_results = await self.search_colpali(search_query, top_k=5)
        
        # Contexto
        document_context = "\n".join([
            f"--- Documento {i+1} (score: {r['score']}) ---\n{r['payload']}"
            for i, r in enumerate(search_results)
        ])
        
        # Respuesta
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '📝 Generando respuesta final...',
            'status': 'generating_response'
        }
        
        final_response = await self.generate_physics_response(
            query, memory_context, classification, visual_findings,
            document_context
        )
        
        self._save_to_memory(context_id, query, final_response)
        
        # Yield final response
        yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': final_response,
            'status': 'completed'
        }
    
    async def clear_memory(self, context_id: str):
        """Limpia la memoria de un contexto específico."""
        if context_id in self.memories:
            self.memories[context_id].clear()
            del self.memories[context_id]
        if context_id in self.visual_findings:
            del self.visual_findings[context_id]
        print(f"🧹 Memoria limpiada para contexto: {context_id}")
    
    async def get_memory_summary(self, context_id: str) -> str:
        """Obtiene un resumen de la memoria de un contexto."""
        if context_id not in self.memories:
            return "No hay memoria para este contexto."
        
        memory = self.memories[context_id]
        return f"""
📊 **Resumen de Memoria (Palimodal)**
- Interacciones guardadas: {len(memory.conversations)}
- Contexto disponible: {'Sí' if memory.get_context() else 'No'}
- Hallazgos visuales: {'Sí' if context_id in self.visual_findings else 'No'}
"""


# ==================== FUNCIÓN AUXILIAR PARA CARGAR PDFs ====================

async def load_pdfs_from_folder(agent: PhysicsPalimodalAgent, folder_path: str = "pdfs") -> str:
    """Carga todos los PDFs de una carpeta."""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        return f"No se encontraron PDFs en {folder_path}"
    
    print(f"📂 Encontrados {len(pdf_files)} PDFs")
    return await agent.procesar_y_almacenar_pdfs(pdf_files)


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    async def main():
        # Inicializar agente
        agent = PhysicsPalimodalAgent()
        
        # Ejemplo de consulta
        query = "¿Qué relación hay entre el trabajo y la energía cinética?"
        context_id = "estudiante_001"
        
        # Modo invoke
        response = await agent.invoke(query, context_id)
        print(f"Respuesta: {response}")
        
        # Modo stream
        print("\n--- Modo Streaming ---")
        async for chunk in agent.stream(query, context_id):
            print(f"[{chunk['status']}] {chunk['content'][:100]}...")
        
        # Resumen de memoria
        summary = await agent.get_memory_summary(context_id)
        print(f"\n{summary}")
        
        # Limpiar memoria
        await agent.clear_memory(context_id)

    asyncio.run(main())
