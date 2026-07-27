# samples/python/agents/multimodal/app/agent.py

import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Protect against broken pipe errors (WinError 233) that occur when
    # the agent runs as a subprocess and the parent's pipe disconnects.
    _orig_stdout_write = sys.stdout.write
    _orig_stderr_write = sys.stderr.write

    def _safe_stdout_write(s):
        try:
            return _orig_stdout_write(s)
        except OSError:
            return 0

    def _safe_stderr_write(s):
        try:
            return _orig_stderr_write(s)
        except OSError:
            return 0

    sys.stdout.write = _safe_stdout_write
    sys.stderr.write = _safe_stderr_write

import asyncio
import base64
import glob
import json
import os
from collections.abc import AsyncIterable
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
import torch.nn.functional as F

# API Key Rotator (proyecto raíz)
import sys as _sys
_project_root = str(Path(__file__).resolve().parents[5])
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)
from api_key_rotator import google_key_rotator, create_google_llm, invoke_with_retry  # noqa: E402

# ==================== CONFIGURACIÓN ====================

GROQ_MODEL = "llama-3.3-70b-versatile"

class SemanticMemory:
    """Memoria conversacional con historial de chat real."""
    
    def __init__(self, llm, max_entries: int = 10):
        self.conversations = []       # lista de {query, response} para serializar
        self.chat_history = []        # lista de LangChain messages (HumanMessage/AIMessage)
        self.max_entries = max_entries
        self.summary = ""
        self.direct_history = ""      # solo para compatibilidad con get_context()
        self.llm = llm
        # Modo socrático
        self.socratic_mode = False
        self.socratic_questions_asked = 0
        self.socratic_answers = []
        self.original_query = ""
        self.socratic_disabled = False
    
    def to_dict(self) -> dict:
        """Serializa la memoria a un diccionario."""
        # Serializar chat_history como lista de {role, content}
        serialized_history = []
        for msg in self.chat_history:
            role = "human" if msg.__class__.__name__ == "HumanMessage" else "ai"
            serialized_history.append({"role": role, "content": msg.content})
        return {
            "conversations": self.conversations,
            "chat_history": serialized_history,
            "summary": self.summary,
            "direct_history": self.direct_history,
            "socratic_mode": self.socratic_mode,
            "socratic_questions_asked": self.socratic_questions_asked,
            "socratic_answers": self.socratic_answers,
            "original_query": self.original_query,
            "socratic_disabled": getattr(self, 'socratic_disabled', False)
        }

    @classmethod
    def from_dict(cls, data: dict, llm) -> 'SemanticMemory':
        """Crea una instancia de SemanticMemory desde un diccionario."""
        from langchain_core.messages import HumanMessage as HM, AIMessage
        mem = cls(llm=llm)
        mem.conversations = data.get("conversations", [])
        mem.summary = data.get("summary", "")
        mem.direct_history = data.get("direct_history", "")
        mem.socratic_mode = data.get("socratic_mode", False)
        mem.socratic_questions_asked = data.get("socratic_questions_asked", 0)
        mem.socratic_answers = data.get("socratic_answers", [])
        mem.original_query = data.get("original_query", "")
        mem.socratic_disabled = data.get("socratic_disabled", False)
        # Reconstruir chat_history desde la lista serializada
        for entry in data.get("chat_history", []):
            if entry["role"] == "human":
                mem.chat_history.append(HM(content=entry["content"]))
            else:
                mem.chat_history.append(AIMessage(content=entry["content"]))
        return mem

    def add_interaction(self, query: str, response: str):
        """Guardar interacción en memoria (histórico + chat real)."""
        from langchain_core.messages import HumanMessage as HM, AIMessage
        
        self.conversations.append({"query": query, "response": response})
        if len(self.conversations) > self.max_entries:
            self.conversations.pop(0)
        
        # Agregar al historial real de chat
        self.chat_history.append(HM(content=query))
        self.chat_history.append(AIMessage(content=response))
        # Mantener solo las últimas 10 rondas (20 mensajes) para no exceder tokens
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        # Mantener historial de texto (compatibilidad)
        recent = self.conversations[-3:]
        self.direct_history = ""
        for conv in recent:
            self.direct_history += f"\nUsuario: {conv['query']}\nAsistente: {conv['response']}\n"
        
        self.update_summary()
    
    def add_socratic_exchange(self, question: str, answer: str):
        """Registra en el historial de chat una pregunta socrática y su respuesta."""
        from langchain_core.messages import HumanMessage as HM, AIMessage
        self.chat_history.append(AIMessage(content=question))   # el tutor preguntó
        self.chat_history.append(HM(content=answer))            # el alumno respondió
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
    
    def update_summary(self):
        """Actualizar resumen."""
        if self.conversations:
            recent_context = "\n".join([
                f"Q: {c['query']}\nA: {c['response']}"
                for c in self.conversations[-5:]
            ])
            self.summary = f"Resumen:\n{recent_context}"
    
    def get_context(self) -> str:
        """Obtener contexto textual (compatibilidad)."""
        return f"{self.summary}\n\nHistorial:\n{self.direct_history}"
    
    def clear(self):
        """Limpiar memoria."""
        self.conversations = []
        self.chat_history = []
        self.summary = ""
        self.direct_history = ""
        self.socratic_mode = False
        self.socratic_questions_asked = 0
        self.socratic_answers = []
        self.original_query = ""
        self.socratic_disabled = False


class PhysicsMultimodalAgent:
    """Agente de física con procesamiento multimodal."""
    
    # Default prompts (fallback when no optimized prompts are available)
    _DEFAULT_SYSTEM_INSTRUCTION = (
        'Eres un tutor socrático multimodal de Física I de la UBA. '
        'Usas el método socrático: ante cada consulta (texto o imagen), '
        'primero haces 3 preguntas guía para activar el pensamiento crítico '
        'del estudiante y luego proporcionas la respuesta completa. '
        'Analizas consultas de texto e imágenes de experimentos, diagramas '
        'y problemas de física.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    
    # Path to DSPy-optimized prompts (relative to this file's directory)
    _OPTIMIZED_PROMPTS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "optimized_prompts.json"
    )
    
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        """Inicializar el agente de física."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Using Google Gemini as the LLM (con key rotativa)
        self.llm = create_google_llm(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=8192
        )
        self.vision_llm = self.llm
        
        # Cargar prompts optimizados (si existen)
        self._optimized_prompts = self._load_optimized_prompts()
        self.SYSTEM_INSTRUCTION = self._get_prompt(
            "system_instruction", self._DEFAULT_SYSTEM_INSTRUCTION
        )
        
        # Qdrant
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_KEY", "")
        self.text_collection = "documentos_pdf_texto_hf"  # Nombre cambiado para evitar mismatch de dimensiones
        self.image_collection = "documentos_pdf_imagenes"
        self.multimodal_collection = "documentos_multimodal"
        
        # Modelo CLIP (para imágenes y búsqueda de imágenes con texto)
        # Forzar CPU para dejar VRAM libre para ColPali del agente médico
        device = "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Modelo HuggingFace (para texto)
        self.hf_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.hf_model = AutoModel.from_pretrained(self.hf_model_name).to(device)
        self.hf_vector_size = 384  # Dimensión de salida de MiniLM-L12
        
        # Memoria conversacional
        self.memories = {}
        self.visual_findings = {}
        # Guardar memorias en el directorio del agente (no /tmp)
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._memories_file = os.path.join(agent_dir, "physics_agent_memories.json")
        
        self._load_memories()

    def _save_memories(self):
        """Guarda todas las memorias en el disco."""
        try:
            data = {
                cid: mem.to_dict() 
                for cid, mem in self.memories.items()
            }
            with open(self._memories_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error guardando memorias: {e}")

    def _load_memories(self):
        """Carga todas las memorias desde el disco."""
        if os.path.exists(self._memories_file):
            try:
                with open(self._memories_file, 'r') as f:
                    data = json.load(f)
                for cid, mem_data in data.items():
                    self.memories[cid] = SemanticMemory.from_dict(mem_data, self.llm)
                    # Resetear estado socrático al cargar para evitar estados stale
                    mem = self.memories[cid]
                    if mem.socratic_mode:
                        print(f"⚠️ Reseteando estado socrático stale para contexto {cid[:12]}...")
                        mem.socratic_mode = False
                        mem.socratic_questions_asked = 0
                        mem.socratic_answers = []
                        mem.original_query = ""
                print(f"📖 Memorias del agente cargadas: {len(self.memories)}")
            except Exception as e:
                print(f"Error cargando memorias del agente: {e}")
                self.memories = {}
        else:
            self.memories = {}
        self.temario = ""
        
        print("✅ PhysicsMultimodalAgent inicializado")
    
    def _load_optimized_prompts(self) -> dict:
        """Carga prompts optimizados por DSPy desde JSON.
        
        Returns:
            dict con prompts optimizados, o dict vacío si no existe el archivo.
        """
        path = self._OPTIMIZED_PROMPTS_PATH
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                prompts = data.get("prompts", {})
                meta = data.get("metadata", {})
                print(f"🧠 Prompts optimizados cargados desde {Path(path).name}")
                print(f"   Modelo de optimización: {meta.get('model', 'desconocido')}")
                print(f"   Prompts disponibles: {list(prompts.keys())}")
                return prompts
            except Exception as e:
                print(f"⚠️ Error cargando prompts optimizados: {e}")
                return {}
        else:
            print("ℹ️ No se encontró optimized_prompts.json, usando prompts por defecto")
            return {}
    
    def _get_prompt(self, key: str, default: str) -> str:
        """Obtiene un prompt optimizado o devuelve el default.
        
        Args:
            key: Clave del prompt en optimized_prompts.json
            default: Valor por defecto si no existe el prompt optimizado
        
        Returns:
            El prompt optimizado si existe, o el default.
        """
        if key in self._optimized_prompts:
            instruction = self._optimized_prompts[key].get("instruction", "")
            if instruction:
                return instruction
        return default
    
    def _get_demos(self, key: str) -> str:
        """Obtiene los few-shot demos como texto formateado.
        
        Args:
            key: Clave del prompt en optimized_prompts.json
        
        Returns:
            String con los demos formateados, o string vacío.
        """
        if key in self._optimized_prompts:
            demos = self._optimized_prompts[key].get("demos", [])
            if demos:
                demo_texts = []
                for i, demo in enumerate(demos, 1):
                    parts = [f"\n--- Ejemplo {i} ---"]
                    for k, v in demo.items():
                        if k != "rationale":  # Skip chain-of-thought internals
                            parts.append(f"{k}: {v}")
                    demo_texts.append("\n".join(parts))
                return "\n\nEJEMPLOS DE REFERENCIA:\n" + "\n".join(demo_texts)
        return ""
    
    # ==================== MÉTODOS DE PROCESAMIENTO DE PDFs ====================
    # (Copiar todos los métodos de procesamiento del archivo original)
    
    def leer_pdf(self, archivo: str) -> str:
        """Leer texto de un PDF."""
        try:
            reader = PdfReader(archivo)
            return "".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")
            return ""
    
    def extraer_imagenes_pdf(self, pdf_path: str, output_folder: str = "extracted_images") -> List[str]:
        """Extraer imágenes embebidas (diagramas, figuras, fotos) de un PDF.
        
        Extrae solo las imágenes reales incrustadas en el PDF, no renderiza
        cada página completa. Filtra imágenes muy pequeñas (íconos, viñetas).
        """
        import os
        from pathlib import Path
        
        os.makedirs(output_folder, exist_ok=True)
        imagenes = []
        seen_xrefs = set()  # Evitar duplicados de la misma imagen en varias páginas
        
        try:
            import fitz
            doc = fitz.open(pdf_path)
            stem = Path(pdf_path).stem
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    
                    # Evitar extraer la misma imagen dos veces
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    
                    try:
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue
                        
                        # Filtrar imágenes muy pequeñas (íconos, viñetas, etc.)
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        if width < 100 or height < 100:
                            continue
                        
                        image_bytes = base_image["image"]
                        image_ext = base_image.get("ext", "png")
                        
                        img_path = os.path.join(
                            output_folder,
                            f"{stem}_p{page_num}_img{img_index}.{image_ext}"
                        )
                        
                        # No re-extraer si ya existe en disco
                        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                            imagenes.append(img_path)
                            continue
                        
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        imagenes.append(img_path)
                    except Exception:
                        continue
            
            doc.close()
            print(f"   🖼️ {len(imagenes)} imágenes extraídas de {Path(pdf_path).name}")
            return imagenes
        except Exception as e:
            print(f"❌ Error extrayendo imágenes: {e}")
            return []
    
    def split_text(self, text: str, chunk_words: int = 350, overlap: int = 50) -> List[str]:
        """Dividir texto en chunks de palabras (máx 512 tokens para modelo HF)."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_words - overlap):
            chunk = " ".join(words[i:i + chunk_words])
            chunks.append(chunk)
            if i + chunk_words >= len(words):
                break
        return chunks
    
    def generate_hf_text_embeddings_batch(self, chunks: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generar embeddings de texto en batch usando HuggingFace."""
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inputs = self.hf_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.hf_model.device)
            
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
            
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_features = sum_embeddings / sum_mask
            text_features = F.normalize(text_features, p=2, dim=1)
            
            embeddings.extend(text_features.cpu().numpy().tolist())
        return embeddings
    
    def generate_image_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """Generar embedding de imagen."""
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.clip_model.device)
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                # Extraer el tensor del output
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    image_features = outputs.last_hidden_state[:, 0]
                else:
                    image_features = outputs
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None
    
    def generate_clip_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generar embedding de texto usando CLIP (para buscar en la colección de imágenes)."""
        try:
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.clip_model.device)
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    text_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    text_features = outputs.last_hidden_state[:, 0]
                else:
                    text_features = outputs
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"❌ Error CLIP text embedding: {e}")
            return None

    def generate_hf_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generar embedding de texto usando HuggingFace."""
        try:
            inputs = self.hf_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.hf_model.device)
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
            
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_features = sum_embeddings / sum_mask
            text_features = F.normalize(text_features, p=2, dim=1)
            
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"❌ Error HF text embedding: {e}")
            return None
    
    async def store_in_qdrant(self, points: List[Any], collection_name: str, vector_size: int = 512):
        """Almacenar puntos en Qdrant."""
        client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=60.0)
        try:
            await client.get_collection(collection_name)
            print(f"📦 Colección '{collection_name}' existe")
        except Exception:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✨ Colección '{collection_name}' creada con dimensión {vector_size}")
            
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await client.upsert(collection_name=collection_name, points=batch, wait=True)
            print(f"🔄 Lote de {len(batch)} puntos insertado ({min(i + batch_size, len(points))}/{len(points)})")
            
        print(f"✅ {len(points)} elementos almacenados en '{collection_name}'")
    
    async def extraer_temario(self, contenido_completo: str) -> str:
        """Extraer temario usando el inicio de cada documento."""
        print("🤖 Extrayendo temario...")
        # Tomamos hasta 40000 caracteres (suficiente para los índices de 14 documentos)
        contenido_limitado = contenido_completo[:40000]
        
        system_message = f"""Eres un profesor de Física I de la UBA.
Extrae el TEMARIO COMPLETO a partir de los siguientes fragmentos, que corresponden al inicio de cada documento del curso.

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
            response = invoke_with_retry(self.llm, messages)
            print(f"✅ Temario extraído ({len(response.content)} caracteres)")
            return response.content
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return "Temario no disponible."
    
    async def procesar_y_almacenar_pdfs(self, pdf_files: List[str]) -> str:
        """Procesar PDFs y almacenar en Qdrant."""
        print("\n" + "="*80)
        print("📚 PROCESANDO PDFs")
        print("="*80)
        
        text_points = []
        image_points = []
        global_id_counter = 0
        contenido_completo_texto = ""
        contenido_para_temario = ""
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
                continue
            
            print(f"\n📄 Procesando: {Path(pdf_file).name}")
            
            # Texto
            text = self.leer_pdf(pdf_file)
            if text:
                contenido_completo_texto += f"\n--- {Path(pdf_file).name} ---\n{text}"
                contenido_para_temario += f"\n--- Documento: {Path(pdf_file).name} ---\n{text[:2500]}\n"
                chunks = self.split_text(text)
                print(f"   📝 {len(chunks)} chunks")
                embeddings = self.generate_hf_text_embeddings_batch(chunks)
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    text_points.append(PointStruct(
                        id=global_id_counter,
                        vector=embedding,
                        payload={
                            "pdf_name": pdf_file,
                            "type": "text",
                            "chunk_id": i,
                            "text": chunk
                        }
                    ))
                    global_id_counter += 1
            
            # Imágenes
            imagenes = self.extraer_imagenes_pdf(pdf_file)
            
            for img_path in imagenes:
                img_embedding = self.generate_image_embedding(open(img_path, 'rb').read())
                if img_embedding:
                    image_points.append(PointStruct(
                        id=global_id_counter,
                        vector=img_embedding,
                        payload={
                            "pdf_name": pdf_file,
                            "type": "image",
                            "image_path": img_path
                        }
                    ))
                    global_id_counter += 1
        
        # Extraer temario
        temario = await self.extraer_temario(contenido_para_temario)
        
        # Guardar temario a disco (en el directorio del agente)
        try:
            agent_dir = Path(os.path.abspath(__file__)).parents[1]
            temario_path = agent_dir / "temario.txt"
            with open(temario_path, "w", encoding="utf-8") as f:
                f.write(temario)
        except Exception as e:
            print(f"⚠️ No se pudo guardar el temario a disco: {e}")
        
        # Almacenar
        if text_points:
            await self.store_in_qdrant(text_points, self.text_collection, vector_size=self.hf_vector_size)
        if image_points:
            await self.store_in_qdrant(image_points, self.image_collection, vector_size=512)
        
        print("\n✅ PROCESAMIENTO COMPLETADO")
        print(f"   📝 Texto: {len(text_points)} chunks")
        print(f"   🖼️ Imágenes: {len(image_points)} embeddings")
        
        self.temario = temario
        return temario
    
    # ==================== MÉTODOS DE ANÁLISIS ====================
    
    def _sanitize_nams_context(self, nams_context: str, max_len: int = 3000) -> str:
        """Filtra y trunca el contexto NAMS para evitar que domine el prompt.
        
        Elimina líneas que parecen historial de chat (ya cubierto por
        SemanticMemory.chat_history) y trunca a max_len caracteres.
        """
        if not nams_context or not nams_context.strip():
            return ""
        
        # Prefijos que indican historial de chat (ya está en chat_history)
        chat_prefixes = (
            'user:', 'assistant:', 'human:', 'ai:',
            'usuario:', 'asistente:', 'q:', 'a:',
            'pregunta:', 'respuesta:',
            '- [user]', '- [assistant]', '- [human]', '- [ai]',
            '- [usuario]', '- [asistente]'
        )
        
        # Secciones completas a ignorar
        ignore_headers = (
            '## conversation history',
            '### relevant past messages',
            'conversation history',
            'relevant past messages'
        )
        
        filtered_lines = []
        in_chat_history_section = False
        
        for line in nams_context.split('\n'):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if not line_lower:
                continue
                
            # Detectar si entramos a una sección de historial de conversación
            if any(h in line_lower for h in ignore_headers):
                in_chat_history_section = True
                continue
                
            # Si entramos en la sección de conocimiento relevante o preferencias, desactivar el ignore
            if '## relevant knowledge' in line_lower or '### user preferences' in line_lower:
                in_chat_history_section = False
                continue
                
            if in_chat_history_section:
                continue
                
            # Saltar líneas que parecen historial de chat individual
            if any(line_lower.startswith(prefix) for prefix in chat_prefixes):
                continue
                
            filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines).strip()
        
        if len(result) > max_len:
            result = result[:max_len] + '\n[... truncado]'
        
        return result

    
    def _get_or_create_memory(self, context_id: str) -> SemanticMemory:
        if context_id not in self.memories:
            self.memories[context_id] = SemanticMemory(llm=self.llm)
        return self.memories[context_id]
    
    def _get_memory_context(self, context_id: str) -> str:
        memory = self._get_or_create_memory(context_id)
        return memory.get_context()
    
    def _save_to_memory(self, context_id: str, query: str, response: str):
        """Guardar interacción en memoria."""
        memory = self._get_or_create_memory(context_id)
        memory.add_interaction(query, response)
        self._save_memories()
    
    def encode_image(self, image_data: bytes) -> str:
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_mime_type(self, content_type: str) -> str:
        mapping = {
            'image/jpeg': 'image/jpeg',
            'image/png': 'image/png',
            'image/webp': 'image/webp',
        }
        return mapping.get(content_type, 'image/png')
    
    @traceable(name="search_qdrant", run_type="retriever", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def search_multimodal(
        self, 
        query: str = None, 
        image_embedding: List[float] = None,
        top_k: int = 5
    ) -> dict[str, List[dict]]:
        """Búsqueda en Qdrant (solo texto para el tutor de física)."""
        client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        results = {"text": [], "image": []}
        
        try:
            if query:
                search_embedding_hf = self.generate_hf_text_embedding(query)
                if search_embedding_hf:
                    try:
                        search_results = await client.query_points(
                            collection_name=self.text_collection,
                            query=search_embedding_hf,
                            limit=top_k
                        )
                        points = search_results.points if hasattr(search_results, 'points') else search_results
                        results["text"] = [{
                            "id": r.id,
                            "score": round(r.score, 4),
                            "payload": r.payload
                        } for r in points]
                    except Exception as e:
                        print(f"⚠️ Error en {self.text_collection}: {e}")
            
            return results
        except Exception as e:
            print(f"❌ Error: {e}")
            return results
    
    @traceable(name="analyze_physics_image", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def analyze_physics_image(self, images: List[dict], pdf_text_context: str = "") -> str:
        """Analiza imágenes de física."""
        if not images:
            return "No se proporcionaron imágenes."
        
        prompt_prefix = ""
        if pdf_text_context and pdf_text_context.strip():
            prompt_prefix = f"TEXTO DEL MATERIAL DE LA MATERIA ASOCIADO A LA IMAGEN (obtenido por búsqueda visual):\n{pdf_text_context}\n\n"

        content = [{
            "type": "text",
            "text": f"""{prompt_prefix}Eres un experto en Física. Observa esta(s) {len(images)} imagen(es) y describí con MÁXIMO DETALLE VISUAL todo lo que ves.

1. **ELEMENTOS VISIBLES**: Enumerá TODOS los objetos, cuerpos, dispositivos, símbolos, flechas, letras, números y etiquetas que aparecen en la imagen. No omitas nada.

2. **CONEXIONES Y RELACIONES**: Describí cómo están conectados o relacionados los elementos entre sí (qué toca qué, qué está encima/debajo/dentro de qué, qué está unido a qué, qué pasa por dónde).

3. **GEOMETRÍA Y DISPOSICIÓN**: Ángulos, orientaciones, posiciones relativas, simetrías, direcciones de flechas o vectores si los hay.

4. **DATOS Y VARIABLES**: Cualquier valor numérico, variable, ecuación o texto escrito en la imagen.

5. **INTERPRETACIÓN FÍSICA**: Basándote en lo que VES, ¿qué fenómeno o problema físico representa la imagen? ¿Qué principios y leyes serían aplicables?

REGLAS:
- Describí ÚNICAMENTE lo que ves. No inventes elementos que no estén en la imagen.
- Sé exhaustivo: es mejor describir de más que omitir algo.
- Si hay fuerzas, tensiones, pesos, reacciones, corrientes, campos, ondas, o cualquier magnitud representada, mencionala explícitamente."""
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
                # Groq limita base64 a 4MB
                b64_size_mb = len(image_data_b64) / (1024 * 1024)
                print(f"🖼️ Imagen {idx}: {mime_type}, {b64_size_mb:.2f} MB (base64)")
                if b64_size_mb > 4.0:
                    print(f"⚠️ Imagen {idx} excede 4MB (límite Groq), omitiendo")
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data_b64}"}
                })
            except Exception as e:
                print(f"❌ Error imagen {idx}: {e}")
        
        try:
            response = invoke_with_retry(self.vision_llm, [HumanMessage(content=content)])
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
            
    def _get_pdf_page_text(self, pdf_path: str, image_path: str) -> str:
        """Extrae el texto de la página del PDF usando el nombre de la imagen."""
        try:
            import fitz
            import re
            basename = os.path.basename(image_path)
            match = re.search(r'_p(\d+)_', basename)
            if match:
                page_idx = int(match.group(1))
                if os.path.exists(pdf_path):
                    doc = fitz.open(pdf_path)
                    if 0 <= page_idx < len(doc):
                        text = doc[page_idx].get_text()
                        print(f"📖 Recuperado texto del PDF {os.path.basename(pdf_path)} Página {page_idx+1} ({len(text)} caracteres)")
                        return text
                else:
                    # Intentar buscar en el directorio PDF local del agente
                    agent_pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PDF")
                    local_path = os.path.join(agent_pdf_dir, os.path.basename(pdf_path))
                    if os.path.exists(local_path):
                        doc = fitz.open(local_path)
                        if 0 <= page_idx < len(doc):
                            text = doc[page_idx].get_text()
                            print(f"📖 Recuperado texto del PDF local {os.path.basename(pdf_path)} Página {page_idx+1} ({len(text)} caracteres)")
                            return text
        except Exception as e:
            print(f"⚠️ Error leyendo texto de página de PDF: {e}")
        return ""
        
    async def _get_visual_findings(self, images: List[dict], context_id: str) -> tuple[str, Optional[List[float]]]:
        """Analiza la imagen directamente usando el LLM de visión, sin realizar búsquedas de imágenes en Qdrant."""
        visual_findings = ""
        image_embedding = None
        
        if images and len(images) > 0:
            first_image_data = images[0].get('data') or images[0].get('bytes')
            if isinstance(first_image_data, str):
                first_image_data = base64.b64decode(first_image_data)
            image_embedding = self.generate_image_embedding(first_image_data)
            
            # El agente multimodal de física no realiza búsquedas de imágenes en la base de datos,
            # analiza la imagen subida directamente usando el LLM de visión para resolver el ejercicio.
            visual_findings = await self.analyze_physics_image(images, "")
            self.visual_findings[context_id] = visual_findings
        else:
            visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
            
        return visual_findings, image_embedding
    
    @traceable(name="classify_query", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
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
            response = invoke_with_retry(self.llm, messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @traceable(name="generate_search_query", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
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
            response = invoke_with_retry(self.llm, messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @traceable(name="generate_physics_response", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def generate_physics_response(
        self, 
        query: str, 
        context: str, 
        classification: str, 
        visual_findings: str,
        document_context: str,
        image_context: str,
        chat_history: list = None,
        nams_context: str = ""
    ) -> str:
        """Genera respuesta final usando historial real de chat."""
        _default_direct = f"""Eres un Profesor de Física I UBA en modo de diálogo directo con el estudiante.
Debes responder la consulta actual teniendo en cuenta TODO el historial de la conversación.

TEMARIO:
{self.temario}

Estructura de respuesta:
1. CONTEXTO DEL TEMA
2. EXPLICACIÓN TEÓRICA (detallada, sin omitir pasos)
3. ANÁLISIS DE IMÁGENES (si hay)
4. ECUACIONES
5. EJEMPLOS
6. RESUMEN

Reglas:
- **PRIORIDAD ABSOLUTA**: Responde SIEMPRE a la CONSULTA ACTUAL del estudiante. El historial de conversación y las preferencias de fondo son solo contexto secundario. Tu respuesta DEBE basarse en la pregunta actual y los DOCUMENTOS DE REFERENCIA.
- RESPONDE ÚNICAMENTE basándote en los DOCUMENTOS DE REFERENCIA proporcionados. Si la información solicitada no está en los documentos, indica explícitamente que el tema no se encuentra en el temario de la materia y NO inventes una respuesta.
- NUNCA repitas lo que ya explicaste antes. Avanza en la conversación.
- Técnico pero claro. Conecta con lo que el estudiante ya dijo.
- **IMPORTANTE: Todas las fórmulas y ecuaciones DEBEN estar en formato LaTeX**
  - Usa `$formula$` para fórmulas inline
  - Usa `$$formula$$` para fórmulas display
- NUNCA uses texto plano para fórmulas
- Usa notación matemática correcta: vectores con \\vec{{}}, fracciones con \\frac{{}}{{}}
"""
        system_prompt = self._get_prompt("direct_response", _default_direct)
        demos = self._get_demos("direct_response")
        if demos:
            system_prompt += demos
        
        # Inyectar contexto NAMS (hechos, conclusiones de aprendizaje y preferencias del usuario)
        sanitized_nams = self._sanitize_nams_context(nams_context)
        if sanitized_nams:
            system_prompt += (
                f"\n\n--- CONOCIMIENTO Y PREFERENCIAS APRENDIDAS (NAMS) ---\n"
                f"IMPORTANTE: Las siguientes son hechos, conclusiones de aprendizaje y preferencias que el estudiante ha establecido o corregido previamente en NAMS. "
                f"Debes respetar estas conclusiones y aplicarlas directamente en tu explicación si se relacionan con el tema tratado. "
                f"Por ejemplo, si hay una conclusión sobre cómo calcular la fuerza de rozamiento en rodadura, incorpórala de forma prioritaria y coherente en tu respuesta:\n"
                f"{sanitized_nams}"
            )
        
        # Agregar contexto de documentos al system prompt
        if document_context and document_context.strip():
            system_prompt += f"\n\nDOCUMENTOS DE REFERENCIA:\n{document_context[:15000]}"
        if image_context and image_context.strip():
            system_prompt += f"\n\nIMÁGENES RELACIONADAS:\n{image_context[:5000]}"
        if visual_findings and visual_findings.strip() and visual_findings != "No hay imágenes.":
            system_prompt += f"\n\nHALLAZGOS VISUALES:\n{visual_findings[:5000]}"
        if classification and classification.strip():
            system_prompt += f"\n\nCLASIFICACIÓN DEL TEMA:\n{classification[:2000]}"
        
        try:
            messages = [SystemMessage(content=system_prompt)]
            
            # Añadir el historial real de chat (lo que se habló antes)
            if chat_history:
                messages.extend(chat_history)
            
            # El mensaje actual del usuario
            messages.append(HumanMessage(content=query))
            
            response = invoke_with_retry(self.llm, messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @traceable(name="check_socratic_intent", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def check_socratic_intent(self, query: str) -> str:
        """Verifica si el usuario quiere cambiar de modo (salir o entrar al modo socrático)."""
        prompt = f"""Analiza la intención del usuario en el siguiente mensaje.
El usuario está interactuando con un tutor de física. 
Determina si el usuario explícitamente pide:
1. SALIR: Salir del modo socrático, dejar de recibir preguntas, que le den la respuesta directa, o dialogar normalmente.
2. ENTRAR: Volver al modo socrático, pedir que le hagan preguntas para pensar, o reiniciar el método socrático.
3. CONTINUAR: Ninguna de las anteriores. Simplemente está respondiendo a una pregunta o haciendo una consulta de física normal.

Mensaje del usuario: "{query}"

Responde SOLO con una de estas palabras: SALIR, ENTRAR, CONTINUAR."""
        try:
            response = invoke_with_retry(self.llm, [HumanMessage(content=prompt)])
            content = response.content.upper()
            if "SALIR" in content: return "SALIR"
            if "ENTRAR" in content: return "ENTRAR"
            return "CONTINUAR"
        except Exception:
            return "CONTINUAR"
    
    async def _search_qdrant_for_context(self, query: str, image_embedding: List[float] = None, top_k: int = 5) -> tuple:
        """Busca en Qdrant y retorna (document_context, image_context) formateados.
        
        Helper centralizado para evitar duplicar la lógica de búsqueda
        en todos los flujos (socrático, directo, etc.).
        """
        try:
            search_results = await self.search_multimodal(
                query=query,
                image_embedding=image_embedding,
                top_k=top_k
            )
            
            document_context = "\n".join([
                f"--- Fragmento {i+1} ---\n{r['payload'].get('text', 'N/A')}"
                for i, r in enumerate(search_results.get('text', []))
            ])
            
            image_context = "\n".join([
                f"--- Imagen {i+1} ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
                for i, r in enumerate(search_results.get('image', []))
            ])
            
            return document_context, image_context
        except Exception as e:
            print(f"⚠️ Error buscando en Qdrant: {e}")
            return "", ""
    
    @traceable(name="generate_socratic_question", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def generate_socratic_question(
        self,
        original_query: str,
        question_number: int,
        previous_answers: List[str],
        visual_findings: str = "",
        document_context: str = "",
        nams_context: str = ""
    ) -> str:
        """Genera una pregunta socrática para guiar al estudiante.
        
        Ahora recibe document_context de Qdrant para basar las preguntas
        en el material real de la materia.
        """
        visual_section = ""
        if visual_findings:
            visual_section = f"""\nHALLAZGOS VISUALES (de imágenes proporcionadas por el estudiante):
{visual_findings}
- Incorpora lo que se observa en las imágenes en tus preguntas.
- Pregunta al estudiante qué fenómenos físicos identifica en la imagen.
"""

        document_section = ""
        if document_context and document_context.strip():
            document_section = f"""\nDOCUMENTOS DE REFERENCIA (material de la materia):
{document_context[:10000]}
- BASA tus preguntas en estos documentos. Usa los conceptos, ecuaciones y ejemplos que aparecen aquí.
- Las preguntas deben guiar al estudiante hacia los conceptos clave de estos documentos.
"""

        _default_socratic = f"""Eres un tutor socrático de Física I de la UBA.

Tu objetivo es guiar al estudiante a descubrir la respuesta por sí mismo mediante preguntas.
Recibís tanto texto como imágenes de experimentos, diagramas y problemas de física.

TEMARIO:
{self.temario}
{visual_section}
{document_section}

Reglas para las preguntas:
- BASA TUS PREGUNTAS ÚNICAMENTE en el contenido del TEMARIO y los DOCUMENTOS DE REFERENCIA proporcionados. Si el tema no pertenece al temario, informa al estudiante que el tema está fuera de alcance.
- Pregunta {question_number + 1}/3
- Haz preguntas que activen el pensamiento crítico
- Si el estudiante envió una imagen, preguntá sobre lo que se observa en ella
- Relaciona con conceptos fundamentales presentes en los documentos
- Progresa desde lo básico a lo específico
- Sé breve y directo
- **Si incluyes fórmulas, usa formato LaTeX**: `$formula$` para inline, `$$formula$$` para display
  - Ejemplo: ¿Qué relación hay entre $F$ y $a$ en la segunda ley de Newton?

Formato de respuesta:
🎓 **Pregunta {question_number + 1}/3**

[Tu pregunta aquí]

💡 *Piensa en los conceptos fundamentales antes de responder.*
"""
        system_prompt = self._get_prompt("socratic_question", _default_socratic)
        # Append few-shot demos if available
        demos = self._get_demos("socratic_question")
        if demos:
            system_prompt += demos
        
        # Inyectar contexto NAMS (hechos, conclusiones de aprendizaje y preferencias del usuario)
        sanitized_nams = self._sanitize_nams_context(nams_context)
        if sanitized_nams:
            system_prompt += (
                f"\n\n--- CONOCIMIENTO Y PREFERENCIAS APRENDIDAS (NAMS) ---\n"
                f"IMPORTANTE: Las siguientes son hechos, conclusiones de aprendizaje y preferencias que el estudiante ha establecido o corregido previamente en NAMS. "
                f"Debes respetar estas conclusiones y aplicarlas directamente en tu explicación si se relacionan con el tema tratado. "
                f"Por ejemplo, si hay una conclusión sobre cómo calcular la fuerza de rozamiento en rodadura, incorpórala de forma prioritaria y coherente en tu respuesta:\n"
                f"{sanitized_nams}"
            )
        
        previous_context = ""
        if previous_answers:
            previous_context = "\n\nRespuestas previas del estudiante:\n" + "\n".join([
                f"Pregunta {i+1}: {ans}"
                for i, ans in enumerate(previous_answers)
            ])
        
        user_prompt = f"""
CONSULTA ORIGINAL DEL ESTUDIANTE:
{original_query}
{previous_context}

Genera la pregunta socrática número {question_number + 1} para guiar al estudiante."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = invoke_with_retry(self.llm, messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @traceable(name="generate_physics_response_with_socratic", run_type="llm", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def generate_physics_response_with_socratic(
        self,
        query: str,
        context: str,
        classification: str,
        visual_findings: str,
        document_context: str,
        image_context: str,
        student_answers: str,
        nams_context: str = ""
    ) -> str:
        """Genera respuesta final después del diálogo socrático."""
        _default_post_socratic = f"""Profesor de Física I UBA que usa el método socrático.

Has guiado al estudiante con 3 preguntas socráticas. Ahora proporciona la respuesta completa.

TEMARIO:
{self.temario}

Estructura de tu respuesta:
1. **REFLEXIÓN SOBRE TUS RESPUESTAS**: Comenta brevemente las respuestas del estudiante
2. **CONTEXTO DEL TEMA**: Sitúa el problema en el temario
3. **EXPLICACIÓN TEÓRICA**: Teoría fundamental
4. **ANÁLISIS DETALLADO**: Conecta con las preguntas socráticas
5. **ECUACIONES Y CÁLCULOS**: Desarrollo matemático
6. **EJEMPLOS PRÁCTICOS**: Aplicaciones
7. **RESUMEN Y CONCLUSIÓN**: Síntesis final

Reglas:
- **PRIORIDAD ABSOLUTA**: Responde SIEMPRE a la CONSULTA ORIGINAL del estudiante y valora sus respuestas socráticas. Las preferencias de fondo son solo contexto secundario.
- RESPONDE ÚNICAMENTE basándote en los DOCUMENTOS DE REFERENCIA proporcionados. Si la información no está en los documentos, indícalo claramente y no inventes.
- Reconoce los aciertos del estudiante
- Corrige errores con tacto
- Conecta sus respuestas con la teoría
- Refuerza el aprendizaje activo
- **CRÍTICO: Todas las fórmulas DEBEN estar en formato LaTeX**
  - Usa `$formula$` para fórmulas inline (en línea con el texto)
  - Usa `$$formula$$` para fórmulas display (en bloque separado)
  - Ejemplos correctos:
    * Inline: La segunda ley de Newton establece que $\\vec{{F}} = m\\vec{{a}}$
    * Display: $$E_k = \\frac{{1}}{{2}}mv^2$$
    * Display con múltiples líneas:
      $$
      \\begin{{align}}
      W &= \\Delta E_k \\\\
      W &= \\frac{{1}}{{2}}mv_f^2 - \\frac{{1}}{{2}}mv_i^2
      \\end{{align}}
      $$
- NUNCA uses texto plano para fórmulas
- Usa notación matemática correcta: \\vec{{}}, \\frac{{}}{{}}, \\Delta, \\theta, etc.
"""
        system_prompt = self._get_prompt("post_socratic_response", _default_post_socratic)
        # Append few-shot demos if available
        demos = self._get_demos("post_socratic_response")
        if demos:
            system_prompt += demos
        
        # Inyectar contexto NAMS (hechos, conclusiones de aprendizaje y preferencias del usuario)
        sanitized_nams = self._sanitize_nams_context(nams_context)
        if sanitized_nams:
            system_prompt += (
                f"\n\n--- CONOCIMIENTO Y PREFERENCIAS APRENDIDAS (NAMS) ---\n"
                f"IMPORTANTE: Las siguientes son hechos, conclusiones de aprendizaje y preferencias que el estudiante ha establecido o corregido previamente en NAMS. "
                f"Debes respetar estas conclusiones y aplicarlas directamente en tu explicación si se relacionan con el tema tratado. "
                f"Por ejemplo, si hay una conclusión sobre cómo calcular la fuerza de rozamiento en rodadura, incorpórala de forma prioritaria y coherente en tu respuesta:\n"
                f"{sanitized_nams}"
            )
        
        user_prompt = f"""
CONSULTA ORIGINAL:
{query}

CONTEXTO PREVIO:
{context}

CLASIFICACIÓN:
{classification}

HALLAZGOS VISUALES:
{visual_findings}

RESPUESTAS DEL ESTUDIANTE A LAS PREGUNTAS SOCRÁTICAS:
{student_answers}

DOCUMENTOS DE REFERENCIA:
{document_context}

IMÁGENES RELACIONADAS:
{image_context}

Proporciona la explicación completa con todas las fórmulas en LaTeX, valorando el proceso de pensamiento del estudiante."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = invoke_with_retry(self.llm, messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ==================== DETECCIÓN DE INTENCIÓN SOCRÁTICA ====================
    
    async def _detect_socratic_intent(self, query: str, is_in_socratic_mode: bool) -> str:
        """Detecta la intención del usuario respecto al modo socrático.
        
        Args:
            query: El mensaje del usuario
            is_in_socratic_mode: Si actualmente está en modo socrático
            
        Returns: "SALIR", "ENTRAR", o "CONTINUAR"
        """
        query_lower = query.lower().strip()
        
        # ─── CAPA 1: Keywords rápidos (sin LLM) ───
        
        # Keywords de SALIDA explícita
        exit_keywords = [
            "salir", "quiero salir", "salir del modo", "salir del socrático", "salir del socratico",
            "no más preguntas", "no mas preguntas", "dame la respuesta",
            "respuesta directa", "sin preguntas", "deja de preguntar",
            "no quiero preguntas", "modo normal",
            "desactivar socrático", "desactivar socratico", "sin modo socratico",
            "sin modo socrático", "sin socrático", "sin socratico",
            "no me hagas preguntas", "dejá de preguntar", "deja de hacer preguntas",
            "quiero la respuesta", "decime la respuesta", "dime la respuesta",
            "explicame directamente", "explicame sin preguntas",
            "no quiero más preguntas", "no quiero mas preguntas",
            "basta de preguntas", "ya no quiero preguntas",
            "solo explicame", "solo explícame", "solo respondeme",
            "respondeme directo", "responde directo",
            "[directo]",
        ]
        
        if any(kw in query_lower for kw in exit_keywords):
            print(f"🔑 [KEYWORD] Exit detectado: '{query[:50]}'")
            return "SALIR"
        
        # Keywords de ENTRADA explícita (solo si NO estamos en modo socrático)
        if not is_in_socratic_mode:
            enter_keywords = [
                "haceme preguntas", "hazme preguntas", "quiero preguntas",
                "activar socrático", "activar socratico", 
                "volver al socrático", "volver al socratico",
                "preguntas socráticas", "preguntas socraticas",
                "con preguntas", "preguntame",
            ]
            # "modo socrático" y "método socrático" SOLO si NO tiene negación antes
            context_keywords = [
                "modo socrático", "modo socratico", "método socrático", "metodo socratico",
            ]
            
            if any(kw in query_lower for kw in enter_keywords):
                print(f"🔑 [KEYWORD] Enter detectado: '{query[:50]}'")
                return "ENTRAR"
            
            for kw in context_keywords:
                if kw in query_lower:
                    # Verificar que no haya negación antes
                    kw_pos = query_lower.index(kw)
                    prefix = query_lower[:kw_pos].strip()
                    negations = ["sin", "no", "quitar", "desactivar", "sacar", "salir", "fuera", "basta"]
                    if not any(prefix.endswith(neg) for neg in negations):
                        print(f"🔑 [KEYWORD] Enter (context) detectado: '{query[:50]}'")
                        return "ENTRAR"
        
        # ─── CAPA 2: Si estamos en modo socrático, usar LLM para ambigüedades ───
        if is_in_socratic_mode:
            try:
                from langchain_core.messages import HumanMessage
                
                prompt = f"""Clasificador de intención. Un estudiante de física participa en preguntas socráticas.

CONTINUAR = cualquier respuesta a la pregunta de física, duda, o interacción normal:
"No sé", "Creo que es la gravedad", "5 m/s", "No entiendo", "Ayuda", "Ni idea", "Sí", "No"

SALIR = pedido EXPLÍCITO de abandonar las preguntas y recibir respuesta directa:
"No quiero más preguntas", "Explicame directamente", "Dame la respuesta", "Quiero salir"

EN CASO DE DUDA → CONTINUAR.

Mensaje: "{query}"
Responde SOLO: SALIR o CONTINUAR"""
                
                response = invoke_with_retry(self.llm, [HumanMessage(content=prompt)])
                result = response.content.strip().upper()
                print(f"🧠 [LLM] Intent socrático: '{query[:50]}' → {result}")
                
                if "SALIR" in result:
                    return "SALIR"
            except Exception as e:
                print(f"⚠️ Error en LLM intent: {e}")
        
        return "CONTINUAR"
    
    # ==================== MÉTODOS PRINCIPALES ====================
    @traceable(name="PhysicsMultimodalAgent.invoke", run_type="chain", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def invoke(self, query: str, context_id: str, 
                    images: List[dict] = None) -> str:
        """Procesa consulta completa con diálogo directo (modo socrático desactivado)."""
        # Extract NAMS context from query if present
        nams_context = ""
        if "[NAMS_CONTEXT]" in query and "[/NAMS_CONTEXT]" in query:
            start_tag = "[NAMS_CONTEXT]"
            end_tag = "[/NAMS_CONTEXT]"
            start_idx = query.find(start_tag)
            end_idx = query.find(end_tag)
            nams_context = query[start_idx + len(start_tag):end_idx].strip()
            remainder = query[end_idx + len(end_tag):]
            if remainder.startswith("\n\n"):
                query = remainder[2:]
            elif remainder.startswith("\n"):
                query = remainder[1:]
            else:
                query = remainder

        print(f"\n{'='*80}")
        print(f"📚 Consulta de física (Directa)")
        print(f"Query: {query[:100]}...")
        if nams_context:
            print(f"NAMS Context: {nams_context[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        try:
            memory = self._get_or_create_memory(context_id)
            memory_context = self._get_memory_context(context_id)
            
            # Force direct mode
            memory.socratic_mode = False
            memory.socratic_disabled = True
            memory.socratic_questions_asked = 0
            memory.socratic_answers = []
            memory.original_query = ""
            
            visual_findings, image_embedding = await self._get_visual_findings(images, context_id)
            
            classification = await self.classify_query(
                query, memory_context, visual_findings
            )
            
            search_query = await self.generate_search_query(
                classification, visual_findings, query
            )
            search_results = await self.search_multimodal(
                query=search_query,
                image_embedding=image_embedding,
                top_k=5
            )
            
            document_context = "\n".join([
                f"--- Fragmento {i+1} ---\n{r['payload'].get('text', 'N/A')}"
                for i, r in enumerate(search_results.get('text', []))
            ])
            
            image_context = "\n".join([
                f"--- Imagen {i+1} ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
                for i, r in enumerate(search_results.get('image', []))
            ])
            
            final_response = await self.generate_physics_response(
                query, memory_context, classification, 
                visual_findings, document_context, image_context,
                chat_history=memory.chat_history,
                nams_context=nams_context
            )
            self._save_to_memory(context_id, query, final_response)
            self._save_memories()
            print(f"✅ Diálogo directo completado\n")
            return final_response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"

    @traceable(name="PhysicsMultimodalAgent.stream", run_type="chain", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def stream(self, query: str, context_id: str, 
                    images: List[dict] = None) -> AsyncIterable[dict[str, Any]]:
        """
        🔧 CRÍTICO: Este método DEBE yieldar diccionarios con:
        - 'is_task_complete': bool
        - 'require_user_input': bool
        - 'content': str
        - 'status': str (opcional)
        
        Implementa diálogo directo (modo socrático desactivado).
        """
        # Extract NAMS context from query if present
        nams_context = ""
        if "[NAMS_CONTEXT]" in query and "[/NAMS_CONTEXT]" in query:
            start_tag = "[NAMS_CONTEXT]"
            end_tag = "[/NAMS_CONTEXT]"
            start_idx = query.find(start_tag)
            end_idx = query.find(end_tag)
            nams_context = query[start_idx + len(start_tag):end_idx].strip()
            remainder = query[end_idx + len(end_tag):]
            if remainder.startswith("\n\n"):
                query = remainder[2:]
            elif remainder.startswith("\n"):
                query = remainder[1:]
            else:
                query = remainder

        print(f"\n{'='*80}")
        print(f"📚 Consulta (streaming - Directa)")
        print(f"Query: {query[:100]}...")
        if nams_context:
            print(f"NAMS Context: {nams_context[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        memory = self._get_or_create_memory(context_id)
        memory_context = self._get_memory_context(context_id)
        
        # Force direct mode
        memory.socratic_mode = False
        memory.socratic_disabled = True
        memory.socratic_questions_asked = 0
        memory.socratic_answers = []
        memory.original_query = ""

        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': f'💬 Entablando diálogo normal sin socrático...',
            'status': 'normal_dialogue'
        }
        
        visual_findings, image_embedding = await self._get_visual_findings(images, context_id)
        if images and len(images) > 0:
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '✅ Fenómenos físicos identificados.',
                'status': 'analyzing_images'
            }
        
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '📚 Analizando consulta...',
            'status': 'classifying'
        }
        classification = await self.classify_query(
            query, memory_context, visual_findings
        )
        
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔎 Buscando información complementaria...',
            'status': 'searching_documents'
        }
        search_query = await self.generate_search_query(
            classification, visual_findings, query
        )
        search_results = await self.search_multimodal(
            query=search_query,
            image_embedding=image_embedding,
            top_k=5
        )
        
        document_context = "\n".join([
            f"--- Fragmento {i+1} ---\n{r['payload'].get('text', 'N/A')}"
            for i, r in enumerate(search_results.get('text', []))
        ])
        
        image_context = "\n".join([
            f"--- Imagen {i+1} ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
            for i, r in enumerate(search_results.get('image', []))
        ])
        
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '📝 Generando respuesta...',
            'status': 'generating_response'
        }
        
        final_response = await self.generate_physics_response(
            query, memory_context, classification, 
            visual_findings, document_context, image_context,
            chat_history=memory.chat_history,
            nams_context=nams_context
        )
        self._save_to_memory(context_id, query, final_response)
        self._save_memories()
        
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
            # self.memories[context_id] = SemanticMemory(llm=self.llm) No borrar la entrada, solo limpiar
        if context_id in self.visual_findings:
            del self.visual_findings[context_id]
        self._save_memories()
        print(f"🧹 Memoria limpiada para contexto: {context_id}")

    async def get_memory_summary(self, context_id: str) -> str:
        """Obtiene un resumen de la memoria de un contexto."""
        if context_id not in self.memories:
            return "No hay memoria para este contexto."
        
        memory = self.memories[context_id]
        return f"""
📊 **Resumen de Memoria**
- Interacciones guardadas: {len(memory.conversations)}
- Contexto disponible: {'Sí' if memory.get_context() else 'No'}
- Hallazgos visuales: {'Sí' if context_id in self.visual_findings else 'No'}
"""

# ==================== FUNCIÓN AUXILIAR PARA CARGAR PDFs ====================

async def load_pdfs_from_folder(agent: PhysicsMultimodalAgent, folder_path: str = "pdfs") -> str:
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
        agent = PhysicsMultimodalAgent()
        
        # Cargar PDFs (opcional)
        # temario = await load_pdfs_from_folder(agent, "pdfs")
        # print(f"Temario extraído:\n{temario}")
        
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
