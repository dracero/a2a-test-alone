# samples/python/agents/multimodal/app/agent.py

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
from langchain_groq import ChatGroq
from langsmith import traceable
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

# ==================== CONFIGURACIÓN ====================

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=4096,
            api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Cargar prompts optimizados (si existen)
        self._optimized_prompts = self._load_optimized_prompts()
        self.SYSTEM_INSTRUCTION = self._get_prompt(
            "system_instruction", self._DEFAULT_SYSTEM_INSTRUCTION
        )
        
        # Qdrant
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_KEY", "")
        self.text_collection = "documentos_pdf_texto"
        self.image_collection = "documentos_pdf_imagenes"
        self.multimodal_collection = "documentos_multimodal"
        
        # Modelo CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Memoria conversacional
        self.memories = {}
        self.visual_findings = {}
        self._memories_file = "/tmp/physics_agent_memories.json"
        
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
        """Extraer imágenes de un PDF."""
        import os
        from pathlib import Path
        
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
    
    def split_text(self, text: str, chunk_words: int = 50, overlap: int = 10) -> List[str]:
        """Dividir texto en chunks de palabras (máx 77 tokens para CLIP)."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_words - overlap):
            chunk = " ".join(words[i:i + chunk_words])
            chunks.append(chunk)
            if i + chunk_words >= len(words):
                break
        return chunks
    
    def generate_text_embeddings_batch(self, chunks: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generar embeddings de texto en batch."""
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inputs = self.clip_processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.clip_model.device)
            
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)
                # Extraer el tensor del output (puede ser un objeto BaseModelOutputWithPooling)
                if hasattr(outputs, 'pooler_output'):
                    text_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    text_features = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    text_features = outputs  # Ya es un tensor
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
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generar embedding de texto."""
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
                # Extraer el tensor del output
                if hasattr(outputs, 'pooler_output'):
                    text_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    text_features = outputs.last_hidden_state[:, 0]
                else:
                    text_features = outputs
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    async def store_in_qdrant(self, points: List[Any], collection_name: str):
        """Almacenar puntos en Qdrant."""
        client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=60.0)
        try:
            await client.get_collection(collection_name)
            print(f"📦 Colección '{collection_name}' existe")
        except Exception:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            print(f"✨ Colección '{collection_name}' creada")
            
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
            response = self.llm.invoke(messages)
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
                embeddings = self.generate_text_embeddings_batch(chunks)
                
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
            print(f"   🖼️ {len(imagenes)} imágenes")
            
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
        
        # Guardar temario a disco
        try:
            with open("temario.txt", "w", encoding="utf-8") as f:
                f.write(temario)
        except Exception as e:
            print(f"⚠️ No se pudo guardar el temario a disco: {e}")
        
        # Almacenar
        if text_points:
            await self.store_in_qdrant(text_points, self.text_collection)
        if image_points:
            await self.store_in_qdrant(image_points, self.image_collection)
        
        print("\n✅ PROCESAMIENTO COMPLETADO")
        print(f"   📝 Texto: {len(text_points)} chunks")
        print(f"   🖼️ Imágenes: {len(image_points)} embeddings")
        
        self.temario = temario
        return temario
    
    # ==================== MÉTODOS DE ANÁLISIS ====================
    
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
    
    @traceable(name="search_qdrant", run_type="retriever")
    async def search_multimodal(
        self, 
        query: str = None, 
        image_embedding: List[float] = None,
        top_k: int = 5
    ) -> dict[str, List[dict]]:
        """Búsqueda en Qdrant."""
        client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        results = {"text": [], "image": []}
        
        try:
            if query and image_embedding:
                collections = [self.text_collection, self.image_collection]
                search_embedding = self.generate_text_embedding(query)
            elif query:
                collections = [self.text_collection]
                search_embedding = self.generate_text_embedding(query)
            elif image_embedding:
                collections = [self.image_collection]
                search_embedding = image_embedding
            else:
                return results
            
            if not search_embedding:
                return results
            
            for collection in collections:
                try:
                    search_results = await client.query_points(
                        collection_name=collection,
                        query=search_embedding,
                        limit=top_k
                    )
                    col_type = collection.split("_")[-1]
                    # query_points returns a QueryResponse object with a 'points' attribute
                    points = search_results.points if hasattr(search_results, 'points') else search_results
                    results[col_type] = [{
                        "id": r.id,
                        "score": round(r.score, 4),
                        "payload": r.payload
                    } for r in points]
                except Exception as e:
                    print(f"⚠️ Error en {collection}: {e}")
            
            return results
        except Exception as e:
            print(f"❌ Error: {e}")
            return results
    
    @traceable(name="analyze_physics_image", run_type="llm")
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
    
    @traceable(name="classify_query", run_type="llm")
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
    
    @traceable(name="generate_search_query", run_type="llm")
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
    
    @traceable(name="generate_physics_response", run_type="llm")
    async def generate_physics_response(
        self, 
        query: str, 
        context: str, 
        classification: str, 
        visual_findings: str,
        document_context: str,
        image_context: str,
        chat_history: list = None
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
        
        # Agregar contexto de documentos al system prompt
        if document_context and document_context.strip():
            system_prompt += f"\n\nDOCUMENTOS DE REFERENCIA:\n{document_context[:3000]}"
        if image_context and image_context.strip():
            system_prompt += f"\n\nIMÁGENES RELACIONADAS:\n{image_context[:1000]}"
        if visual_findings and visual_findings.strip() and visual_findings != "No hay imágenes.":
            system_prompt += f"\n\nHALLAZGOS VISUALES:\n{visual_findings[:1000]}"
        if classification and classification.strip():
            system_prompt += f"\n\nCLASIFICACIÓN DEL TEMA:\n{classification[:500]}"
        
        try:
            messages = [SystemMessage(content=system_prompt)]
            
            # Añadir el historial real de chat (lo que se habló antes)
            if chat_history:
                messages.extend(chat_history)
            
            # El mensaje actual del usuario
            messages.append(HumanMessage(content=query))
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @traceable(name="check_socratic_intent", run_type="llm")
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
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.upper()
            if "SALIR" in content: return "SALIR"
            if "ENTRAR" in content: return "ENTRAR"
            return "CONTINUAR"
        except Exception:
            return "CONTINUAR"
    
    @traceable(name="generate_socratic_question", run_type="llm")
    async def generate_socratic_question(
        self,
        original_query: str,
        question_number: int,
        previous_answers: List[str],
        visual_findings: str = ""
    ) -> str:
        """Genera una pregunta socrática para guiar al estudiante."""
        visual_section = ""
        if visual_findings:
            visual_section = f"""\nHALLAZGOS VISUALES (de imágenes proporcionadas por el estudiante):
{visual_findings}
- Incorpora lo que se observa en las imágenes en tus preguntas.
- Pregunta al estudiante qué fenómenos físicos identifica en la imagen.
"""

        _default_socratic = f"""Eres un tutor socrático de Física I de la UBA.

Tu objetivo es guiar al estudiante a descubrir la respuesta por sí mismo mediante preguntas.
Recibís tanto texto como imágenes de experimentos, diagramas y problemas de física.

TEMARIO:
{self.temario}
{visual_section}

Reglas para las preguntas:
- BASA TUS PREGUNTAS ÚNICAMENTE en el contenido del TEMARIO y los documentos. Si el tema no pertenece al temario, informa al estudiante que el tema está fuera de alcance.
- Pregunta {question_number + 1}/3
- Haz preguntas que activen el pensamiento crítico
- Si el estudiante envió una imagen, preguntá sobre lo que se observa en ella
- Relaciona con conceptos fundamentales
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
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @traceable(name="generate_physics_response_with_socratic", run_type="llm")
    async def generate_physics_response_with_socratic(
        self,
        query: str,
        context: str,
        classification: str,
        visual_findings: str,
        document_context: str,
        image_context: str,
        student_answers: str
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
            response = self.llm.invoke(messages)
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
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                result = response.content.strip().upper()
                print(f"🧠 [LLM] Intent socrático: '{query[:50]}' → {result}")
                
                if "SALIR" in result:
                    return "SALIR"
            except Exception as e:
                print(f"⚠️ Error en LLM intent: {e}")
        
        return "CONTINUAR"
    
    # ==================== MÉTODOS PRINCIPALES ====================
    
    @traceable(name="PhysicsMultimodalAgent.invoke", run_type="chain")
    async def invoke(self, query: str, context_id: str, 
                    images: List[dict] = None) -> str:
        """Procesa consulta completa con modo socrático."""
        print(f"\n{'='*80}")
        print(f"📚 Consulta de física")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        try:
            memory = self._get_or_create_memory(context_id)
            memory_context = self._get_memory_context(context_id)
            
            query_for_generation = query  # Default; may be overridden by socratic exit
            exiting_socratic = False  # Flag to track if we're exiting socratic mode
            
            # === CASO 1: Estamos en modo socrático (respondiendo preguntas) ===
            if memory.socratic_mode:
                # Detección unificada: keywords rápidos + LLM fallback
                intent = await self._detect_socratic_intent(query, is_in_socratic_mode=True)
                wants_exit = (intent == "SALIR")
                
                if wants_exit:
                    print("🚪 Salida del modo socrático detectada.")
                    memory.socratic_disabled = True
                    memory.socratic_mode = False
                    query_for_generation = memory.original_query if memory.original_query else query
                    exiting_socratic = True
                    memory.socratic_questions_asked = 0
                    memory.socratic_answers = []
                    memory.original_query = ""
                    # Caer al flujo de respuesta directa abajo (CASO 2)
                else:
                    # Procesar como respuesta socrática normal
                    memory.socratic_answers.append(query)
                    memory.socratic_questions_asked += 1
                    
                    print(f"🎓 Modo socrático: {memory.socratic_questions_asked}/3 preguntas respondidas")
                    
                    # Si ya respondió las 3 preguntas, dar la respuesta completa
                    if memory.socratic_questions_asked >= 3:
                        print(f"✅ Completadas las 3 preguntas, generando respuesta final...")
                        
                        visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
                        image_embedding = None
                        
                        classification = await self.classify_query(
                            memory.original_query, memory_context, visual_findings
                        )
                        
                        search_query = await self.generate_search_query(
                            classification, visual_findings, memory.original_query
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
                        
                        student_answers_summary = "\n".join([
                            f"Pregunta {i+1}: {ans}"
                            for i, ans in enumerate(memory.socratic_answers)
                        ])
                        
                        final_response = await self.generate_physics_response_with_socratic(
                            memory.original_query, memory_context, classification, 
                            visual_findings, document_context, image_context,
                            student_answers_summary
                        )
                        
                        # LaTeX rendering is handled by KaTeX in the frontend
                        # Resetear modo socrático
                        memory.socratic_mode = False
                        memory.socratic_questions_asked = 0
                        memory.socratic_answers = []
                        memory.original_query = ""
                        
                        self._save_to_memory(context_id, query, final_response)
                        print(f"✅ Completado\n")
                        
                        return final_response
                    else:
                        # Generar siguiente pregunta socrática
                        next_question = await self.generate_socratic_question(
                            memory.original_query,
                            memory.socratic_questions_asked,
                            memory.socratic_answers,
                            visual_findings=self.visual_findings.get(context_id, "")
                        )
                        
                        # LaTeX rendering is handled by KaTeX in the frontend
                        return next_question + "\n\n<!-- SOCRATIC_EXIT -->"
            
            # === CASO 2: No estamos en modo socrático ===
            # Si venimos del CASO 1 con exit, query_for_generation ya tiene el original_query
            if not exiting_socratic:
                query_for_generation = query
            
            # Verificar si el usuario quiere volver al modo socrático
            # SOLO si NO estamos saliendo del socrático
            if not exiting_socratic:
                enter_intent = await self._detect_socratic_intent(query, is_in_socratic_mode=False)
                if enter_intent == "ENTRAR":
                    print("🎓 El usuario decidió volver al modo socrático.")
                    memory.socratic_disabled = False
            
            # Si el modo socrático está deshabilitado, entablar diálogo normal
            if memory.socratic_disabled:
                print(f"💬 Entablando diálogo normal sin socrático...")
                
                # Analizar imágenes si las hay
                visual_findings = ""
                image_embedding = None
                
                if images and len(images) > 0:
                    print(f"🖼️ Analizando imágenes para diálogo...")
                    visual_findings = await self.analyze_physics_image(images)
                    self.visual_findings[context_id] = visual_findings
                    
                    first_image_data = images[0].get('data') or images[0].get('bytes')
                    if isinstance(first_image_data, str):
                        first_image_data = base64.b64decode(first_image_data)
                    image_embedding = self.generate_image_embedding(first_image_data)
                else:
                    visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
                
                classification = await self.classify_query(
                    query_for_generation, memory_context, visual_findings
                )
                
                search_query = await self.generate_search_query(
                    classification, visual_findings, query_for_generation
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
                    query_for_generation, memory_context, classification, 
                    visual_findings, document_context, image_context,
                    chat_history=memory.chat_history
                )
                self._save_to_memory(context_id, query, final_response)
                print(f"✅ Diálogo completado\n")
                return final_response

            # Modo normal: iniciar modo socrático
            print(f"🎓 Iniciando modo socrático...")
            
            # Analizar imágenes
            visual_findings = ""
            image_embedding = None
            
            if images and len(images) > 0:
                print(f"🖼️ Analizando imágenes...")
                visual_findings = await self.analyze_physics_image(images)
                self.visual_findings[context_id] = visual_findings
                
                first_image_data = images[0].get('data') or images[0].get('bytes')
                if isinstance(first_image_data, str):
                    first_image_data = base64.b64decode(first_image_data)
                image_embedding = self.generate_image_embedding(first_image_data)
            
            # Activar modo socrático
            memory.socratic_mode = True
            memory.original_query = query
            memory.socratic_questions_asked = 0
            memory.socratic_answers = []
            
            # Generar primera pregunta socrática
            first_question = await self.generate_socratic_question(
                query, 0, [],
                visual_findings=visual_findings
            )
            
            # LaTeX rendering is handled by KaTeX in the frontend
            return first_question
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"
    
    @traceable(name="PhysicsMultimodalAgent.stream", run_type="chain")
    async def stream(self, query: str, context_id: str, 
                    images: List[dict] = None) -> AsyncIterable[dict[str, Any]]:
        """
        🔧 CRÍTICO: Este método DEBE yieldar diccionarios con:
        - 'is_task_complete': bool
        - 'require_user_input': bool
        - 'content': str
        - 'status': str (opcional)
        
        Implementa modo socrático con 3 preguntas antes de la respuesta.
        """
        print(f"\n{'='*80}")
        print(f"📚 Consulta (streaming)")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        memory = self._get_or_create_memory(context_id)
        memory_context = self._get_memory_context(context_id)
        
        query_for_generation = query  # Default; may be overridden by socratic exit
        exiting_socratic = False  # Flag to track if we're exiting socratic mode
        
        # === CASO 1: Estamos en modo socrático (respondiendo preguntas) ===
        if memory.socratic_mode:
            print(f"🔍 [SOCRATIC] Query: '{query}'")
            
            # Detección unificada: keywords rápidos + LLM fallback
            intent = await self._detect_socratic_intent(query, is_in_socratic_mode=True)
            wants_exit = (intent == "SALIR")
            print(f"🔍 [SOCRATIC] Intent: {intent}, wants_exit: {wants_exit}")
            
            if wants_exit:
                print("🚪 Salida del modo socrático detectada.")
                memory.socratic_disabled = True
                memory.socratic_mode = False
                query_for_generation = memory.original_query if memory.original_query else query
                exiting_socratic = True
                memory.socratic_questions_asked = 0
                memory.socratic_answers = []
                memory.original_query = ""
                print(f"🔍 [SOCRATIC] Exit complete. Falling to CASO 2.")
                # Caer al flujo de respuesta directa abajo (CASO 2)
            else:
                # Procesar como respuesta socrática normal
                memory.socratic_answers.append(query)
                memory.socratic_questions_asked += 1
                
                print(f"🎓 Modo socrático: {memory.socratic_questions_asked}/3 preguntas respondidas")
                
                # Si ya respondió las 3 preguntas, dar la respuesta completa
                if memory.socratic_questions_asked >= 3:
                    print(f"✅ Completadas las 3 preguntas, generando respuesta final...")
                    
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': '🎓 Excelente! Has completado las 3 preguntas. Ahora te daré la explicación completa...',
                        'status': 'socratic_complete'
                    }
                    
                    visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
                    image_embedding = None
                    
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': '📚 Analizando tu proceso de pensamiento...',
                        'status': 'classifying'
                    }
                    
                    classification = await self.classify_query(
                        memory.original_query, memory_context, visual_findings
                    )
                    
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': '🔎 Buscando información complementaria...',
                        'status': 'searching_documents'
                    }
                    
                    search_query = await self.generate_search_query(
                        classification, visual_findings, memory.original_query
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
                        'content': '📝 Generando explicación completa basada en tus respuestas...',
                        'status': 'generating_response'
                    }
                    
                    student_answers_summary = "\n".join([
                        f"Pregunta {i+1}: {ans}"
                        for i, ans in enumerate(memory.socratic_answers)
                    ])
                    
                    final_response = await self.generate_physics_response_with_socratic(
                        memory.original_query, memory_context, classification,
                        visual_findings, document_context, image_context,
                        student_answers_summary
                    )
                    
                    # LaTeX rendering is handled by KaTeX in the frontend
                    # Resetear modo socrático
                    memory.socratic_mode = False
                    memory.socratic_questions_asked = 0
                    memory.socratic_answers = []
                    memory.original_query = ""
                    
                    self._save_to_memory(context_id, query, final_response)
                    
                    yield {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': final_response,
                        'status': 'completed'
                    }
                else:
                    # Generar siguiente pregunta socrática
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': f'💭 Procesando tu respuesta {memory.socratic_questions_asked}/3...',
                        'status': 'socratic_processing'
                    }
                    
                    next_question = await self.generate_socratic_question(
                        memory.original_query,
                        memory.socratic_questions_asked,
                        memory.socratic_answers,
                        visual_findings=self.visual_findings.get(context_id, "")
                    )
                    
                    # Guardar el intercambio socrático en el chat_history real
                    last_answer = memory.socratic_answers[-1] if memory.socratic_answers else ""
                    memory.add_socratic_exchange(next_question, last_answer)
                    
                    # LaTeX rendering is handled by KaTeX in the frontend
                    # CRÍTICO: is_task_complete=False + require_user_input=True
                    yield {
                        'is_task_complete': False,
                        'require_user_input': True,
                        'content': next_question + "\n\n<!-- SOCRATIC_EXIT -->",
                        'status': 'socratic_question'
                    }
                # Return temprano: ya procesamos la respuesta socrática (continuar o completar)
                return
        
        # === CASO 2: No estamos en modo socrático ===
        print(f"🔍 [CASO2] Reached. exiting_socratic={exiting_socratic}, socratic_disabled={memory.socratic_disabled}")
        # Si venimos del CASO 1 con exit, query_for_generation ya tiene el original_query
        if not exiting_socratic:
            query_for_generation = query
        print(f"🔍 [CASO2] query_for_generation='{query_for_generation[:80]}'")
        
        # === Detectar comandos de botón del frontend ===
        query_stripped = query.strip()
        
        if query_stripped == "[SOCRATIC]":
            # El estudiante eligió modo socrático via botón
            print("🎓 Estudiante eligió SOCRÁTICO via botón")
            original_q = memory.original_query if memory.original_query else query
            
            memory.socratic_mode = True
            memory.socratic_questions_asked = 0
            memory.socratic_answers = []
            memory.socratic_disabled = False
            
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🎓 Iniciando método socrático: te haré 3 preguntas para guiar tu aprendizaje...',
                'status': 'socratic_init'
            }
            
            first_question = await self.generate_socratic_question(
                original_q, 0, [],
                visual_findings=self.visual_findings.get(context_id, "")
            )
            
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': first_question + "\n\n<!-- SOCRATIC_EXIT -->",
                'status': 'socratic_question'
            }
            return
        
        if query_stripped == "[DIRECTO]":
            # El estudiante eligió explicación directa via botón
            print("📖 Estudiante eligió DIRECTO via botón")
            memory.socratic_disabled = True
            memory.socratic_mode = False
            query_for_generation = memory.original_query if memory.original_query else query
            # Continuar al flujo de respuesta directa abajo
        
        # Verificar si el usuario quiere volver al modo socrático
        # SOLO si NO estamos saliendo del socrático
        if not exiting_socratic and query_stripped not in ("[SOCRATIC]", "[DIRECTO]"):
            enter_intent = await self._detect_socratic_intent(query, is_in_socratic_mode=False)
            if enter_intent == "ENTRAR":
                print("🎓 El usuario decidió volver al modo socrático.")
                memory.socratic_disabled = False
        
        # Si el modo socrático está deshabilitado, entablar diálogo normal
        if memory.socratic_disabled:
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f'💬 Entablando diálogo normal sin socrático...',
                'status': 'normal_dialogue'
            }
            
            visual_findings = ""
            image_embedding = None
            
            if images and len(images) > 0:
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f'🖼️ Analizando {len(images)} imagen(es) para diálogo...',
                    'status': 'analyzing_images'
                }
                
                visual_findings = await self.analyze_physics_image(images)
                self.visual_findings[context_id] = visual_findings
                
                first_image_data = images[0].get('data') or images[0].get('bytes')
                if isinstance(first_image_data, str):
                    first_image_data = base64.b64decode(first_image_data)
                image_embedding = self.generate_image_embedding(first_image_data)
                
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '✅ Fenómenos físicos identificados.',
                    'status': 'analyzing_images'
                }
            else:
                visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
            
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '📚 Analizando consulta...',
                'status': 'classifying'
            }
            classification = await self.classify_query(
                query_for_generation, memory_context, visual_findings
            )
            
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🔎 Buscando información complementaria...',
                'status': 'searching_documents'
            }
            search_query = await self.generate_search_query(
                classification, visual_findings, query_for_generation
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
                query_for_generation, memory_context, classification, 
                visual_findings, document_context, image_context,
                chat_history=memory.chat_history
            )
            self._save_to_memory(context_id, query, final_response)
            
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': final_response,
                'status': 'completed'
            }

        else:
            # === Primer contacto: presentar opciones al estudiante ===
            print(f"🎓 Primera consulta. Presentando opciones...")
            
            visual_findings = ""
            image_embedding = None
            
            if images and len(images) > 0:
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f'🖼️ Analizando {len(images)} imagen(es)...',
                    'status': 'analyzing_images'
                }
                
                visual_findings = await self.analyze_physics_image(images)
                self.visual_findings[context_id] = visual_findings
                
                first_image_data = images[0].get('data') or images[0].get('bytes')
                if isinstance(first_image_data, str):
                    first_image_data = base64.b64decode(first_image_data)
                image_embedding = self.generate_image_embedding(first_image_data)
                
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '✅ Fenómenos físicos identificados.',
                    'status': 'analyzing_images'
                }
            
            # Guardar query original por si elige socrático
            memory.original_query = query
            
            # Enviar mensaje con marcador para que el frontend muestre botones
            choice_message = "📚 **Tu consulta:** *" + query + "*\n\n¿Cómo preferís aprender este tema?\n\n<!-- SOCRATIC_CHOICE -->"
            
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': choice_message,
                'status': 'awaiting_choice'
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
