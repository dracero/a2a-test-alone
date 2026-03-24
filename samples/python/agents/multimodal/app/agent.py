# samples/python/agents/multimodal/app/agent.py (CORREGIDO)

import asyncio
import base64
import glob
import os
from collections.abc import AsyncIterable
from io import BytesIO
from pathlib import Path
from typing import Any, List, Literal, Optional

import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

# ==================== CONFIGURACIÓN ====================

GEMINI_MODEL = "gemini-2.5-flash"

class SemanticMemory:
    """Memoria conversacional simplificada sin dependencias deprecadas."""
    
    def __init__(self, llm, max_entries: int = 10):
        self.conversations = []
        self.max_entries = max_entries
        self.summary = ""
        self.direct_history = ""
        self.llm = llm
    
    def add_interaction(self, query: str, response: str):
        """Guardar interacción en memoria."""
        self.conversations.append({"query": query, "response": response})
        
        if len(self.conversations) > self.max_entries:
            self.conversations.pop(0)
        
        # Mantener solo las últimas 3 conversaciones en el historial directo
        if len(self.conversations) > 3:
            recent = self.conversations[-3:]
            self.direct_history = ""
            for conv in recent:
                self.direct_history += f"\nUsuario: {conv['query']}\nAsistente: {conv['response']}\n"
        else:
            self.direct_history = ""
            for conv in self.conversations:
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


class PhysicsMultimodalAgent:
    """Agente de física con procesamiento multimodal."""
    
    SYSTEM_INSTRUCTION = (
        'Eres un profesor experto en Física I de la UBA. '
        'Analizas consultas, imágenes de experimentos y proporciona '
        'explicaciones claras y didácticas.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        """Inicializar el agente de física."""
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=4096,
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
        self.temario = ""
        
        print("✅ PhysicsMultimodalAgent inicializado")
    
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
    
    def split_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Dividir texto en chunks."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
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
        """Procesar PDFs y almacenar en Qdrant."""
        print("\n" + "="*80)
        print("📚 PROCESANDO PDFs")
        print("="*80)
        
        text_points = []
        image_points = []
        global_id_counter = 0
        contenido_completo_texto = ""
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
                continue
            
            print(f"\n📄 Procesando: {Path(pdf_file).name}")
            
            # Texto
            text = self.leer_pdf(pdf_file)
            if text:
                contenido_completo_texto += f"\n--- {Path(pdf_file).name} ---\n{text}"
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
        temario = await self.extraer_temario(contenido_completo_texto)
        
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
        memory = self._get_or_create_memory(context_id)
        memory.add_interaction(query, response)
    
    def encode_image(self, image_data: bytes) -> str:
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_mime_type(self, content_type: str) -> str:
        mapping = {
            'image/jpeg': 'image/jpeg',
            'image/png': 'image/png',
            'image/webp': 'image/webp',
        }
        return mapping.get(content_type, 'image/png')
    
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
        document_context: str,
        image_context: str
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

DOCUMENTOS:
{document_context}

IMÁGENES:
{image_context}

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
        print(f"📚 Consulta de física")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        try:
            memory_context = self._get_memory_context(context_id)
            
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
            else:
                visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
            
            # Clasificar
            print(f"🔍 Clasificando...")
            classification = await self.classify_query(query, memory_context, visual_findings)
            
            # Buscar
            print(f"🔎 Buscando...")
            search_query = await self.generate_search_query(
                classification, visual_findings, query
            )
            search_results = await self.search_multimodal(
                query=search_query,
                image_embedding=image_embedding,
                top_k=5
            )
            
            # Contexto
            document_context = "\n".join([
                f"--- Fragmento {i+1} ---\n{r['payload'].get('text', 'N/A')}"
                for i, r in enumerate(search_results.get('text', []))
            ])
            
            image_context = "\n".join([
                f"--- Imagen {i+1} ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
                for i, r in enumerate(search_results.get('image', []))
            ])
            
            # Respuesta
            print(f"📝 Generando respuesta...")
            final_response = await self.generate_physics_response(
                query, memory_context, classification, visual_findings,
                document_context, image_context
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
        🔧 CRÍTICO: Este método DEBE yieldar diccionarios con:
        - 'is_task_complete': bool
        - 'require_user_input': bool
        - 'content': str
        - 'status': str (opcional)
        """
        print(f"\n{'='*80}")
        print(f"📚 Consulta (streaming)")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        memory_context = self._get_memory_context(context_id)
        
        # Analizar imágenes
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
        
        # Buscar
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔎 Buscando en documentos...',
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

        # Contexto
        document_context = "\n".join([
            f"--- Fragmento {i+1} ---\n{r['payload'].get('text', 'N/A')}"
            for i, r in enumerate(search_results.get('text', []))
        ])

        image_context = "\n".join([
            f"--- Imagen {i+1} ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
            for i, r in enumerate(search_results.get('image', []))
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
            document_context, image_context
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
