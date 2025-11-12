import asyncio
import base64
import glob
import os
from collections.abc import AsyncIterable
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterable, List, Literal, Optional

import torch
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

# ==================== CONFIGURACIÓN ====================

# Carpeta fija para PDFs
PDF_FOLDER = "/home/cetec/Downloads/apuntes_fisica"

# Modelo correcto de Gemini
GEMINI_MODEL = "gemini-2.5-flash"

class PhysicsResponseFormat(BaseModel):
    """Formato de respuesta de física estructurada."""
    
    status: Literal[
        'analyzing_images', 
        'classifying', 
        'searching_documents',
        'searching_images',
        'generating_response', 
        'completed', 
        'error'
    ] = 'analyzing_images'
    message: str
    section: Literal[
        'visual_analysis', 
        'classification', 
        'document_search',
        'image_search',
        'final_response', 
        'general'
    ] = 'general'


class SemanticMemory:
    """Memoria conversacional con resumen semántico."""
    
    def __init__(self, llm, max_entries: int = 10):
        self.conversations = []
        self.max_entries = max_entries
        self.summary = ""
        self.direct_history = ""
        self.llm = llm
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True
        )
    
    def add_interaction(self, query: str, response: str):
        """Guardar interacción en memoria."""
        self.memory.save_context({"input": query}, {"output": response})
        self.conversations.append({"query": query, "response": response})
        
        if len(self.conversations) > self.max_entries:
            self.conversations.pop(0)
        
        self.direct_history += f"\nUsuario: {query}\nAsistente: {response}\n"
        
        # Mantener solo las últimas 3 conversaciones en el historial directo
        if len(self.conversations) > 3:
            recent = self.conversations[-3:]
            self.direct_history = ""
            for conv in recent:
                self.direct_history += f"\nUsuario: {conv['query']}\nAsistente: {conv['response']}\n"
        
        self.update_summary()
    
    def update_summary(self):
        """Actualizar resumen de conversaciones recientes."""
        if self.conversations:
            recent_context = "\n".join([
                f"Q: {c['query']}\nA: {c['response']}"
                for c in self.conversations[-5:]
            ])
            self.summary = f"Resumen de conversaciones recientes:\n{recent_context}"
    
    def get_context(self) -> str:
        """Obtener contexto completo de la memoria."""
        return f"{self.summary}\n\nHistorial directo:\n{self.direct_history}"
    
    def clear(self):
        """Limpiar la memoria."""
        self.conversations = []
        self.summary = ""
        self.direct_history = ""
        self.memory.clear()


class PhysicsMultimodalAgent:
    """Agente de física con procesamiento multimodal (texto + imágenes) y búsqueda en Qdrant."""
    
    SYSTEM_INSTRUCTION = (
        'Eres un profesor experto en Física I de la Universidad de Buenos Aires. '
        'Analizas consultas de física, imágenes de experimentos, diagramas y proporciona '
        'explicaciones claras y didácticas. Utilizas el temario y los documentos disponibles '
        'para responder de manera precisa. Siempre relacionas la teoría con las imágenes cuando '
        'están disponibles.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    
    def __init__(self):
        """Inicializar el agente de física."""
        # Modelo principal (corregido)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=4096,
        )
        
        # Qdrant para búsqueda vectorial
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.text_collection = "documentos_pdf_texto"
        self.image_collection = "documentos_pdf_imagenes"
        self.multimodal_collection = "documentos_multimodal"
        
        # Modelo CLIP para embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Memoria conversacional por contexto
        self.memories = {}
        
        # Almacenamiento temporal de hallazgos visuales y temario
        self.visual_findings = {}
        self.temario = ""
        
        print("✅ PhysicsMultimodalAgent inicializado")
    
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
        """Extraer imágenes de un PDF usando pdf2image."""
        import os
        from pathlib import Path
        
        os.makedirs(output_folder, exist_ok=True)
        imagenes = []
        
        try:
            from pdf2image import convert_from_path

            # Convertir páginas a imágenes
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
        
        except ImportError:
            print("⚠️ pdf2image no instalado. Instalando...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'pdf2image'])
            subprocess.check_call(['apt-get', 'install', '-y', 'poppler-utils'])
            return self.extraer_imagenes_pdf(pdf_path, output_folder)
        except Exception as e:
            print(f"❌ Error extrayendo imágenes de {pdf_path}: {e}")
            return []
    
    def split_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Dividir texto en chunks."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    def generate_text_embeddings_batch(self, chunks: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generar embeddings de texto en batch usando CLIP."""
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
                text_features = self.clip_model.get_text_features(**inputs)
            
            embeddings.extend(text_features.cpu().numpy().tolist())
        
        return embeddings
    
    def generate_image_embedding(self, image_data: bytes) -> Optional[List[float]]:
        """Generar embedding de una imagen usando CLIP."""
        try:
            from PIL import Image
            
            image = Image.open(BytesIO(image_data)).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.clip_model.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"❌ Error generando embedding de imagen: {e}")
            return None
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generar embedding de texto usando CLIP."""
        try:
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.clip_model.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"❌ Error generando embedding de texto: {e}")
            return None
    
    def generate_multimodal_embedding(
        self, 
        text: str, 
        image_path: str = None, 
        text_weight: float = 0.5
    ) -> Optional[List[float]]:
        """Generar embedding combinado de texto e imagen."""
        import numpy as np
        embeddings = []
        
        # Embedding de texto
        if text:
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.clip_model.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            text_emb = text_features.cpu().numpy().flatten()
            embeddings.append(text_emb * text_weight)
        
        # Embedding de imagen
        if image_path:
            try:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.clip_model.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                
                image_emb = image_features.cpu().numpy().flatten()
                embeddings.append(image_emb * (1 - text_weight))
            except Exception as e:
                print(f"⚠️ Error procesando imagen {image_path}: {e}")
        
        if embeddings:
            combined = np.sum(embeddings, axis=0) / len(embeddings)
            return combined.tolist()
        
        return None
    
    async def store_in_qdrant(self, points: List[Any], collection_name: str):
        """Almacenar puntos en Qdrant."""
        client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
        try:
            await client.get_collection(collection_name)
            print(f"📦 Colección '{collection_name}' ya existe")
        except Exception:
            # CLIP usa 512 dimensiones
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            print(f"✨ Colección '{collection_name}' creada (512 dims)")
        
        await client.upsert(collection_name=collection_name, points=points, wait=True)
        print(f"✅ {len(points)} elementos almacenados en '{collection_name}'")
    
    async def extraer_temario(self, contenido_completo: str) -> str:
        """Extraer temario de los PDFs usando Gemini."""
        print("🤖 Extrayendo temario con Gemini...")
        
        # Limitar contenido para no exceder tokens
        contenido_limitado = contenido_completo[:15000]
        
        system_message = f"""Eres un experto profesor de Física I de la Universidad de Buenos Aires.
Analiza el siguiente contenido de los PDFs del curso y extrae el TEMARIO COMPLETO.

Formato esperado:
TEMA 1: [Título]
- Subtema 1.1: [Descripción]
- Subtema 1.2: [Descripción]

TEMA 2: [Título]
- Subtema 2.1: [Descripción]
...

Incluye todos los temas principales y subtemas que encuentres.

Contenido de los PDFs:
---
{contenido_limitado}
---
"""
        
        user_question = "Extrae el temario completo y estructurado del curso de Física I basándote en el contenido proporcionado."
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_question),
        ]
        
        try:
            response = self.llm.invoke(messages)
            temario = response.content
            print(f"✅ Temario extraído ({len(temario)} caracteres)")
            return temario
        except Exception as e:
            print(f"⚠️ Error extrayendo temario: {e}")
            return "Temario no disponible. Error en extracción."
    
    async def procesar_y_almacenar_pdfs(self, pdf_files: Optional[List[str]] = None) -> str:
        """
        Procesar PDFs, extraer temario y almacenar en Qdrant.
        Si no se especifican archivos, usa todos los PDFs de PDF_FOLDER.
        
        Returns:
            Temario extraído como string
        """
        
        # Si no se especifican archivos, buscar en la carpeta fija
        if not pdf_files:
            pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
            if not pdf_files:
                print(f"⚠️ No se encontraron PDFs en {PDF_FOLDER}")
                return "Temario no disponible. No se encontraron PDFs."
        
        print("\n" + "="*80)
        print("📚 PROCESANDO PDFs DE FÍSICA")
        print("="*80)
        
        text_points = []
        image_points = []
        multimodal_points = []
        global_id_counter = 0
        
        # Para extraer el temario
        contenido_completo_texto = ""
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
                continue
            
            print(f"\n📄 Procesando: {Path(pdf_file).name}")
            
            # ===== PROCESAR TEXTO =====
            text = self.leer_pdf(pdf_file)
            if text:
                contenido_completo_texto += f"\n--- {Path(pdf_file).name} ---\n{text}"
                
                chunks = self.split_text(text)
                print(f"   📝 {len(chunks)} chunks de texto")
                
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
            
            # ===== PROCESAR IMÁGENES =====
            imagenes = self.extraer_imagenes_pdf(pdf_file)
            print(f"   🖼️ {len(imagenes)} imágenes extraídas")
            
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
                    
                    # ===== EMBEDDING MULTIMODAL =====
                    context_text = text[:200] if text else ""
                    multimodal_emb = self.generate_multimodal_embedding(context_text, img_path)
                    if multimodal_emb:
                        multimodal_points.append(PointStruct(
                            id=global_id_counter + 10000,
                            vector=multimodal_emb,
                            payload={
                                "pdf_name": pdf_file,
                                "type": "multimodal",
                                "image_path": img_path,
                                "text_context": context_text
                            }
                        ))
                    
                    global_id_counter += 1
        
        # ===== EXTRAER TEMARIO =====
        print("\n" + "="*80)
        print("📋 EXTRAYENDO TEMARIO")
        print("="*80)
        
        temario = await self.extraer_temario(contenido_completo_texto)
        
        # ===== ALMACENAR EN QDRANT =====
        print("\n" + "="*80)
        print("💾 ALMACENANDO EN QDRANT")
        print("="*80)
        
        if text_points:
            await self.store_in_qdrant(text_points, self.text_collection)
        
        if image_points:
            await self.store_in_qdrant(image_points, self.image_collection)
        
        if multimodal_points:
            await self.store_in_qdrant(multimodal_points, self.multimodal_collection)
        
        print("\n" + "="*80)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("="*80)
        print(f"   📝 Texto: {len(text_points)} chunks")
        print(f"   🖼️ Imágenes: {len(image_points)} embeddings")
        print(f"   🔗 Multimodal: {len(multimodal_points)} embeddings")
        print(f"   📋 Temario: {len(temario)} caracteres")
        
        # Guardar temario internamente
        self.temario = temario
        
        return temario
    
    def set_temario(self, temario: str):
        """Establecer el temario de física."""
        self.temario = temario
        print(f"✅ Temario actualizado ({len(temario)} caracteres)")
    
    def _get_or_create_memory(self, context_id: str) -> SemanticMemory:
        """Obtener o crear memoria para un contexto específico."""
        if context_id not in self.memories:
            self.memories[context_id] = SemanticMemory(llm=self.llm)
        return self.memories[context_id]
    
    def _get_memory_context(self, context_id: str) -> str:
        """Obtener el contexto de memoria para un contexto específico."""
        memory = self._get_or_create_memory(context_id)
        return memory.get_context()
    
    def _save_to_memory(self, context_id: str, query: str, response: str):
        """Guardar interacción en memoria."""
        memory = self._get_or_create_memory(context_id)
        memory.add_interaction(query, response)
    
    def encode_image(self, image_data: bytes) -> str:
        """Codifica imagen en base64."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_mime_type(self, content_type: str) -> str:
        """Mapea content_type a MIME type para Gemini."""
        mapping = {
            'image/jpeg': 'image/jpeg',
            'image/png': 'image/png',
            'image/webp': 'image/webp',
            'image/gif': 'image/gif',
        }
        return mapping.get(content_type, 'image/png')
    
    async def search_multimodal(
        self, 
        query: str = None, 
        image_embedding: List[float] = None,
        top_k: int = 5
    ) -> dict[str, List[dict]]:
        """Búsqueda multimodal en Qdrant."""
        client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        results = {"text": [], "image": [], "multimodal": []}
        
        try:
            # Determinar qué colecciones buscar
            if query and image_embedding:
                collections = [self.text_collection, self.image_collection, self.multimodal_collection]
                search_embedding = self.generate_text_embedding(query)
            elif query:
                collections = [self.text_collection, self.multimodal_collection]
                search_embedding = self.generate_text_embedding(query)
            elif image_embedding:
                collections = [self.image_collection, self.multimodal_collection]
                search_embedding = image_embedding
            else:
                print("⚠️ Debe proporcionar query o image_embedding")
                return results
            
            if not search_embedding:
                print("⚠️ No se pudo generar embedding de búsqueda")
                return results
            
            # Buscar en cada colección
            for collection in collections:
                try:
                    search_results = await client.search(
                        collection_name=collection,
                        query_vector=search_embedding,
                        limit=top_k
                    )
                    
                    col_type = collection.split("_")[-1]
                    results[col_type] = [{
                        "id": r.id,
                        "score": round(r.score, 4),
                        "payload": r.payload
                    } for r in search_results]
                    
                except Exception as e:
                    print(f"⚠️ Error buscando en {collection}: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda multimodal: {e}")
            return results
    
    async def analyze_physics_image(self, images: List[dict]) -> str:
        """
        Analiza imágenes de física con Gemini Vision.
        
        Args:
            images: Lista de diccionarios con:
                - 'data' (bytes o str base64)
                - 'mime_type' (str)
        
        Returns:
            Análisis físico como string
        """
        if not images:
            return "No se proporcionaron imágenes para análisis."
        
        content = [{
            "type": "text",
            "text": f"""Analiza estas {len(images)} imágenes relacionadas con física y proporciona:

1. FENÓMENO FÍSICO: ¿Qué fenómeno físico se observa?
2. CONSERVACIÓN DE CANTIDAD DE MOVIMIENTO: ¿Se conserva? (SÍ/NO/PARCIAL)
3. PRINCIPIOS FÍSICOS: ¿Qué leyes o principios aplican?
4. ECUACIONES RELEVANTES: Fórmulas aplicables
5. TIPO DE SISTEMA: Clasificación (péndulo, colisión, fluidos, etc.)
6. DESCRIPCIÓN DETALLADA: Análisis completo del fenómeno

Sé específico y técnico. Usa terminología física precisa."""
        }]
        
        # Agregar imágenes
        for idx, img in enumerate(images):
            try:
                image_data_raw = img.get('data') or img.get('bytes')
                
                if isinstance(image_data_raw, bytes):
                    image_data_b64 = self.encode_image(image_data_raw)
                elif isinstance(image_data_raw, str):
                    image_data_b64 = image_data_raw
                else:
                    print(f"⚠️ Tipo de dato no soportado para imagen {idx}: {type(image_data_raw)}")
                    continue
                
                mime_type = self.get_mime_type(img.get('mime_type', 'image/png'))
                
                content.append({
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_data_b64}"
                })
                
                print(f"✅ Imagen {idx} agregada para análisis: {mime_type}")
                
            except Exception as e:
                print(f"❌ Error procesando imagen {idx}: {e}")
                continue
        
        message = HumanMessage(content=content)
        
        try:
            response = self.llm.invoke([message])
            print(f"✅ Análisis de imágenes de física completado")
            return response.content
        except Exception as e:
            error_msg = f"Error en análisis de imágenes: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    async def classify_query(self, query: str, context: str, visual_findings: str) -> str:
        """Clasifica la consulta de física."""
        system_prompt = f"""Eres un profesor experto en Física I que clasifica consultas de estudiantes.

TEMARIO DISPONIBLE:
{self.temario}

Tu tarea es identificar:
1. El tema del temario al que pertenece la consulta
2. Los subtemas relevantes
3. Palabras clave para búsqueda
4. Tipo de contenido necesario (texto/imagen/ambos)

Formato de respuesta:
TEMA: [número y título del temario]
SUBTEMAS: [lista de subtemas]
KEYWORDS: [palabras clave físicas]
TIPO_CONTENIDO: [texto/imagen/ambos]
"""
        
        user_prompt = f"""
HALLAZGOS VISUALES (si hay imágenes):
{visual_findings}

CONTEXTO DE CONVERSACIÓN PREVIA:
{context}

CONSULTA DEL ESTUDIANTE:
{query}

Clasifica esta consulta según el temario de física."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error en clasificación: {str(e)}"
    
    async def generate_search_query(self, classification: str, visual_findings: str, 
                                   original_query: str) -> str:
        """Genera consulta de búsqueda optimizada."""
        system_prompt = """Eres un experto en búsqueda de información física.
Genera una consulta de búsqueda precisa y técnica.

Reglas:
- Usa terminología física precisa
- Incluye conceptos clave del temario
- Prioriza información relevante para física universitaria

Responde SOLO con la consulta optimizada, sin explicaciones."""
        
        user_prompt = f"""
CLASIFICACIÓN:
{classification}

HALLAZGOS VISUALES:
{visual_findings}

CONSULTA ORIGINAL:
{original_query}

Genera la mejor consulta de búsqueda para encontrar información relevante."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generando consulta: {str(e)}"
    
    async def generate_physics_response(
        self, 
        query: str, 
        context: str, 
        classification: str, 
        visual_findings: str,
        document_context: str,
        image_context: str
    ) -> str:
        """Genera la respuesta final de física."""
        system_prompt = f"""Eres un profesor experto en Física I de la Universidad de Buenos Aires.

TEMARIO DEL CURSO:
{self.temario}

**Estructura tu respuesta:**
1. CONTEXTO DEL TEMA: Ubicación en el temario
2. EXPLICACIÓN TEÓRICA: Conceptos fundamentales
3. ANÁLISIS DE IMÁGENES (si hay): Relacionar teoría con imágenes
4. ECUACIONES Y FÓRMULAS: Desarrollo matemático
5. EJEMPLOS Y APLICACIONES: Casos prácticos
6. RESUMEN: Puntos clave

**Reglas:**
- Usa lenguaje técnico pero claro
- Relaciona conceptos con el temario
- Si hay imágenes, descríbelas y conéctalas con la teoría
- Incluye ecuaciones cuando sea relevante
- Proporciona ejemplos concretos
"""
        
        user_prompt = f"""
**CONSULTA DEL ESTUDIANTE:**
{query}

**CONTEXTO DE CONVERSACIÓN:**
{context}

**CLASIFICACIÓN:**
{classification}

**HALLAZGOS VISUALES:**
{visual_findings}

**INFORMACIÓN DE DOCUMENTOS:**
{document_context}

**INFORMACIÓN DE IMÁGENES RELACIONADAS:**
{image_context}

Proporciona una explicación completa y didáctica."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
    
    async def invoke(self, query: str, context_id: str, 
                    images: List[dict] = None) -> str:
        """
        Procesa una consulta de física y retorna la respuesta completa.
        
        Args:
            query: Consulta del usuario
            context_id: ID del contexto/sesión
            images: Lista de imágenes (opcional)
        
        Returns:
            Respuesta de física completa como string
        """
        print(f"\n{'='*80}")
        print(f"📚 Procesando consulta de física")
        print(f"Context ID: {context_id}")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes recibidas: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        try:
            memory_context = self._get_memory_context(context_id)
            
            # PASO 1: Analizar imágenes si existen
            visual_findings = ""
            image_embedding = None
            
            if images and len(images) > 0:
                print(f"🖼️ Analizando {len(images)} imagen(es)...")
                visual_findings = await self.analyze_physics_image(images)
                self.visual_findings[context_id] = visual_findings
                
                # Generar embedding de la primera imagen para búsqueda
                first_image_data = images[0].get('data') or images[0].get('bytes')
                if isinstance(first_image_data, str):
                    first_image_data = base64.b64decode(first_image_data)
                image_embedding = self.generate_image_embedding(first_image_data)
                
                print(f"✅ Análisis visual completado")
            else:
                visual_findings = self.visual_findings.get(
                    context_id, 
                    "No se proporcionaron imágenes para análisis."
                )
            
            # PASO 2: Clasificar consulta
            print(f"🔍 Clasificando consulta...")
            classification = await self.classify_query(query, memory_context, visual_findings)
            print(f"✅ Clasificación completada")
            
            # PASO 3: Generar consulta de búsqueda
            print(f"🔎 Generando consulta de búsqueda...")
            search_query = await self.generate_search_query(
                classification, visual_findings, query
            )
            
            # PASO 4: Buscar en Qdrant
            print(f"🗄️ Buscando en documentos...")
            search_results = await self.search_multimodal(
                query=search_query,
                image_embedding=image_embedding,
                top_k=5
            )
            print(f"✅ Búsqueda completada")
            
            # PASO 5: Formatear contexto de documentos
            document_context = "\n".join([
                f"--- Fragmento {i+1} (Score: {r['score']}) ---\n{r['payload'].get('text', 'N/A')}"
                for i, r in enumerate(search_results.get('text', []))
            ])
            
            image_context = "\n".join([
                f"--- Imagen {i+1} (Score: {r['score']}) ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
                for i, r in enumerate(search_results.get('image', []))
            ])
            
            # PASO 6: Generar respuesta final
            print(f"📝 Generando respuesta final...")
            final_response = await self.generate_physics_response(
                query, memory_context, classification, visual_findings,
                document_context, image_context
            )
            
            print(f"✅ Respuesta generada: {len(final_response)} caracteres")
            
            # Guardar en memoria
            self._save_to_memory(context_id, query, final_response)
            
            print(f"✅ Consulta de física completada\n")
            
            return final_response
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    async def stream(self, query: str, context_id: str, 
                    images: List[dict] = None) -> AsyncIterable[PhysicsResponseFormat]:
        """
        Procesa una consulta de física con streaming.
        
        Yields:
            PhysicsResponseFormat con estado y contenido
        """
        print(f"\n{'='*80}")
        print(f"📚 Procesando consulta de física (streaming)")
        print(f"Context ID: {context_id}")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes recibidas: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        memory_context = self._get_memory_context(context_id)
        
        # PASO 1: Analizar imágenes
        visual_findings = ""
        image_embedding = None
        
        if images and len(images) > 0:
            yield PhysicsResponseFormat(
                status='analyzing_images',
                message=f'🖼️ Analizando {len(images)} imagen(es) de física...',
                section='visual_analysis'
            )
            
            visual_findings = await self.analyze_physics_image(images)
            self.visual_findings[context_id] = visual_findings
            
            # Generar embedding
            first_image_data = images[0].get('data') or images[0].get('bytes')
            if isinstance(first_image_data, str):
                first_image_data = base64.b64decode(first_image_data)
            image_embedding = self.generate_image_embedding(first_image_data)
            
            yield PhysicsResponseFormat(
                status='analyzing_images',
                message='✅ Fenómenos físicos identificados en las imágenes.',
                section='visual_analysis'
            )
        else:
            visual_findings = self.visual_findings.get(
                context_id, 
                "No se proporcionaron imágenes para análisis."
            )
        
        # PASO 2: Clasificar
        yield PhysicsResponseFormat(
            status='classifying',
            message='📚 Clasificando consulta según el temario...',
            section='classification'
        )
        
        classification = await self.classify_query(query, memory_context, visual_findings)
        
        # PASO 3: Buscar
        yield PhysicsResponseFormat(
            status='searching_documents',
            message='🔎 Buscando en documentos de física...',
            section='document_search'
        )
        
        search_query = await self.generate_search_query(
            classification, visual_findings, query
        )
        
        search_results = await self.search_multimodal(
            query=search_query,
            image_embedding=image_embedding,
            top_k=5
        )
        
        # PASO 4: Formatear contexto
        document_context = "\n".join([
            f"--- Fragmento {i+1} (Score: {r['score']}) ---\n{r['payload'].get('text', 'N/A')}"
            for i, r in enumerate(search_results.get('text', []))
        ])
        
        image_context = "\n".join([
            f"--- Imagen {i+1} (Score: {r['score']}) ---\nPDF: {r['payload'].get('pdf_name', 'N/A')}"
            for i, r in enumerate(search_results.get('image', []))
        ])
        
        # PASO 5: Generar respuesta
        yield PhysicsResponseFormat(
            status='generating_response',
            message='📝 Generando explicación didáctica...',
            section='final_response'
        )
        
        final_response = await self.generate_physics_response(
            query, memory_context, classification, visual_findings,
            document_context, image_context
        )
        
        # Guardar en memoria
        self._save_to_memory(context_id, query, final_response)
        
        # PASO 6: Respuesta final
        yield PhysicsResponseFormat(
            status='completed',
            message=final_response,
            section='final_response'
        )


# ==================== INTEGRACIÓN CON A2A ====================

# Instancia global del agente
agent_instance = None

async def initialize_agent():
    """Inicializar el agente y procesar PDFs automáticamente."""
    global agent_instance
    
    print("\n" + "="*80)
    print("🔧 INICIALIZANDO AGENTE DE FÍSICA MULTIMODAL")
    print("="*80)
    
    agent_instance = PhysicsMultimodalAgent()
    
    # Limpiar colecciones antiguas si existen
    await cleanup_collections()
    
    # Procesar PDFs de la carpeta fija
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    
    if pdf_files:
        print(f"\n📂 Encontrados {len(pdf_files)} PDFs en {PDF_FOLDER}")
        for pdf in pdf_files:
            print(f"   📄 {os.path.basename(pdf)}")
        
        # Procesar y almacenar
        temario = await agent_instance.procesar_y_almacenar_pdfs(pdf_files)
        
        print("\n✅ Agente inicializado y listo para consultas")
        print(f"📋 Temario: {len(temario)} caracteres")
    else:
        print(f"\n⚠️ No se encontraron PDFs en {PDF_FOLDER}")
        print("El agente se inicializará con conocimiento base.")
    
    return agent_instance

async def cleanup_collections():
    """Eliminar colecciones antiguas."""
    client = AsyncQdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", "")
    )
    
    collections = ["documentos_pdf_texto", "documentos_pdf_imagenes", "documentos_multimodal"]
    
    for collection in collections:
        try:
            await client.delete_collection(collection)
            print(f"🗑️ Colección '{collection}' eliminada")
        except Exception:
            print(f"ℹ️ Colección '{collection}' no existía")


# ==================== HANDLERS A2A ====================

async def handle_a2a_message(message: dict, context_id: str = "default"):
    """Handler para mensajes A2A."""
    global agent_instance
    
    if not agent_instance:
        await initialize_agent()
    
    query = message.get("text", "")
    images = message.get("images", [])
    
    # Convertir imágenes al formato esperado
    processed_images = []
    for img in images:
        if isinstance(img, dict):
            processed_images.append({
                "data": img.get("data") or img.get("bytes"),
                "mime_type": img.get("mime_type", "image/png")
            })
    
    # Usar streaming para obtener la respuesta completa
    response_parts = []
    async for response_format in agent_instance.stream(query, context_id, processed_images):
        if response_format.status == 'completed':
            response_parts.append(response_format.message)
    
    return {
        "response": "".join(response_parts),
        "status": "completed",
        "context_id": context_id
    }

async def handle_a2a_stream(message: dict, context_id: str = "default"):
    """Handler para streaming A2A."""
    global agent_instance
    
    if not agent_instance:
        await initialize_agent()
    
    query = message.get("text", "")
    images = message.get("images", [])
    
    # Convertir imágenes al formato esperado
    processed_images = []
    for img in images:
        if isinstance(img, dict):
            processed_images.append({
                "data": img.get("data") or img.get("bytes"),
                "mime_type": img.get("mime_type", "image/png")
            })
    
    # Stream de respuestas
    async for response_format in agent_instance.stream(query, context_id, processed_images):
        yield {
            "status": response_format.status,
            "message": response_format.message,
            "section": response_format.section
        }


# ==================== MAIN PARA EJECUCIÓN DIRECTA ====================

if __name__ == "__main__":
    # Para ejecutar directamente (testing)
    async def main():
        agent = await initialize_agent()
        
        # Ejemplo de consulta
        resultado = await agent.invoke(
            query="¿Qué es la conservación del momento lineal?",
            context_id="test_user"
        )
        print("\nResultado:", resultado[:200] + "...")
    
    asyncio.run(main())
