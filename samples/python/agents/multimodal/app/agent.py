# samples/python/agents/multimodal/app/agent.py (CORREGIDO)

import asyncio
import base64
import glob
import io
import os
import re
import json
from collections.abc import AsyncIterable
from io import BytesIO
from pathlib import Path
from typing import Any, List, Literal, Optional

# Importar matplotlib para renderizar LaTeX
import matplotlib
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
from matplotlib import mathtext

# ==================== CONFIGURACIÓN ====================

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ==================== UTILIDADES PARA RENDERIZAR LATEX ====================

def latex_to_image_base64(latex_formula: str, fontsize: int = 16) -> str:
    """
    Convierte una fórmula LaTeX a imagen PNG en base64.
    
    Args:
        latex_formula: Fórmula LaTeX (sin delimitadores $ o $$)
        fontsize: Tamaño de fuente
    
    Returns:
        String base64 de la imagen PNG
    """
    try:
        # Crear figura con fondo transparente
        fig = plt.figure(figsize=(0.01, 0.01), dpi=150)
        fig.patch.set_alpha(0)
        
        # Renderizar LaTeX
        text = fig.text(
            0, 0, 
            f'${latex_formula}$',
            fontsize=fontsize,
            color='black'
        )
        
        # Ajustar tamaño de figura al texto
        fig.canvas.draw()
        bbox = text.get_window_extent(fig.canvas.get_renderer())
        width, height = bbox.width / fig.dpi, bbox.height / fig.dpi
        fig.set_size_inches(width * 1.2, height * 1.2)
        
        # Reposicionar texto centrado
        text.set_position((0.5, 0.5))
        text.set_horizontalalignment('center')
        text.set_verticalalignment('center')
        
        # Guardar en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0.1, transparent=True, dpi=150)
        plt.close(fig)
        
        # Convertir a base64
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        return img_base64
    except Exception as e:
        print(f"❌ Error renderizando LaTeX '{latex_formula}': {e}")
        return None


def convert_latex_to_images_in_text(text: str) -> str:
    """
    Convierte todas las fórmulas LaTeX en un texto a imágenes embebidas en markdown.
    
    Busca patrones:
    - $formula$ (inline)
    - $$formula$$ (display)
    
    Y los reemplaza con:
    ![formula](data:image/png;base64,...)
    
    Args:
        text: Texto con fórmulas LaTeX
    
    Returns:
        Texto con fórmulas convertidas a imágenes
    """
    # Patrón para display math ($$...$$)
    display_pattern = r'\$\$(.*?)\$\$'
    # Patrón para inline math ($...$)
    inline_pattern = r'\$([^\$]+?)\$'
    
    def replace_display(match):
        latex = match.group(1).strip()
        img_base64 = latex_to_image_base64(latex, fontsize=18)
        if img_base64:
            return f'<div style="text-align:center;margin:8px 0;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;vertical-align:middle;" /></div>'
        return match.group(0)  # Si falla, mantener original
    
    def replace_inline(match):
        latex = match.group(1).strip()
        img_base64 = latex_to_image_base64(latex, fontsize=14)
        if img_base64:
            return f'<img src="data:image/png;base64,{img_base64}" style="vertical-align:middle;height:1.2em;" />'
        return match.group(0)  # Si falla, mantener original
    
    # Primero reemplazar display ($$...$$) para evitar conflictos
    text = re.sub(display_pattern, replace_display, text, flags=re.DOTALL)
    # Luego reemplazar inline ($...$)
    text = re.sub(inline_pattern, replace_inline, text)
    
    return text

class SemanticMemory:
    """Memoria conversacional simplificada sin dependencias deprecadas."""
    
    def __init__(self, llm, max_entries: int = 10):
        self.conversations = []
        self.max_entries = max_entries
        self.summary = ""
        self.direct_history = ""
        self.llm = llm
        # Modo socrático
        self.socratic_mode = False
        self.socratic_questions_asked = 0
        self.socratic_answers = []
        self.original_query = ""
    
    def to_dict(self) -> dict:
        """Serializa la memoria a un diccionario."""
        return {
            "conversations": self.conversations,
            "summary": self.summary,
            "direct_history": self.direct_history,
            "socratic_mode": self.socratic_mode,
            "socratic_questions_asked": self.socratic_questions_asked,
            "socratic_answers": self.socratic_answers,
            "original_query": self.original_query
        }

    @classmethod
    def from_dict(cls, data: dict, llm) -> 'SemanticMemory':
        """Crea una instancia de SemanticMemory desde un diccionario."""
        mem = cls(llm=llm)
        mem.conversations = data.get("conversations", [])
        mem.summary = data.get("summary", "")
        mem.direct_history = data.get("direct_history", "")
        mem.socratic_mode = data.get("socratic_mode", False)
        mem.socratic_questions_asked = data.get("socratic_questions_asked", 0)
        mem.socratic_answers = data.get("socratic_answers", [])
        mem.original_query = data.get("original_query", "")
        return mem

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
        self.socratic_mode = False
        self.socratic_questions_asked = 0
        self.socratic_answers = []
        self.original_query = ""


class PhysicsMultimodalAgent:
    """Agente de física con procesamiento multimodal."""
    
    SYSTEM_INSTRUCTION = (
        'Eres un tutor socrático multimodal de Física I de la UBA. '
        'Usas el método socrático: ante cada consulta (texto o imagen), '
        'primero haces 3 preguntas guía para activar el pensamiento crítico '
        'del estudiante y luego proporcionas la respuesta completa. '
        'Analizas consultas de texto e imágenes de experimentos, diagramas '
        'y problemas de física.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        """Inicializar el agente de física."""
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=4096,
            api_key=os.getenv('GROQ_API_KEY')
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
- **IMPORTANTE: Todas las fórmulas y ecuaciones DEBEN estar en formato LaTeX**
  - Usa `$formula$` para fórmulas inline (en línea con el texto)
  - Usa `$$formula$$` para fórmulas display (en bloque separado)
  - Ejemplos:
    * Inline: La energía cinética es $E_k = \\frac{{1}}{{2}}mv^2$
    * Display: $$F = ma$$
    * Display complejo: $$\\vec{{F}} = m\\vec{{a}}$$
- NUNCA uses texto plano para fórmulas (NO escribas "F = m*a" o "E = 1/2*m*v^2")
- Usa notación matemática correcta: vectores con \\vec{{}}, fracciones con \\frac{{}}{{}}, subíndices con _, superíndices con ^
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

Explicación completa con todas las fórmulas en formato LaTeX."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
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

        system_prompt = f"""Eres un tutor socrático de Física I de la UBA.

Tu objetivo es guiar al estudiante a descubrir la respuesta por sí mismo mediante preguntas.
Recibís tanto texto como imágenes de experimentos, diagramas y problemas de física.

TEMARIO:
{self.temario}
{visual_section}

Reglas para las preguntas:
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
        system_prompt = f"""Profesor de Física I UBA que usa el método socrático.

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
    
    # ==================== MÉTODOS PRINCIPALES ====================
    
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
            
            # Verificar si estamos en modo socrático
            if memory.socratic_mode:
                # Guardar respuesta del estudiante
                memory.socratic_answers.append(query)
                memory.socratic_questions_asked += 1
                
                print(f"🎓 Modo socrático: {memory.socratic_questions_asked}/3 preguntas respondidas")
                
                # Si ya respondió las 3 preguntas, dar la respuesta completa
                if memory.socratic_questions_asked >= 3:
                    print(f"✅ Completadas las 3 preguntas, generando respuesta final...")
                    
                    # Analizar imágenes si las hay
                    visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
                    image_embedding = None
                    
                    # Clasificar
                    classification = await self.classify_query(
                        memory.original_query, memory_context, visual_findings
                    )
                    
                    # Buscar
                    search_query = await self.generate_search_query(
                        classification, visual_findings, memory.original_query
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
                    
                    # Generar respuesta final incluyendo las respuestas del estudiante
                    student_answers_summary = "\n".join([
                        f"Pregunta {i+1}: {ans}"
                        for i, ans in enumerate(memory.socratic_answers)
                    ])
                    
                    final_response = await self.generate_physics_response_with_socratic(
                        memory.original_query, memory_context, classification, 
                        visual_findings, document_context, image_context,
                        student_answers_summary
                    )
                    
                    # Convertir fórmulas LaTeX a imágenes PNG embebidas (HTML <img>)
                    final_response = convert_latex_to_images_in_text(final_response)
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
                    
                    # Convertir fórmulas LaTeX a imágenes PNG embebidas (HTML <img>)
                    final_response = convert_latex_to_images_in_text(final_response)
                    return next_question
            
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
            
            # Convertir fórmulas LaTeX a imágenes PNG embebidas (HTML <img>)
            first_question = convert_latex_to_images_in_text(first_question)
            return first_question
            
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
        
        Implementa modo socrático con 3 preguntas antes de la respuesta.
        """
        print(f"\n{'='*80}")
        print(f"📚 Consulta (streaming)")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        memory = self._get_or_create_memory(context_id)
        memory_context = self._get_memory_context(context_id)
        
        # Verificar si estamos en modo socrático
        if memory.socratic_mode:
            # Guardar respuesta del estudiante
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
                
                # Analizar imágenes si las hay
                visual_findings = self.visual_findings.get(context_id, "No hay imágenes.")
                image_embedding = None
                
                # Clasificar
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': '📚 Analizando tu proceso de pensamiento...',
                    'status': 'classifying'
                }
                
                classification = await self.classify_query(
                    memory.original_query, memory_context, visual_findings
                )
                
                # Buscar
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
                    'content': '📝 Generando explicación completa basada en tus respuestas...',
                    'status': 'generating_response'
                }
                
                # Generar respuesta final incluyendo las respuestas del estudiante
                student_answers_summary = "\n".join([
                    f"Pregunta {i+1}: {ans}"
                    for i, ans in enumerate(memory.socratic_answers)
                ])
                
                final_response = await self.generate_physics_response_with_socratic(
                    memory.original_query, memory_context, classification,
                    visual_findings, document_context, image_context,
                    student_answers_summary
                )
                
                # Convertir fórmulas LaTeX a imágenes PNG embebidas (HTML <img>)
                final_response = convert_latex_to_images_in_text(final_response)
                # Resetear modo socrático
                memory.socratic_mode = False
                memory.socratic_questions_asked = 0
                memory.socratic_answers = []
                memory.original_query = ""
                
                self._save_to_memory(context_id, query, final_response)
                
                # Yield final response
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
                
                # Convertir fórmulas LaTeX a imágenes PNG embebidas (HTML <img>)
                next_question = convert_latex_to_images_in_text(next_question)
                # CRÍTICO: is_task_complete=False + require_user_input=True
                # para que el executor marque como input_required y mantenga la memoria
                yield {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': next_question,
                    'status': 'socratic_question'
                }
        else:
            # Modo normal: iniciar modo socrático
            print(f"🎓 Iniciando modo socrático...")
            
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
            
            # Activar modo socrático
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': '🎓 Iniciando método socrático: te haré 3 preguntas para guiar tu aprendizaje...',
                'status': 'socratic_init'
            }
            
            memory.socratic_mode = True
            memory.original_query = query
            memory.socratic_questions_asked = 0
            memory.socratic_answers = []
            
            # Generar primera pregunta socrática
            first_question = await self.generate_socratic_question(
                query, 0, [],
                visual_findings=visual_findings
            )
            
            # Convertir fórmulas LaTeX a imágenes PNG embebidas (HTML <img>)
            first_question = convert_latex_to_images_in_text(first_question)
            # CRÍTICO: is_task_complete=False + require_user_input=True
            # para que el executor marque como input_required y mantenga la memoria
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': first_question,
                'status': 'socratic_question'
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
