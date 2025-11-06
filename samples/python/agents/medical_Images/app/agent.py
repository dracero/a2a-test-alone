import base64
import os
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any, Literal

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


class MedicalResponseFormat(BaseModel):
    """Formato de respuesta médica estructurada."""
    
    status: Literal['analyzing_images', 'classifying', 'searching', 'generating_response', 'input_required', 'completed', 'error'] = 'analyzing_images'
    message: str
    section: Literal['visual_findings', 'classification', 'search_results', 'final_response', 'general'] = 'general'


class MedicalAgent:
    """Agente médico con análisis de imágenes, búsqueda y memoria conversacional."""
    
    SYSTEM_INSTRUCTION = (
        'Eres un médico especialista experimentado que analiza consultas médicas, '
        'imágenes médicas y proporciona análisis profesionales. '
        'Debes ser claro, objetivo y siempre incluir disclaimers apropiados. '
        'NUNCA proporciones diagnósticos definitivos. '
        'Siempre menciona que tu análisis no sustituye una consulta médica presencial.'
    )
    
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/jpeg', 'image/png', 'image/webp']
    
    def __init__(self):
        """Inicializar el agente médico."""
        # Modelo principal
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=4096,
        )
        
        # Herramienta de búsqueda
        self.tavily_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        # Memoria conversacional por contexto
        self.memories = {}
        
        # Almacenamiento temporal de hallazgos visuales por contexto
        self.visual_findings = {}
    
    def _get_or_create_memory(self, context_id: str):
        """Obtener o crear memoria para un contexto específico."""
        if context_id not in self.memories:
            self.memories[context_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=2000,
                return_messages=True
            )
        return self.memories[context_id]
    
    def _get_memory_context(self, context_id: str) -> str:
        """Obtener el contexto de memoria para un contexto específico."""
        memory = self._get_or_create_memory(context_id)
        try:
            memory_variables = memory.load_memory_variables({})
            history_messages = memory_variables.get("history", [])
            if history_messages:
                return "\n".join([
                    f"{type(msg).__name__}: {msg.content}" 
                    for msg in history_messages
                ])
            return "Primera consulta del paciente."
        except Exception:
            return "Primera consulta del paciente."
    
    def _save_to_memory(self, context_id: str, query: str, response: str):
        """Guardar interacción en memoria."""
        memory = self._get_or_create_memory(context_id)
        memory.save_context({"input": query}, {"output": response})
    
    def encode_image(self, image_data: bytes) -> str:
        """Codifica imagen en base64."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def decode_base64_image(self, base64_string: str) -> bytes:
        """Decodifica una imagen desde base64."""
        return base64.b64decode(base64_string)
    
    def get_mime_type(self, content_type: str) -> str:
        """Mapea content_type a MIME type para Gemini."""
        mapping = {
            'image/jpeg': 'image/jpeg',
            'image/png': 'image/png',
            'image/webp': 'image/webp',
            'image/gif': 'image/gif',
        }
        return mapping.get(content_type, 'image/png')
    
    async def analyze_images(self, images: list[dict]) -> str:
        """
        Analiza imágenes médicas con Gemini Vision.
        
        Args:
            images: Lista de diccionarios con:
                - 'data' (bytes o str base64)
                - 'mime_type' (str)
        
        Returns:
            Hallazgos visuales como string
        """
        if not images:
            return "No se proporcionaron imágenes para análisis."
        
        # Preparar contenido del mensaje
        content = [{
            "type": "text",
            "text": f"""Analiza estas {len(images)} imágenes médicas y proporciona una lista de HALLAZGOS CLAVES.

Formato de salida:
HALLAZGO 1: [Descripción específica del hallazgo]
HALLAZGO 2: [Descripción específica del hallazgo]
...

Enfócate en:
- Anomalías visibles
- Características anatómicas relevantes
- Patrones de interés clínico
- Comparaciones con normalidad esperada

Sé específico y objetivo. Evita diagnósticos definitivos."""
        }]
        
        # Agregar imágenes
        for idx, img in enumerate(images):
            try:
                # Manejar tanto bytes como base64 string
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
                
                print(f"✅ Imagen {idx} agregada: {mime_type}")
                
            except Exception as e:
                print(f"❌ Error procesando imagen {idx}: {e}")
                continue
        
        message = HumanMessage(content=content)
        
        try:
            response = self.llm.invoke([message])
            print(f"✅ Análisis de imágenes completado")
            return response.content
        except Exception as e:
            error_msg = f"Error en análisis de imágenes: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    async def classify_query(self, query: str, context: str, visual_findings: str) -> str:
        """Clasifica la consulta médica."""
        system_prompt = """Eres un médico especialista que analiza consultas médicas.
Tu tarea es identificar:
1. El tipo de consulta (diagnóstico, seguimiento, segunda opinión, etc.)
2. Los hallazgos visuales mencionados o relevantes
3. Las palabras clave médicas importantes

Formato de respuesta:
TIPO_CONSULTA: [tipo]
HALLAZGOS_RELEVANTES: [lista]
KEYWORDS_MEDICAS: [palabras clave]
ESPECIALIDAD_SUGERIDA: [especialidad médica relevante]"""
        
        user_prompt = f"""
HALLAZGOS VISUALES IDENTIFICADOS:
{visual_findings}

CONTEXTO DE CONVERSACIÓN PREVIA:
{context}

CONSULTA DEL PACIENTE/USUARIO:
{query}

Clasifica esta consulta médica según los hallazgos y el contexto proporcionado."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error en clasificación: {str(e)}"
    
    async def generate_search_queries(self, classification: str, visual_findings: str, 
                                     original_query: str) -> str:
        """Genera consultas de búsqueda optimizadas."""
        system_prompt = """Eres un experto en búsqueda de información médica.
Genera consultas de búsqueda médicas precisas y profesionales.

Reglas:
- Usa terminología médica precisa
- Enfócate en información clínica relevante
- Prioriza fuentes médicas confiables

Responde SOLO con la consulta de búsqueda optimizada, sin explicaciones."""
        
        user_prompt = f"""
CLASIFICACIÓN MÉDICA:
{classification}

HALLAZGOS VISUALES:
{visual_findings}

CONSULTA ORIGINAL:
{original_query}

Genera la mejor consulta de búsqueda médica para esta información."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generando consultas: {str(e)}"
    
    async def search_medical_info(self, search_query: str) -> str:
        """Busca información médica en Tavily."""
        try:
            result = self.tavily_tool.invoke({"query": search_query})
            
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get('answer', result[0].get('content', ''))
                return f"Búsqueda: {search_query}\n{content}"
            else:
                return f"Búsqueda: {search_query}\nNo se encontró información específica."
        except Exception as e:
            return f"Error en búsqueda: {str(e)}"
    
    async def generate_medical_response(self, query: str, context: str, 
                                       classification: str, visual_findings: str,
                                       search_info: str) -> str:
        """Genera la respuesta médica final."""
        system_prompt = """Eres un médico especialista experimentado que proporciona análisis médicos claros y profesionales.

**Estructura tu respuesta en estas secciones:**
1. DESCRIPCIÓN GENERAL: Resumen breve de la consulta
2. HALLAZGOS PRINCIPALES: Basado en los hallazgos visuales identificados
3. INTERPRETACIÓN CLÍNICA: Integra la información de búsqueda médica
4. CONSIDERACIONES DIAGNÓSTICAS: Posibles diagnósticos diferenciales
5. RECOMENDACIONES: Pasos siguientes sugeridos

**Reglas importantes:**
- Usa lenguaje médico profesional pero accesible
- Siempre incluye disclaimers apropiados (no sustituye consulta médica presencial)
- Basa tus conclusiones en evidencia
- Sé claro sobre las limitaciones del análisis remoto
- Nunca proporciones diagnósticos definitivos"""
        
        user_prompt = f"""
**CONSULTA ORIGINAL DEL USUARIO:**
{query}

**CONTEXTO DE CONVERSACIÓN ANTERIOR:**
{context}

**CLASIFICACIÓN MÉDICA:**
{classification}

**HALLAZGOS VISUALES DE LAS IMÁGENES:**
{visual_findings}

**INFORMACIÓN MÉDICA DE BÚSQUEDA:**
{search_info}

Proporciona un análisis médico completo y profesional."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
    
    async def stream(self, query: str, context_id: str, 
                    images: list[dict] = None) -> AsyncIterable[dict[str, Any]]:
        """
        Procesa una consulta médica con streaming.
        
        IMPORTANTE: Este método DEBE yieldar diccionarios con estas claves:
        - 'is_task_complete': bool
        - 'require_user_input': bool
        - 'content': str
        - 'status': str (opcional)
        """
        print(f"\n{'='*80}")
        print(f"🏥 Procesando consulta médica")
        print(f"Context ID: {context_id}")
        print(f"Query: {query[:100]}...")
        print(f"Imágenes recibidas: {len(images) if images else 0}")
        print(f"{'='*80}\n")
        
        # Obtener contexto de memoria
        memory_context = self._get_memory_context(context_id)
        
        # PASO 1: Analizar imágenes si existen
        visual_findings = ""
        if images and len(images) > 0:
            print(f"📸 Analizando {len(images)} imagen(es)...")
            
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f'🔍 Analizando {len(images)} imagen(es) médica(s)...',
                'status': 'analyzing_images'
            }
            
            visual_findings = await self.analyze_images(images)
            self.visual_findings[context_id] = visual_findings
            
            print(f"✅ Análisis visual completado")
            
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f'✅ Hallazgos visuales identificados.',
                'status': 'analyzing_images'
            }
        else:
            visual_findings = self.visual_findings.get(
                context_id, 
                "No se proporcionaron imágenes para análisis."
            )
        
        # PASO 2: Clasificar consulta
        print(f"🔍 Clasificando consulta...")
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🏥 Clasificando consulta médica...',
            'status': 'classifying'
        }
        
        classification = await self.classify_query(query, memory_context, visual_findings)
        print(f"✅ Clasificación completada")
        
        # PASO 3: Generar consultas de búsqueda
        print(f"🔎 Generando consultas de búsqueda...")
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '🔎 Buscando información médica relevante...',
            'status': 'searching'
        }
        
        search_query = await self.generate_search_queries(
            classification, visual_findings, query
        )
        
        # PASO 4: Buscar información
        print(f"🌐 Buscando información médica...")
        search_info = await self.search_medical_info(search_query)
        print(f"✅ Búsqueda completada")
        
        # PASO 5: Generar respuesta final
        print(f"📝 Generando respuesta médica final...")
        yield {
            'is_task_complete': False,
            'require_user_input': False,
            'content': '📝 Generando análisis médico completo...',
            'status': 'generating_response'
        }
        
        final_response = await self.generate_medical_response(
            query, memory_context, classification, visual_findings, search_info
        )
        
        print(f"✅ Respuesta generada: {len(final_response)} caracteres")
        print(f"📄 Respuesta preview: {final_response[:200]}...")
        
        # Guardar en memoria
        self._save_to_memory(context_id, query, final_response)
        
        print(f"✅ Consulta médica completada\n")
        
        # CRÍTICO: Retornar respuesta final con is_task_complete=True
        yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': final_response,
            'status': 'completed'
        }
