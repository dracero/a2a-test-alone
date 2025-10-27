import base64
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (InternalError, InvalidParamsError, Part, TaskState,
                       TextPart, UnsupportedOperationError)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from app.agent import MedicalAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalAgentExecutor(AgentExecutor):
    """Executor para el Asistente Médico."""
    
    def __init__(self):
        self.agent = MedicalAgent()
    
    def _extract_images_from_message(self, context: RequestContext) -> list[dict]:
        """
        Extrae imágenes del mensaje del usuario.
        Soporta tanto ImagePart (kind='image') como FilePart (kind='file').
        
        Returns:
            Lista de diccionarios con 'data' (str base64), 'mime_type' (str)
        """
        images = []
        
        if not context.message or not context.message.parts:
            logger.info("No hay partes en el mensaje")
            return images
        
        logger.info(f"Procesando {len(context.message.parts)} partes del mensaje")
        
        for idx, part in enumerate(context.message.parts):
            part_root = part.root
            part_kind = getattr(part_root, 'kind', None)
            
            logger.info(f"Parte {idx}: kind='{part_kind}', tipo={type(part_root).__name__}")
            
            # OPCIÓN 1: ImagePart (kind='image')
            if part_kind == 'image':
                try:
                    if hasattr(part_root, 'data') and hasattr(part_root, 'mime_type'):
                        image_data = part_root.data
                        mime_type = part_root.mime_type
                        
                        # Si data es bytes, convertir a base64
                        if isinstance(image_data, bytes):
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        
                        images.append({
                            'data': image_data,  # String base64
                            'mime_type': mime_type
                        })
                        logger.info(f"✅ ImagePart extraída: {mime_type}")
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo ImagePart: {e}")
            
            # OPCIÓN 2: FilePart (kind='file') - Lo que envía el cliente A2A
            elif part_kind == 'file':
                try:
                    # FilePart tiene un atributo 'file' que puede ser FileWithBytes o FileWithUri
                    if hasattr(part_root, 'file'):
                        file_obj = part_root.file
                        
                        logger.info(f"FilePart detectada, tipo de file: {type(file_obj).__name__}")
                        
                        # FileWithBytes tiene 'bytes' y 'mime_type'
                        if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
                            image_data = file_obj.bytes  # Ya debería ser string base64
                            mime_type = file_obj.mime_type
                            
                            # Verificar si es bytes y convertir si es necesario
                            if isinstance(image_data, bytes):
                                image_data = base64.b64encode(image_data).decode('utf-8')
                            
                            images.append({
                                'data': image_data,  # String base64
                                'mime_type': mime_type
                            })
                            logger.info(f"✅ FilePart (FileWithBytes) extraída: {mime_type}, tamaño base64: {len(image_data)}")
                        
                        # FileWithUri tiene 'uri' y 'mime_type'
                        elif hasattr(file_obj, 'uri') and hasattr(file_obj, 'mime_type'):
                            uri = file_obj.uri
                            mime_type = file_obj.mime_type
                            
                            logger.info(f"⚠️ FilePart con URI detectada: {uri}")
                            logger.warning("FileWithUri no está implementado aún. Se debe descargar la imagen desde la URI.")
                            # TODO: Implementar descarga desde URI si es necesario
                            # import httpx
                            # async with httpx.AsyncClient() as client:
                            #     response = await client.get(uri)
                            #     image_bytes = response.content
                            #     image_data = base64.b64encode(image_bytes).decode('utf-8')
                            #     images.append({'data': image_data, 'mime_type': mime_type})
                        
                        else:
                            logger.warning(f"⚠️ Estructura de file no reconocida: {dir(file_obj)}")
                    
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo FilePart: {e}", exc_info=True)
            
            # OPCIÓN 3: Intentar detectar por tipo de clase
            elif part_root.__class__.__name__ in ['ImagePart', 'FilePart']:
                logger.info(f"Detectado por nombre de clase: {part_root.__class__.__name__}")
                try:
                    # Intentar extraer como ImagePart
                    if hasattr(part_root, 'data') and hasattr(part_root, 'mime_type'):
                        image_data = part_root.data
                        mime_type = part_root.mime_type
                        
                        if isinstance(image_data, bytes):
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        
                        images.append({
                            'data': image_data,
                            'mime_type': mime_type
                        })
                        logger.info(f"✅ Imagen extraída por clase: {mime_type}")
                    # Intentar extraer como FilePart
                    elif hasattr(part_root, 'file'):
                        file_obj = part_root.file
                        if hasattr(file_obj, 'bytes'):
                            image_data = file_obj.bytes
                            if isinstance(image_data, bytes):
                                image_data = base64.b64encode(image_data).decode('utf-8')
                            images.append({
                                'data': image_data,
                                'mime_type': file_obj.mime_type
                            })
                            logger.info(f"✅ FilePart extraída por clase: {file_obj.mime_type}")
                except Exception as e:
                    logger.warning(f"❌ Error en extracción por clase: {e}")
            
            else:
                logger.debug(f"Parte {idx} ignorada: kind='{part_kind}'")
        
        logger.info(f"📊 Total de imágenes extraídas: {len(images)}")
        
        # Debug: Mostrar información de las imágenes extraídas
        for i, img in enumerate(images):
            data_preview = img['data'][:50] if isinstance(img['data'], str) else str(type(img['data']))
            logger.info(f"  Imagen {i}: {img['mime_type']}, data: {data_preview}...")
        
        return images
    
    def _extract_text_from_message(self, context: RequestContext) -> str:
        """
        Extrae el texto del mensaje del usuario.
        
        Returns:
            Texto combinado de todas las partes de texto
        """
        text_parts = []
        
        if not context.message or not context.message.parts:
            return ""
        
        for part in context.message.parts:
            part_root = part.root
            part_kind = getattr(part_root, 'kind', None)
            
            # Verificar si es texto usando kind
            if part_kind == 'text':
                if hasattr(part_root, 'text'):
                    text_parts.append(part_root.text)
                    logger.debug(f"Texto extraído: {part_root.text[:50]}...")
            # Verificar por tipo de clase
            elif part_root.__class__.__name__ == 'TextPart':
                if hasattr(part_root, 'text'):
                    text_parts.append(part_root.text)
                    logger.debug(f"Texto extraído (por clase): {part_root.text[:50]}...")
        
        combined_text = " ".join(text_parts).strip()
        logger.info(f"📝 Texto extraído total: {combined_text[:100]}...")
        return combined_text
    
    def _validate_request(self, context: RequestContext) -> bool:
        """
        Valida que la solicitud tenga al menos texto o imágenes.
        
        Returns:
            True si hay error, False si es válida
        """
        text = self._extract_text_from_message(context)
        images = self._extract_images_from_message(context)
        
        if not text and not images:
            logger.error("❌ Solicitud inválida: sin texto ni imágenes")
            return True
        
        logger.info(f"✅ Solicitud válida: texto={bool(text)}, imágenes={len(images)}")
        return False
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Ejecuta el agente médico.
        
        Args:
            context: Contexto de la solicitud con mensaje y metadatos
            event_queue: Cola de eventos para comunicar progreso
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 Iniciando ejecución de MedicalAgentExecutor")
        logger.info("="*80)
        
        # Validar solicitud
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())
        
        # Extraer contenido
        query = self._extract_text_from_message(context)
        images = self._extract_images_from_message(context)
        
        # Si no hay texto pero hay imágenes, usar texto por defecto
        if not query and images:
            query = "Por favor, analiza estas imágenes médicas."
            logger.info("ℹ️ Usando texto por defecto para análisis de imágenes")
        
        logger.info(f"📋 Consulta médica: {query[:100]}...")
        logger.info(f"🖼️ Imágenes adjuntas: {len(images)}")
        
        # Obtener o crear tarea
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
            logger.info(f"✨ Nueva tarea creada: {task.id}")
        else:
            logger.info(f"♻️ Usando tarea existente: {task.id}")
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Procesar con el agente
            logger.info("🔄 Iniciando streaming del agente médico...")
            
            async for item in self.agent.stream(query, task.context_id, images):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']
                content = item['content']
                status = item.get('status', 'general')
                
                logger.debug(f"📦 Item recibido: complete={is_task_complete}, input={require_user_input}, status={status}")
                
                if not is_task_complete and not require_user_input:
                    # Actualizar progreso
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                    logger.info(f"⚙️ Estado actualizado: {status}")
                    
                elif require_user_input:
                    # Requiere input adicional del usuario
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    logger.info("⏸️ Esperando input del usuario")
                    break
                    
                else:
                    # Tarea completada - agregar resultado como artifact
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='medical_analysis',
                    )
                    await updater.complete()
                    logger.info("✅ Tarea completada exitosamente")
                    logger.info(f"📄 Respuesta: {len(content)} caracteres")
                    break
        
        except Exception as e:
            logger.error(f'❌ Error durante la ejecución del agente médico: {e}', exc_info=True)
            raise ServerError(error=InternalError()) from e
        
        finally:
            logger.info("="*80)
            logger.info("🏁 Ejecución de MedicalAgentExecutor finalizada")
            logger.info("="*80 + "\n")
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancelar ejecución (no soportado actualmente)."""
        logger.warning("⚠️ Intento de cancelación (operación no soportada)")
        raise ServerError(error=UnsupportedOperationError())
