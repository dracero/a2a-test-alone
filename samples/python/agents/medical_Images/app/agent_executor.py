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
        Soporta ImagePart (kind='image') y FilePart (kind='file').
        """
        images = []
        
        if not context.message or not context.message.parts:
            logger.info("No hay partes en el mensaje")
            return images
        
        logger.info(f"Procesando {len(context.message.parts)} partes del mensaje")
        
        for idx, part in enumerate(context.message.parts):
            part_root = part.root
            part_kind = getattr(part_root, 'kind', None)
            
            logger.debug(f"Parte {idx}: kind='{part_kind}', tipo={type(part_root).__name__}")
            
            # ImagePart
            if part_kind == 'image':
                try:
                    if hasattr(part_root, 'data') and hasattr(part_root, 'mime_type'):
                        image_data = part_root.data
                        mime_type = part_root.mime_type
                        
                        if isinstance(image_data, bytes):
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        
                        images.append({
                            'data': image_data,
                            'mime_type': mime_type
                        })
                        logger.info(f"✅ ImagePart extraída: {mime_type}")
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo ImagePart: {e}")
            
            # FilePart - LO QUE ENVÍA EL CLIENTE
            elif part_kind == 'file':
                try:
                    if hasattr(part_root, 'file'):
                        file_obj = part_root.file
                        
                        logger.debug(f"FilePart detectada, tipo: {type(file_obj).__name__}")
                        
                        # FileWithBytes
                        if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
                            image_data = file_obj.bytes
                            mime_type = file_obj.mime_type
                            
                            if isinstance(image_data, bytes):
                                image_data = base64.b64encode(image_data).decode('utf-8')
                            
                            images.append({
                                'data': image_data,
                                'mime_type': mime_type
                            })
                            logger.info(f"✅ FilePart extraída: {mime_type}")
                        
                        # FileWithUri
                        elif hasattr(file_obj, 'uri') and hasattr(file_obj, 'mime_type'):
                            logger.warning(f"⚠️ FileWithUri no implementado: {file_obj.uri}")
                    
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo FilePart: {e}", exc_info=True)
            
            # Fallback por nombre de clase
            elif part_root.__class__.__name__ in ['ImagePart', 'FilePart']:
                try:
                    if hasattr(part_root, 'data'):
                        image_data = part_root.data
                        if isinstance(image_data, bytes):
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        images.append({
                            'data': image_data,
                            'mime_type': getattr(part_root, 'mime_type', 'image/png')
                        })
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
                except Exception as e:
                    logger.warning(f"❌ Error en fallback: {e}")
        
        logger.info(f"📊 Total imágenes extraídas: {len(images)}")
        return images
    
    def _extract_text_from_message(self, context: RequestContext) -> str:
        """Extrae el texto del mensaje."""
        text_parts = []
        
        if not context.message or not context.message.parts:
            return ""
        
        for part in context.message.parts:
            part_root = part.root
            part_kind = getattr(part_root, 'kind', None)
            
            if part_kind == 'text' or part_root.__class__.__name__ == 'TextPart':
                if hasattr(part_root, 'text'):
                    text_parts.append(part_root.text)
        
        combined_text = " ".join(text_parts).strip()
        logger.info(f"📝 Texto extraído: {combined_text[:100]}...")
        return combined_text
    
    def _validate_request(self, context: RequestContext) -> bool:
        """Valida que haya texto o imágenes."""
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
        """Ejecuta el agente médico."""
        logger.info("\n" + "="*80)
        logger.info("🚀 INICIANDO EJECUCIÓN MEDICAL AGENT")
        logger.info(f"   Task ID: {context.task_id}")
        logger.info(f"   Context ID: {context.context_id}")
        logger.info(f"   Message ID: {context.message.message_id if context.message else 'N/A'}")
        logger.info("="*80)
        
        # Validar
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())
        
        # Extraer contenido
        query = self._extract_text_from_message(context)
        images = self._extract_images_from_message(context)
        
        if not query and images:
            query = "Por favor, analiza estas imágenes médicas."
        
        logger.info(f"📋 Query: {query[:100]}...")
        logger.info(f"🖼️ Imágenes: {len(images)}")
        
        # CRÍTICO: Obtener o crear tarea
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
            logger.info(f"✨ Nueva tarea creada: {task.id}")
        else:
            logger.info(f"♻️ Usando tarea existente: {task.id}")
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        # Variables de estado
        final_response = None
        has_error = False
        
        try:
            logger.info("🔄 Iniciando streaming del agente...")
            
            chunk_count = 0
            last_status = None
            
            async for item in self.agent.stream(query, task.context_id, images):
                chunk_count += 1
                
                # Validar estructura
                if not isinstance(item, dict):
                    logger.error(f"❌ Item inválido (no es dict): {type(item)}")
                    continue
                
                is_complete = item.get('is_task_complete', False)
                require_input = item.get('require_user_input', False)
                content = item.get('content', '')
                status = item.get('status', 'general')
                
                # Solo log si el estado cambió (reducir spam)
                if status != last_status:
                    logger.info(f"📦 Chunk {chunk_count}: status={status}, complete={is_complete}")
                    last_status = status
                
                if is_complete:
                    # RESPUESTA FINAL - Guardar para procesar después del loop
                    final_response = content
                    logger.info(f"🎉 RESPUESTA FINAL RECIBIDA ({len(content)} caracteres)")
                    logger.info(f"   Preview: {content[:200]}...")
                    break
                    
                elif require_input:
                    # Requiere input del usuario
                    logger.info("⏸️ Requiere input del usuario")
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                    
                else:
                    # Estado intermedio - actualizar progreso
                    # Solo actualizar si hay cambio significativo
                    if chunk_count == 1 or chunk_count % 2 == 0:  # Cada 2 chunks
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                content,
                                task.context_id,
                                task.id,
                            ),
                        )
            
            # PROCESAR RESPUESTA FINAL FUERA DEL LOOP
            if final_response:
                logger.info("📤 Enviando respuesta final al cliente...")
                
                # 1. Agregar como artifact
                await updater.add_artifact(
                    [Part(root=TextPart(text=final_response))],
                    name='medical_analysis',
                )
                logger.info("✅ Artifact agregado")
                
                # 2. Completar la tarea
                await updater.complete()
                logger.info("✅ Tarea completada y marcada como finalizada")
                
            else:
                # No se recibió respuesta final
                logger.error("❌ No se recibió respuesta final del agente")
                has_error = True
                
                # Marcar tarea como fallida
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        "Error: No se pudo generar una respuesta médica completa.",
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
            
            logger.info(f"📊 Total chunks procesados: {chunk_count}")
        
        except Exception as e:
            logger.error(f'❌ EXCEPCIÓN EN EJECUCIÓN: {e}', exc_info=True)
            has_error = True
            
            # Intentar marcar como fallida
            try:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        f"Error interno: {str(e)}",
                        task.context_id,
                        task.id,
                    ),
                    final=True,
                )
            except:
                pass
            
            raise ServerError(error=InternalError()) from e
        
        finally:
            logger.info("="*80)
            if has_error:
                logger.info("❌ EJECUCIÓN FINALIZADA CON ERRORES")
            else:
                logger.info("✅ EJECUCIÓN FINALIZADA EXITOSAMENTE")
            logger.info("="*80 + "\n")
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancelar (no soportado)."""
        logger.warning("⚠️ Cancelación no soportada")
        raise ServerError(error=UnsupportedOperationError())
