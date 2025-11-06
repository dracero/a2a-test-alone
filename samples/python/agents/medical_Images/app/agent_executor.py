import base64
import logging
from typing import Any

import httpx  # Asegúrate de que httpx esté en tu pyproject.toml
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
    
    async def _extract_images_from_message(self, context: RequestContext) -> list[dict]:
        """
        Extrae imágenes del mensaje del usuario.
        Soporta ImagePart (kind='image') y FilePart (kind='file').
        """
        images = []
        
        # --- DEBUG LOG 1 ---
        if not context.message or not context.message.parts:
            logger.info("DEBUG EXECUTOR: _extract_images_from_message FUE LLAMADO PERO NO HAY PARTES (context.message.parts está vacío)")
            return images
        
        # --- DEBUG LOG 2 ---
        logger.info(f"DEBUG EXECUTOR: Procesando {len(context.message.parts)} partes del mensaje")
        
        # --- DEBUG LOG 3 ---
        if context.message.metadata:
            logger.info(f"DEBUG EXECUTOR: Metadata del mensaje: {context.message.metadata}")
        else:
            logger.info("DEBUG EXECUTOR: El mensaje NO tiene metadata")

        logger.info("DEBUG EXECUTOR: Iniciando bucle de partes...")
        # ---------------------

        for idx, part in enumerate(context.message.parts):
            part_root = part.root
            part_kind = getattr(part_root, 'kind', None)
            
            # --- DEBUG LOG 4 (El más importante) ---
            logger.info(f"DEBUG EXECUTOR Parte {idx}: kind='{part_kind}', tipo={type(part_root).__name__}")
            logger.info(f"DEBUG EXECUTOR Parte {idx} (repr): {repr(part_root)}")
            # ---------------------------------------
            
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
                            logger.info(f"✅ FilePart (bytes) extraída: {mime_type}")
                        
                        # FileWithUri
                        elif hasattr(file_obj, 'uri') and hasattr(file_obj, 'mime_type'):
                            try:
                                # 1. Construir la URL completa
                                host_url = None
                                if context.message.metadata:
                                    host_url = context.message.metadata.get('host_base_url')

                                # --- DEBUG LOG 5 ---
                                logger.info(f"DEBUG EXECUTOR (FileWithUri): host_url extraído de metadata: {host_url}")
                                
                                if not host_url:
                                    host_url = "http://localhost:8080" # ¡AJUSTA ESTO si tu UI no corre en 8080!
                                    logger.warning(f"DEBUG EXECUTOR: No se encontró 'host_base_url', usando fallback: {host_url}")
                                
                                image_url = file_obj.uri
                                if not image_url.startswith('http'):
                                    image_url = f"{host_url.rstrip('/')}/{image_url.lstrip('/')}"
                                
                                # --- DEBUG LOG 6 ---
                                logger.info(f"DEBUG EXECUTOR (FileWithUri): Intentando descargar desde URL completa: {image_url}")
                                
                                # 2. Descargar la imagen
                                async with httpx.AsyncClient() as client:
                                    response = await client.get(image_url)
                                    response.raise_for_status() # Lanza error si no es 200
                                    image_data_bytes = response.content
                                
                                # 3. Codificar y agregar
                                image_data_b64 = base64.b64encode(image_data_bytes).decode('utf-8')
                                images.append({
                                    'data': image_data_b64,
                                    'mime_type': file_obj.mime_type
                                })
                                logger.info(f"✅ FileWithUri extraída: {file_obj.mime_type} desde {image_url}")

                            except Exception as e:
                                logger.warning(f"❌ DEBUG EXECUTOR: Error extrayendo FileWithUri ({file_obj.uri}): {e}", exc_info=True)
                    
                except Exception as e:
                    logger.warning(f"❌ DEBUG EXECUTOR: Error extrayendo FilePart: {e}", exc_info=True)
            
            # Fallback por nombre de clase
            elif part_root.__class__.__name__ in ['ImagePart', 'FilePart']:
                try:
                    if hasattr(part_root, 'data'):
                        # ... (lógica de fallback) ...
                        pass
                    elif hasattr(part_root, 'file'):
                        # ... (lógica de fallback) ...
                        pass
                except Exception as e:
                    logger.warning(f"❌ DEBUG EXECUTOR: Error en fallback: {e}")
        
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
        if not context.message or not context.message.parts:
            logger.error("❌ Solicitud inválida: sin partes de mensaje")
            return True
        
        logger.info("✅ Solicitud válida (partes detectadas)")
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
        images = await self._extract_images_from_message(context)
        
        if not query and not images:
             logger.error("❌ Solicitud inválida: sin texto ni imágenes extraíbles")
             raise ServerError(error=InvalidParamsError(message="No text or images found in message"))

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
                    if chunk_count == 1 or chunk_count % 2 == 0:
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
                
                await updater.add_artifact(
                    [Part(root=TextPart(text=final_response))],
                    name='medical_analysis',
                )
                logger.info("✅ Artifact agregado")
                
                await updater.complete()
                logger.info("✅ Tarea completada y marcada como finalizada")
                
            else:
                logger.error("❌ No se recibió respuesta final del agente")
                has_error = True
                
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
