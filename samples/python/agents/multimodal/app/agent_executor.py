# samples/python/agents/multimodal/app/agent_executor.py (CORREGIDO)

import base64
import logging
from typing import Any

import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (InternalError, InvalidParamsError, Part, TaskState,
                       TextPart, UnsupportedOperationError)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from app.agent import PhysicsMultimodalAgent
from app.langsmith_config import traceable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsAgentExecutor(AgentExecutor):
    """Executor para el Asistente de Física Multimodal."""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        """
        Inicializar executor.
        
        Args:
            qdrant_url: URL de Qdrant
            qdrant_api_key: API Key de Qdrant
        """
        self.agent = PhysicsMultimodalAgent(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
    
    async def _extract_images_from_message(self, context: RequestContext) -> list[dict]:
        """
        Extrae imágenes del mensaje del usuario.
        Soporta ImagePart (kind='image') y FilePart (kind='file').
        """
        images = []
        if not context.message or not context.message.parts:
            logger.info("DEBUG: _extract_images_from_message sin partes.")
            return images

        logger.info(f"DEBUG: Procesando {len(context.message.parts)} partes del mensaje")

        for idx, part in enumerate(context.message.parts):
            # Obtener dict para soportar Pydantic v2 / inline_data de ADK
            if hasattr(part, 'model_dump'):
                part_dict = part.model_dump()
            elif hasattr(part, 'dict'):
                part_dict = part.dict()
            elif isinstance(part, dict):
                part_dict = part
            else:
                part_dict = {}

            # 1. Caso inline_data (formato ADK host)
            if 'inline_data' in part_dict:
                inline_data = part_dict['inline_data']
                if isinstance(inline_data, dict):
                    image_bytes = inline_data.get('data')
                    mime_type = inline_data.get('mime_type', 'image/png')
                else:
                    image_bytes = getattr(inline_data, 'data', None)
                    mime_type = getattr(inline_data, 'mime_type', 'image/png')
                
                if image_bytes:
                    if isinstance(image_bytes, bytes):
                        image_data = base64.b64encode(image_bytes).decode('utf-8')
                    else:
                        image_data = str(image_bytes)
                    images.append({'data': image_data, 'mime_type': mime_type})
                    logger.info(f"✅ inline_data extraída: {mime_type}, {len(image_data)} chars")
                    continue

            # 2. Caso dict con root/file
            if 'root' in part_dict and isinstance(part_dict['root'], dict):
                root_dict = part_dict['root']
                if root_dict.get('kind') == 'file' and 'file' in root_dict:
                    file_dict = root_dict['file']
                    if isinstance(file_dict, dict) and 'bytes' in file_dict and file_dict['bytes']:
                        b_data = file_dict['bytes']
                        m_type = file_dict.get('mime_type', 'image/png')
                        if isinstance(b_data, bytes):
                            i_data = base64.b64encode(b_data).decode('utf-8')
                        else:
                            i_data = str(b_data)
                        images.append({'data': i_data, 'mime_type': m_type})
                        logger.info(f"✅ Dict FilePart extraída: {m_type}, {len(i_data)} chars")
                        continue

            part_root = getattr(part, 'root', part)
            part_kind = getattr(part_root, 'kind', None)
            part_class_name = type(part_root).__name__

            logger.info(f"DEBUG Parte {idx}: kind='{part_kind}', tipo='{part_class_name}'")

            # ImagePart
            if part_kind == 'image' or part_class_name == 'ImagePart':
                try:
                    if hasattr(part_root, 'data') and hasattr(part_root, 'mime_type'):
                        image_data = part_root.data
                        mime_type = part_root.mime_type
                        
                        if isinstance(image_data, bytes):
                            logger.info(f"✅ ImagePart (bytes): {mime_type}, {len(image_data)} bytes")
                            # Convertir a base64 para consistencia
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        elif isinstance(image_data, str):
                            logger.info(f"✅ ImagePart (string): {mime_type}, {len(image_data)} chars")
                        else:
                            logger.warning(f"⚠️ ImagePart con data desconocida: {type(image_data)}")
                            continue
                        
                        images.append({
                            'data': image_data,
                            'mime_type': mime_type
                        })
                        continue
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo ImagePart: {e}")

            # FilePart
            elif part_kind == 'file' or part_class_name == 'FilePart':
                try:
                    if hasattr(part_root, 'file'):
                        file_obj = part_root.file
                        logger.debug(f"FilePart detectada, tipo: {type(file_obj).__name__}")

                        # FileWithBytes
                        if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
                            image_data = file_obj.bytes
                            mime_type = file_obj.mime_type
                            
                            if isinstance(image_data, bytes):
                                # Convertir a base64 para consistencia
                                image_data = base64.b64encode(image_data).decode('utf-8')
                                logger.info(f"✅ FilePart (bytes → base64): {mime_type}, {len(image_data)} chars")
                            elif isinstance(image_data, str):
                                logger.info(f"✅ FilePart (string): {mime_type}, {len(image_data)} chars")
                            else:
                                logger.warning(f"⚠️ FilePart con bytes desconocido: {type(image_data)}")
                                continue
                            
                            images.append({
                                'data': image_data,
                                'mime_type': mime_type
                            })
                            continue

                        # FileWithUri
                        elif hasattr(file_obj, 'uri') and hasattr(file_obj, 'mime_type'):
                            try:
                                ui_host = os.getenv("A2A_UI_HOST", "localhost")
                                if ui_host == "0.0.0.0":
                                    ui_host = "localhost"
                                ui_port = os.getenv("A2A_UI_PORT", "12000")
                                host_url = f"http://{ui_host}:{ui_port}"
                                if context.message.metadata:
                                    host_url = context.message.metadata.get('host_base_url', host_url)

                                logger.info(f"DEBUG (FileWithUri): host_url: {host_url}")
                                
                                image_url = file_obj.uri
                                if not image_url.startswith('http'):
                                    image_url = f"{host_url.rstrip('/')}/{image_url.lstrip('/')}"
                                
                                logger.info(f"DEBUG (FileWithUri): Descargando desde: {image_url}")
                                
                                async with httpx.AsyncClient() as client:
                                    response = await client.get(image_url)
                                    response.raise_for_status()
                                    image_data_bytes = response.content
                                
                                # Convertir a base64 para consistencia
                                image_data_b64 = base64.b64encode(image_data_bytes).decode('utf-8')
                                
                                images.append({
                                    'data': image_data_b64,
                                    'mime_type': file_obj.mime_type
                                })
                                logger.info(f"✅ FileWithUri extraída: {file_obj.mime_type}, {len(image_data_b64)} chars")
                                continue

                            except Exception as e:
                                logger.warning(f"❌ Error extrayendo FileWithUri ({file_obj.uri}): {e}", exc_info=True)
                
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo FilePart: {e}", exc_info=True)
        
        logger.info(f"📊 Total imágenes extraídas: {len(images)}")
        for i, img in enumerate(images):
            logger.info(f"   Imagen {i+1}: {img['mime_type']}, {len(img['data'])} chars (base64)")
        
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
        logger.info(f"📝 Texto extraído: {combined_text}")
        return combined_text
    
    def _validate_request(self, context: RequestContext) -> bool:
        """Valida que haya texto o imágenes."""
        if not context.message or not context.message.parts:
            logger.error("❌ Solicitud inválida: sin partes de mensaje")
            return True
        
        logger.info("✅ Solicitud válida (partes detectadas)")
        return False
    
    @traceable(name="physics_agent_execution", run_type="chain", tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"])
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Ejecuta el agente de física."""
        logger.info("\n" + "="*80)
        logger.info("🚀 INICIANDO EJECUCIÓN PHYSICS AGENT")
        logger.info(f"   Task ID: {context.task_id}")
        logger.info(f"   Context ID: {context.context_id}")
        
        # Validar
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())
        
        # Extraer contenido
        query = self._extract_text_from_message(context)
        images = await self._extract_images_from_message(context) or []
        
        if not query and not images:
             logger.error("❌ Solicitud inválida: sin texto ni imágenes")
             raise ServerError(error=InvalidParamsError(message="No text or images found"))

        if not query and images:
            query = "Por favor, analiza estas imágenes de física."
        
        logger.info(f"📋 Query: {query}")
        logger.info(f"🖼️ Imágenes: {len(images)}")
        
        # Obtener o crear tarea
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
            logger.info(f"✨ Nueva tarea creada: {task.id}")
        else:
            logger.info(f"♻️ Usando tarea existente: {task.id}")
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        final_response = None
        has_error = False
        handled_as_input_required = False
        
        try:
            logger.info("🔄 Iniciando streaming del agente...")
            
            chunk_count = 0
            last_status = None
            
            # 🔧 CORRECCIÓN CRÍTICA: Manejar objetos Pydantic
            async for item in self.agent.stream(query, task.context_id, images):
                chunk_count += 1
                
                # Convertir a dict si es un objeto Pydantic
                if hasattr(item, 'dict'):
                    item_dict = item.dict()
                elif hasattr(item, 'model_dump'):  # Para Pydantic v2
                    item_dict = item.model_dump()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    logger.error(f"❌ Item inválido: {type(item)} - {item}")
                    continue
                
                is_complete = item_dict.get('is_task_complete', False)
                require_input = item_dict.get('require_user_input', False)
                content = item_dict.get('content', '')
                status = item_dict.get('status', 'working')
                
                if status != last_status:
                    logger.info(f"📦 Chunk {chunk_count}: status={status}, complete={is_complete}")
                    last_status = status
                
                if require_input:
                    # CRÍTICO: Procesar require_input ANTES que is_complete
                    # para mantener la tarea viva (ej: preguntas socráticas)
                    logger.info("⏸️ Requiere input del usuario")
                    handled_as_input_required = True
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

                elif is_complete:
                    final_response = content
                    logger.info(f"🎉 RESPUESTA FINAL ({len(content)} caracteres)")
                    break
                    
                else:
                    # Enviar updates cada 2 chunks para no saturar
                    if chunk_count % 2 == 0:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                content,
                                task.context_id,
                                task.id,
                            ),
                        )
            
            logger.info(f"📊 Loop finalizado. chunk_count={chunk_count}, handled_as_input_required={handled_as_input_required}, bool(final_response)={bool(final_response)}")
            
            if handled_as_input_required:
                logger.info("✅ Tarea en espera de input del usuario (input_required)")
            elif final_response:
                logger.info("📤 Enviando respuesta final...")
                
                await updater.add_artifact(
                    [Part(root=TextPart(text=final_response))],
                    name='physics_analysis',
                )
                logger.info("✅ Artifact agregado")
                
                await updater.complete()
                logger.info("✅ Tarea completada")
                
            else:
                if not has_error:
                    logger.error("❌ No se recibió respuesta final")
                    has_error = True
                    await updater.update_status(
                        TaskState.failed,
                        new_agent_text_message(
                            "Error: No se pudo generar una respuesta completa.",
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
            
            logger.info(f"📊 Total chunks: {chunk_count}")
        
        except OSError as pipe_err:
            # WinError 233 or other pipe errors — typically from stdout/stderr
            # pipe breaking when parent process restarts. Not a real agent error.
            logger.warning(f'⚠️ Pipe/OS error (non-fatal): {pipe_err}')
            import sys
            if sys.platform.startswith('win'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    pass
            # Retry the stream once after fixing pipes
            try:
                async for item in self.agent.stream(query, task.context_id, images):
                    if hasattr(item, 'model_dump'):
                        item_dict = item.model_dump()
                    elif isinstance(item, dict):
                        item_dict = item
                    else:
                        continue
                    
                    is_complete = item_dict.get('is_task_complete', False)
                    require_input = item_dict.get('require_user_input', False)
                    content = item_dict.get('content', '')
                    
                    if require_input:
                        await updater.update_status(
                            TaskState.input_required,
                            new_agent_text_message(content, task.context_id, task.id),
                            final=True,
                        )
                        return
                    elif is_complete:
                        await updater.add_artifact(
                            [Part(root=TextPart(text=content))],
                            name='physics_analysis',
                        )
                        await updater.complete()
                        return
            except Exception as retry_err:
                logger.error(f'❌ Retry also failed: {retry_err}')
                has_error = True
                try:
                    await updater.update_status(
                        TaskState.failed,
                        new_agent_text_message(
                            f"Error interno: {str(retry_err)}",
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                except:
                    pass
                raise ServerError(error=InternalError()) from retry_err
        
        except Exception as e:
            logger.error(f'❌ EXCEPCIÓN: {e}', exc_info=True)
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
