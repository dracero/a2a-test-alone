"""
Base Agent Executor — Patrón Template Method (Refactoring Guru / GoF).

Define el esqueleto algorítmico invariable para la ejecución de agentes bajo el
protocolo Agent2Agent (A2A):
  1. Validación del RequestContext
  2. Extracción polimórfica de texto e imágenes (ImagePart, FilePart, inline_data, URL)
  3. Inicialización o recuperación de la tarea y TaskUpdater
  4. Consumo del stream del agente con manejo de estados (input_required, working, completed)
  5. Generación de artefactos y finalización
  6. Resiliencia y recuperación ante roturas de pipes / encoding

Las subclases concretas solo implementan las operaciones primitivas:
  - get_agent_stream()
  - get_artifact_name()
  - get_agent_title()
"""

import os
import sys
import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional, List, Dict

import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)


class BaseA2AAgentExecutor(AgentExecutor, ABC):
    """Clase base que implementa el Patrón Template Method para agentes A2A."""

    # ── Operaciones Primitivas (Hooks abstractos que debe definir cada agente) ──

    @abstractmethod
    def get_agent_stream(
        self, query: str, context_id: str, images: List[Dict[str, Any]]
    ) -> AsyncIterator[Any]:
        """Produce el stream de respuestas/estados del agente específico."""
        pass

    @abstractmethod
    def get_artifact_name(self) -> str:
        """Nombre del artefacto final devuelto por el agente."""
        pass

    def get_agent_title(self) -> str:
        """Nombre identificativo del agente para logs."""
        return self.__class__.__name__

    def get_fallback_query(self, images: List[Dict[str, Any]]) -> str:
        """Query por defecto cuando solo se proporcionan imágenes."""
        return "Por favor, analiza las imágenes proporcionadas."

    # ── Métodos Auxiliares y Reutilizables ──

    def _extract_text_from_message(self, context: RequestContext) -> str:
        """Extrae el contenido textual del mensaje de forma segura."""
        text_parts = []
        if not context.message or not context.message.parts:
            return ""

        for part in context.message.parts:
            part_root = getattr(part, 'root', part)
            part_kind = getattr(part_root, 'kind', None)
            part_class_name = type(part_root).__name__

            if part_kind == 'text' or part_class_name == 'TextPart':
                if hasattr(part_root, 'text') and part_root.text:
                    text_parts.append(part_root.text)

        combined_text = " ".join(text_parts).strip()
        logger.info(f"📝 [{self.get_agent_title()}] Texto extraído: {combined_text}")
        return combined_text

    async def _extract_images_from_message(self, context: RequestContext) -> List[Dict[str, Any]]:
        """Extrae imágenes del mensaje soportando formatos:
        - ImagePart (bytes o base64)
        - FilePart (FileWithBytes o FileWithUri)
        - inline_data (ADK format)
        """
        images = []
        if not context.message or not context.message.parts:
            return images

        for idx, part in enumerate(context.message.parts):
            # Normalizar diccionarios de Pydantic v1/v2 si aplica
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
                    image_data = base64.b64encode(image_bytes).decode('utf-8') if isinstance(image_bytes, bytes) else str(image_bytes)
                    images.append({'data': image_data, 'mime_type': mime_type})
                    continue

            # 2. Caso dict con root/file
            if 'root' in part_dict and isinstance(part_dict['root'], dict):
                root_dict = part_dict['root']
                if root_dict.get('kind') == 'file' and 'file' in root_dict:
                    file_dict = root_dict['file']
                    if isinstance(file_dict, dict) and file_dict.get('bytes'):
                        b_data = file_dict['bytes']
                        m_type = file_dict.get('mime_type', 'image/png')
                        i_data = base64.b64encode(b_data).decode('utf-8') if isinstance(b_data, bytes) else str(b_data)
                        images.append({'data': i_data, 'mime_type': m_type})
                        continue

            part_root = getattr(part, 'root', part)
            part_kind = getattr(part_root, 'kind', None)
            part_class_name = type(part_root).__name__

            # 3. ImagePart de A2A
            if part_kind == 'image' or part_class_name == 'ImagePart':
                try:
                    if hasattr(part_root, 'data') and hasattr(part_root, 'mime_type'):
                        image_data = part_root.data
                        mime_type = part_root.mime_type
                        if isinstance(image_data, bytes):
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        elif isinstance(image_data, str):
                            pass
                        else:
                            continue
                        images.append({'data': image_data, 'mime_type': mime_type})
                        continue
                except Exception as e:
                    logger.warning(f"❌ Error extrayendo ImagePart: {e}")

            # 4. FilePart de A2A
            elif part_kind == 'file' or part_class_name == 'FilePart':
                try:
                    if hasattr(part_root, 'file'):
                        file_obj = part_root.file
                        if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
                            image_data = file_obj.bytes
                            mime_type = file_obj.mime_type
                            if isinstance(image_data, bytes):
                                image_data = base64.b64encode(image_data).decode('utf-8')
                            elif isinstance(image_data, str):
                                pass
                            else:
                                continue
                            images.append({'data': image_data, 'mime_type': mime_type})
                            continue

                        elif hasattr(file_obj, 'uri') and hasattr(file_obj, 'mime_type'):
                            try:
                                ui_host = os.getenv("A2A_UI_HOST", "localhost")
                                if ui_host == "0.0.0.0":
                                    ui_host = "localhost"
                                ui_port = os.getenv("A2A_UI_PORT", "12000")
                                host_url = f"http://{ui_host}:{ui_port}"
                                if context.message.metadata:
                                    host_url = context.message.metadata.get('host_base_url', host_url)

                                image_url = file_obj.uri
                                if not image_url.startswith('http'):
                                    image_url = f"{host_url.rstrip('/')}/{image_url.lstrip('/')}"

                                async with httpx.AsyncClient() as client:
                                    resp = await client.get(image_url)
                                    resp.raise_for_status()
                                    image_data_bytes = resp.content

                                image_data_b64 = base64.b64encode(image_data_bytes).decode('utf-8')
                                images.append({'data': image_data_b64, 'mime_type': file_obj.mime_type})
                            except Exception as e:
                                logger.error(f"❌ Error descargando archivo desde URI: {e}")
                except Exception as e:
                    logger.warning(f"❌ Error procesando FilePart: {e}")

        return images

    def _validate_request(self, context: RequestContext) -> bool:
        """Verifica si el contexto contiene un mensaje válido."""
        return not (context.message and context.message.parts)

    # ── Template Method (Esqueleto Invariable de Ejecución GoF) ──

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Template Method que implementa el ciclo de vida de ejecución A2A."""
        agent_title = self.get_agent_title()
        logger.info("\n" + "=" * 80)
        logger.info(f"🚀 INICIANDO EJECUCIÓN {agent_title.upper()}")
        logger.info(f"   Task ID: {context.task_id}")
        logger.info(f"   Context ID: {context.context_id}")

        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        query = self._extract_text_from_message(context)
        images = await self._extract_images_from_message(context) or []

        if not query and not images:
            logger.error(f"❌ [{agent_title}] Solicitud inválida: sin texto ni imágenes")
            raise ServerError(error=InvalidParamsError(message="No text or images found"))

        if not query and images:
            query = self.get_fallback_query(images)

        logger.info(f"📋 [{agent_title}] Query: {query}")
        logger.info(f"🖼️ [{agent_title}] Imágenes: {len(images)}")

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
            logger.info(f"🔄 [{agent_title}] Iniciando streaming del agente...")
            chunk_count = 0
            last_status = None

            async for item in self.get_agent_stream(query, task.context_id, images):
                chunk_count += 1
                if hasattr(item, 'dict'):
                    item_dict = item.dict()
                elif hasattr(item, 'model_dump'):
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
                    logger.info("⏸️ Requiere input del usuario")
                    handled_as_input_required = True
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(content, task.context_id, task.id),
                        final=True,
                    )
                    break
                elif is_complete:
                    final_response = content
                    logger.info(f"🎉 RESPUESTA FINAL ({len(content)} caracteres)")
                    break
                else:
                    if chunk_count % 2 == 0:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(content, task.context_id, task.id),
                        )

            if handled_as_input_required:
                logger.info("✅ Tarea en espera de input del usuario (input_required)")
            elif final_response:
                logger.info("📤 Enviando respuesta final...")
                await updater.add_artifact(
                    [Part(root=TextPart(text=final_response))],
                    name=self.get_artifact_name(),
                )
                await updater.complete()
                logger.info("✅ Tarea completada exitosamente")
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

        except OSError as pipe_err:
            logger.warning(f"⚠️ Pipe/OS error (recuperando): {pipe_err}")
            if sys.platform.startswith('win'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    pass

            try:
                async for item in self.get_agent_stream(query, task.context_id, images):
                    item_dict = item.model_dump() if hasattr(item, 'model_dump') else (item.dict() if hasattr(item, 'dict') else item)
                    if not isinstance(item_dict, dict):
                        continue

                    if item_dict.get('require_user_input', False):
                        await updater.update_status(
                            TaskState.input_required,
                            new_agent_text_message(item_dict.get('content', ''), task.context_id, task.id),
                            final=True,
                        )
                        return
                    elif item_dict.get('is_task_complete', False):
                        await updater.add_artifact(
                            [Part(root=TextPart(text=item_dict.get('content', '')))],
                            name=self.get_artifact_name(),
                        )
                        await updater.complete()
                        return
            except Exception as retry_err:
                has_error = True
                raise ServerError(error=InternalError()) from retry_err

        except Exception as e:
            logger.error(f"❌ EXCEPCIÓN en {agent_title}: {e}", exc_info=True)
            has_error = True
            try:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(f"Error interno: {str(e)}", task.context_id, task.id),
                    final=True,
                )
            except Exception:
                pass
            raise ServerError(error=InternalError()) from e

        finally:
            logger.info("=" * 80)
            status_msg = "❌ FINALIZADA CON ERRORES" if has_error else "✅ FINALIZADA EXITOSAMENTE"
            logger.info(f"{status_msg} ({agent_title})")
            logger.info("=" * 80 + "\n")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancelación estándar de la tarea."""
        logger.warning(f"⚠️ Cancelación no soportada en {self.get_agent_title()}")
        raise ServerError(error=UnsupportedOperationError())
