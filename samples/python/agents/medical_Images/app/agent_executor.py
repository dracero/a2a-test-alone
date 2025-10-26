import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (ImagePart, InternalError, InvalidParamsError, Part,
                       TaskState, TextPart, UnsupportedOperationError)
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
        
        Returns:
            Lista de diccionarios con 'data' (bytes) y 'mime_type' (str)
        """
        images = []
        
        if not context.message or not context.message.parts:
            return images
        
        for part in context.message.parts:
            part_root = part.root
            
            # Verificar si es una imagen
            if hasattr(part_root, '__class__') and part_root.__class__.__name__ == 'ImagePart':
                try:
                    # Extraer datos de la imagen
                    if hasattr(part_root, 'data') and hasattr(part_root, 'mime_type'):
                        images.append({
                            'data': part_root.data,
                            'mime_type': part_root.mime_type
                        })
                        logger.info(f"Imagen detectada: {part_root.mime_type}")
                except Exception as e:
                    logger.warning(f"Error extrayendo imagen: {e}")
        
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
            
            # Verificar si es texto
            if hasattr(part_root, '__class__') and part_root.__class__.__name__ == 'TextPart':
                if hasattr(part_root, 'text'):
                    text_parts.append(part_root.text)
        
        return " ".join(text_parts).strip()
    
    def _validate_request(self, context: RequestContext) -> bool:
        """
        Valida que la solicitud tenga al menos texto o imágenes.
        
        Returns:
            True si hay error, False si es válida
        """
        text = self._extract_text_from_message(context)
        images = self._extract_images_from_message(context)
        
        if not text and not images:
            logger.error("Solicitud inválida: sin texto ni imágenes")
            return True
        
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
        
        logger.info(f"Consulta médica recibida: {query[:100]}...")
        if images:
            logger.info(f"Imágenes adjuntas: {len(images)}")
        
        # Obtener o crear tarea
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Procesar con el agente
            async for item in self.agent.stream(query, task.context_id, images):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']
                content = item['content']
                status = item.get('status', 'general')
                
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
                    break
                else:
                    # Tarea completada - agregar resultado como artifact
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='medical_analysis',
                    )
                    await updater.complete()
                    break
        
        except Exception as e:
            logger.error(f'Error durante la ejecución del agente médico: {e}', exc_info=True)
            raise ServerError(error=InternalError()) from e
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancelar ejecución (no soportado actualmente)."""
        raise ServerError(error=UnsupportedOperationError())
