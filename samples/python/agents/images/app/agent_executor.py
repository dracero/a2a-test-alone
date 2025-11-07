"""Agent executor for A2A protocol with LangSmith monitoring."""

import base64
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (FilePart, FileWithBytes, InvalidParamsError, Part, Task,
                       TextPart, UnsupportedOperationError)
from a2a.utils import completed_task, new_artifact
from a2a.utils.errors import ServerError
from app.agent import ImageGenerationAgent
# Import LangSmith configuration
from app.langsmith_config import LANGSMITH_ENABLED, langsmith_client, traceable

logger = logging.getLogger(__name__)


class ImageGenerationAgentExecutor(AgentExecutor):
    """Image Generation AgentExecutor with LangSmith monitoring."""

    def __init__(self) -> None:
        """Initialize the executor with LangSmith tracing."""
        self.agent = ImageGenerationAgent()
        self._is_cancelled = False
        
        # Log executor initialization
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="executor_initialized",
                    value={
                        "agent_type": "ImageGenerationAgent",
                        "langsmith_enabled": True
                    }
                )
                logger.info("✅ AgentExecutor initialized with LangSmith monitoring")
            except Exception as e:
                logger.debug(f"LangSmith feedback error: {e}")

    async def cancel(self, context: RequestContext) -> None:
        """Cancel the current execution.
        
        Args:
            context: Request context for the execution to cancel
        """
        logger.info(f"🛑 Cancelling execution for task: {context.task_id}")
        self._is_cancelled = True
        
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="execution_cancelled",
                    value={
                        "task_id": context.task_id,
                        "context_id": context.context_id
                    }
                )
            except Exception as e:
                logger.debug(f"LangSmith feedback error: {e}")

    @traceable(name="a2a_request_execution", run_type="chain")
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute A2A request with complete LangSmith tracing.
        
        Args:
            context: Request context containing task and user input
            event_queue: Queue for sending events back to client
        """
        # Reset cancellation flag
        self._is_cancelled = False
        
        # Log request start
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="a2a_request_start",
                    value={
                        "task_id": context.task_id,
                        "context_id": context.context_id,
                        "user_input_length": len(context.get_user_input())
                    }
                )
            except Exception as e:
                logger.debug(f"LangSmith feedback error: {e}")
        
        # Validate request
        error = self._validate_request(context)
        if error:
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="a2a_request_validation_error",
                        value={
                            "task_id": context.task_id,
                            "error": "Invalid request parameters"
                        }
                    )
                except:
                    pass
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        
        # Log query details
        logger.info(f"Processing query: {query[:100]}...")
        print(f"📨 Received request - Task ID: {context.task_id}")
        print(f"📝 Query: {query}")
        
        try:
            # Check for cancellation
            if self._is_cancelled:
                logger.info("Execution cancelled before agent execution")
                return
            
            # Execute agent with tracing
            result = await self._execute_agent_with_tracing(
                query, 
                context.context_id,
                context.task_id
            )
            
            # Check for cancellation after execution
            if self._is_cancelled:
                logger.info("Execution cancelled after agent execution")
                return
            
            print(f'✅ Agent execution completed')
            logger.info(f'Final Result: {result}')
            
            # ✅ CORRECCIÓN: Validar el resultado antes de procesarlo
            if result is None:
                error_msg = 'Agent returned None - no response generated'
                logger.error(error_msg)
                print(f'❌ {error_msg}')
                
                parts = [Part(root=TextPart(text=error_msg))]
                
                if LANGSMITH_ENABLED:
                    try:
                        langsmith_client.create_feedback(
                            run_id=None,
                            key="a2a_request_failure",
                            value={
                                "task_id": context.task_id,
                                "context_id": context.context_id,
                                "error": error_msg,
                                "query": query[:200]
                            }
                        )
                    except:
                        pass
                
                await event_queue.enqueue_event(
                    completed_task(
                        context.task_id,
                        context.context_id,
                        [new_artifact(parts, f'error_{context.task_id}')],
                        [context.message],
                    )
                )
                return
            
            # ✅ Convertir resultado a string y validar
            result_str = str(result).strip()
            
            if not result_str:
                error_msg = 'Agent returned empty response'
                logger.error(error_msg)
                print(f'❌ {error_msg}')
                
                parts = [Part(root=TextPart(text=error_msg))]
                
                await event_queue.enqueue_event(
                    completed_task(
                        context.task_id,
                        context.context_id,
                        [new_artifact(parts, f'error_{context.task_id}')],
                        [context.message],
                    )
                )
                return
            
            # ✅ Verificar si es un mensaje de error del agente
            if result_str.startswith("ERROR:"):
                error_msg = result_str[6:].strip()  # Remover prefijo "ERROR:"
                logger.error(f'Agent error: {error_msg}')
                print(f'❌ Agent error: {error_msg}')
                
                parts = [Part(root=TextPart(text=error_msg))]
                
                if LANGSMITH_ENABLED:
                    try:
                        langsmith_client.create_feedback(
                            run_id=None,
                            key="a2a_request_agent_error",
                            value={
                                "task_id": context.task_id,
                                "context_id": context.context_id,
                                "error": error_msg,
                                "query": query[:200]
                            }
                        )
                    except:
                        pass
                
                await event_queue.enqueue_event(
                    completed_task(
                        context.task_id,
                        context.context_id,
                        [new_artifact(parts, f'error_{context.task_id}')],
                        [context.message],
                    )
                )
                return
            
            # ✅ Extraer image_key del resultado
            # El resultado puede ser un objeto con .raw o directamente un string
            try:
                image_key = result.raw if hasattr(result, 'raw') else result_str
            except Exception as e:
                logger.warning(f"Could not extract .raw attribute: {e}")
                image_key = result_str
            
            # ✅ Validar que image_key no esté vacío
            if not image_key or not isinstance(image_key, str):
                error_msg = f'Invalid image key: {image_key}'
                logger.error(error_msg)
                print(f'❌ {error_msg}')
                
                parts = [Part(root=TextPart(text=error_msg))]
                
                await event_queue.enqueue_event(
                    completed_task(
                        context.task_id,
                        context.context_id,
                        [new_artifact(parts, f'error_{context.task_id}')],
                        [context.message],
                    )
                )
                return
            
            # ✅ Procesar la imagen generada
            logger.info(f'Processing image with key: {image_key}')
            print(f'🖼️  Image key: {image_key}')
            
            # ✅ OBTENER LA IMAGEN REAL DEL CACHE
            try:
                image_data = self.agent.get_image_data(context.context_id, image_key)
                
                # Validar que se obtuvo la imagen correctamente
                if image_data.error:
                    error_msg = f'Error retrieving image: {image_data.error}'
                    logger.error(error_msg)
                    print(f'❌ {error_msg}')
                    
                    parts = [Part(root=TextPart(text=error_msg))]
                    
                    await event_queue.enqueue_event(
                        completed_task(
                            context.task_id,
                            context.context_id,
                            [new_artifact(parts, f'error_{context.task_id}')],
                            [context.message],
                        )
                    )
                    return
                
                if not image_data.bytes:
                    error_msg = 'Image data is empty'
                    logger.error(error_msg)
                    print(f'❌ {error_msg}')
                    
                    parts = [Part(root=TextPart(text=error_msg))]
                    
                    await event_queue.enqueue_event(
                        completed_task(
                            context.task_id,
                            context.context_id,
                            [new_artifact(parts, f'error_{context.task_id}')],
                            [context.message],
                        )
                    )
                    return
                
                # ✅ NO decodificar - mantener como base64 string
                # FileWithBytes espera base64 string, no bytes crudos
                image_base64 = image_data.bytes
                
                # Calcular tamaño aproximado para logging
                estimated_size = len(base64.b64decode(image_base64))
                print(f'✅ Image retrieved: ~{estimated_size} bytes')
                
                # Crear parte de archivo con la imagen en base64
                file_part = FilePart(
                    file=FileWithBytes(
                        name=image_data.name or f"{image_key}.png",
                        mime_type=image_data.mime_type or "image/png",
                        bytes=image_base64  # ✅ Base64 string, NO bytes crudos
                    )
                )
                
                parts = [
                    Part(root=TextPart(text=f"Image generated successfully")),
                    Part(root=file_part)
                ]
                
            except Exception as e:
                error_msg = f'Error retrieving image from cache: {str(e)}'
                logger.exception(error_msg)
                print(f'❌ {error_msg}')
                
                parts = [Part(root=TextPart(text=error_msg))]
                
                await event_queue.enqueue_event(
                    completed_task(
                        context.task_id,
                        context.context_id,
                        [new_artifact(parts, f'error_{context.task_id}')],
                        [context.message],
                    )
                )
                return
            
            # Log success
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="a2a_request_success",
                        value={
                            "task_id": context.task_id,
                            "context_id": context.context_id,
                            "image_key": image_key,
                            "image_size_bytes": estimated_size,
                            "query": query[:200]
                        }
                    )
                    logger.info("✅ Request completed successfully")
                except Exception as e:
                    logger.debug(f"LangSmith feedback error: {e}")
            
            # Enviar respuesta completa
            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f'image_{context.task_id}')],
                    [context.message],
                )
            )
            
        except Exception as e:
            error_msg = f'Unexpected error during execution: {str(e)}'
            logger.exception(error_msg)
            print(f'❌ {error_msg}')
            
            # Log error to LangSmith
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="a2a_request_exception",
                        value={
                            "task_id": context.task_id,
                            "context_id": context.context_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "query": query[:200]
                        }
                    )
                except:
                    pass
            
            parts = [Part(root=TextPart(text=f"Error: {str(e)}"))]
            
            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f'error_{context.task_id}')],
                    [context.message],
                )
            )

    async def _execute_agent_with_tracing(
        self, 
        query: str, 
        context_id: str,
        task_id: str
    ) -> any:
        """Execute agent with LangSmith tracing."""
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="agent_execution_start",
                    value={
                        "task_id": task_id,
                        "context_id": context_id,
                        "query_length": len(query)
                    }
                )
            except Exception as e:
                logger.debug(f"LangSmith feedback error: {e}")
        
        # Execute the agent using invoke() method with session_id (context_id)
        # Note: invoke() is synchronous, not async
        result = self.agent.invoke(query, context_id)
        
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="agent_execution_complete",
                    value={
                        "task_id": task_id,
                        "context_id": context_id,
                        "result_type": type(result).__name__,
                        "result": str(result)[:200]  # First 200 chars
                    }
                )
            except Exception as e:
                logger.debug(f"LangSmith feedback error: {e}")
        
        return result

    def _validate_request(self, context: RequestContext) -> str | None:
        """Validate the incoming request."""
        user_input = context.get_user_input()
        
        if not user_input or not user_input.strip():
            return "Empty user input"
        
        # Validación adicional si es necesario
        if len(user_input) > 10000:
            return "User input too long (max 10000 characters)"
        
        return None
