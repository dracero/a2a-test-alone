"""Agent executor for A2A protocol with LangSmith monitoring."""

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
            # Execute agent with tracing
            result = await self._execute_agent_with_tracing(
                query, 
                context.context_id,
                context.task_id
            )
            
            print(f'✅ Agent execution completed')
            logger.info(f'Final Result: {result}')
            
            # Get image data
            data = self.agent.get_image_data(
                session_id=context.context_id, 
                image_key=result.raw
            )
            
            # Prepare response parts
            if data and not data.error:
                parts = [
                    FilePart(
                        file=FileWithBytes(
                            bytes=data.bytes,
                            mime_type=data.mime_type,
                            name=data.id,
                        )
                    )
                ]
                
                # Log success with image details
                if LANGSMITH_ENABLED:
                    try:
                        langsmith_client.create_feedback(
                            run_id=None,
                            key="a2a_request_success",
                            value={
                                "task_id": context.task_id,
                                "context_id": context.context_id,
                                "image_id": data.id,
                                "mime_type": data.mime_type,
                                "query": query[:200]
                            }
                        )
                    except:
                        pass
                
                print(f'🖼️ Image ready: {data.id}')
            else:
                error_msg = data.error if data else 'failed to generate image'
                parts = [
                    Part(
                        root=TextPart(text=error_msg)
                    )
                ]
                
                # Log failure
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
                
                logger.error(f'Image generation failed: {error_msg}')
                print(f'❌ Generation failed: {error_msg}')
            
            # Send completion event
            await event_queue.enqueue_event(
                completed_task(
                    context.task_id,
                    context.context_id,
                    [new_artifact(parts, f'image_{context.task_id}')],
                    [context.message],
                )
            )
            
            print(f'📤 Response sent to client')
            
        except Exception as e:
            error_msg = f'Error invoking agent: {e}'
            logger.error(error_msg)
            print(f'❌ Execution error: {error_msg}')
            
            # Log exception to LangSmith
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
            
            raise ServerError(error=ValueError(error_msg)) from e

    @traceable(name="agent_invocation", run_type="chain")
    async def _execute_agent_with_tracing(
        self, 
        query: str, 
        context_id: str,
        task_id: str
    ):
        """Execute agent with detailed tracing.
        
        Args:
            query: User query/prompt
            context_id: Session/context identifier
            task_id: Task identifier
            
        Returns:
            Agent execution result
        """
        # Log agent invocation details
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="agent_invocation_start",
                    value={
                        "task_id": task_id,
                        "context_id": context_id,
                        "query_length": len(query),
                        "query_preview": query[:100]
                    }
                )
            except:
                pass
        
        try:
            # Execute the agent
            result = self.agent.invoke(query, context_id)
            
            # Log successful invocation
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="agent_invocation_success",
                        value={
                            "task_id": task_id,
                            "context_id": context_id,
                            "result_preview": str(result)[:200]
                        }
                    )
                except:
                    pass
            
            return result
            
        except Exception as e:
            # Log invocation failure
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="agent_invocation_failure",
                        value={
                            "task_id": task_id,
                            "context_id": context_id,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                except:
                    pass
            
            raise

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """Cancel operation (not supported).
        
        Args:
            request: Request context
            event_queue: Event queue
            
        Raises:
            ServerError: Always raises UnsupportedOperationError
        """
        # Log cancellation attempt
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="cancel_attempt",
                    value={
                        "task_id": request.task_id,
                        "context_id": request.context_id,
                        "result": "unsupported"
                    }
                )
            except:
                pass
        
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate incoming request.
        
        Args:
            context: Request context to validate
            
        Returns:
            True if validation fails, False if validation passes
        """
        # Add validation logic here if needed
        # Return True to indicate error, False to indicate success
        
        # Log validation
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="request_validation",
                    value={
                        "task_id": context.task_id,
                        "context_id": context.context_id,
                        "has_user_input": bool(context.get_user_input()),
                        "validation_passed": True
                    }
                )
            except:
                pass
        
        return False
    
    def get_metrics(self, context_id: str) -> dict:
        """Get execution metrics for a context/session.
        
        Args:
            context_id: Session identifier
            
        Returns:
            Dictionary with execution metrics
        """
        try:
            from in_memory_cache import InMemoryCache
            cache = InMemoryCache()
            session_data = cache.get(context_id)
            
            metrics = {
                "context_id": context_id,
                "total_images": len(session_data) if session_data else 0,
                "langsmith_enabled": LANGSMITH_ENABLED
            }
            
            # Log metrics retrieval
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="metrics_retrieved",
                        value=metrics
                    )
                except:
                    pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                "context_id": context_id,
                "error": str(e)
            }
