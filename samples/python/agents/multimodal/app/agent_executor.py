"""
Physics Agent Executor — Implementación concreta del Template Method (GoF).

Hereda de BaseA2AAgentExecutor y define las operaciones primitivas específicas
del Tutor Socrático Multimodal de Física I.
"""

import logging
from typing import Any, AsyncIterator, List, Dict

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from app.agent import PhysicsMultimodalAgent
from app.base_agent_executor import BaseA2AAgentExecutor
from app.langsmith_config import traceable

logger = logging.getLogger(__name__)


class PhysicsAgentExecutor(BaseA2AAgentExecutor):
    """Executor para el Asistente de Física Multimodal basado en Template Method."""

    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        """Inicializar executor y el agente multimodal subyacente.

        Args:
            qdrant_url: URL de Qdrant.
            qdrant_api_key: API Key de Qdrant.
        """
        self.agent = PhysicsMultimodalAgent(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )

    # ── Operaciones Primitivas (Hooks requeridos por BaseA2AAgentExecutor) ──

    def get_agent_title(self) -> str:
        return "Physics Agent"

    def get_artifact_name(self) -> str:
        return "physics_analysis"

    def get_fallback_query(self, images: List[Dict[str, Any]]) -> str:
        return "Por favor, analiza estas imágenes de física."

    def get_agent_stream(
        self, query: str, context_id: str, images: List[Dict[str, Any]]
    ) -> AsyncIterator[Any]:
        """Invoca el stream de razonamiento socrático y análisis multimodal del agente."""
        return self.agent.stream(query, context_id, images)

    @traceable(
        name="physics_agent_execution",
        run_type="chain",
        tags=["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"],
    )
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Punto de entrada decorado para observabilidad con LangSmith.

        Delega en el Template Method de la clase base BaseA2AAgentExecutor.
        """
        return await super().execute(context, event_queue)
