# demo/ui/service/client/client.py (SOLUCIÓN)

import json
from typing import Any

import httpx
from service.types import (AgentClientHTTPError, AgentClientJSONError,
                           CreateConversationRequest,
                           CreateConversationResponse, GetEventRequest,
                           GetEventResponse, JSONRPCRequest, ListAgentRequest,
                           ListAgentResponse, ListConversationRequest,
                           ListConversationResponse, ListMessageRequest,
                           ListMessageResponse, ListTaskRequest,
                           ListTaskResponse, PendingMessageRequest,
                           PendingMessageResponse, RegisterAgentRequest,
                           RegisterAgentResponse, SendMessageRequest,
                           SendMessageResponse)


class ConversationClient:
    """
    Cliente para comunicarse con el servidor A2A.
    
    IMPORTANTE: Acepta un httpx.AsyncClient externo para evitar crear
    múltiples clientes y acumular conexiones abiertas.
    """
    
    def __init__(self, base_url: str, httpx_client: httpx.AsyncClient | None = None):
        """
        Inicializa el cliente.
        
        Args:
            base_url: URL base del servidor
            httpx_client: Cliente httpx reutilizable (RECOMENDADO para polling)
                         Si es None, se crea uno nuevo por request (solo para acciones puntuales)
        """
        self.base_url = base_url.rstrip('/')
        self._external_client = httpx_client
        self._owns_client = httpx_client is None
    
    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        """
        Envía una solicitud JSON-RPC al servidor.
        
        Si se proporcionó un cliente externo, lo usa.
        Si no, crea uno temporal (NO RECOMENDADO para polling frecuente).
        """
        if self._external_client is not None:
            # CASO ÓPTIMO: Usar cliente externo (reutilizable)
            return await self._send_with_client(self._external_client, request)
        else:
            # CASO TEMPORAL: Crear cliente nuevo (solo para acciones puntuales)
            async with httpx.AsyncClient(timeout=30.0) as client:
                return await self._send_with_client(client, request)
    
    async def _send_with_client(
        self, 
        client: httpx.AsyncClient, 
        request: JSONRPCRequest
    ) -> dict[str, Any]:
        """Envía la solicitud usando el cliente proporcionado."""
        try:
            response = await client.post(
                self.base_url + '/' + request.method,
                json=request.model_dump(mode='json', exclude_none=True),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print('http error', e)
            raise AgentClientHTTPError(
                e.response.status_code, str(e)
            ) from e
        except json.JSONDecodeError as e:
            print('decode error', e)
            raise AgentClientJSONError(str(e)) from e
    
    # ============================================================================
    # MÉTODOS DE LA API
    # ============================================================================
    
    async def send_message(
        self, payload: SendMessageRequest
    ) -> SendMessageResponse:
        return SendMessageResponse(**await self._send_request(payload))

    async def create_conversation(
        self, payload: CreateConversationRequest
    ) -> CreateConversationResponse:
        return CreateConversationResponse(**await self._send_request(payload))

    async def list_conversation(
        self, payload: ListConversationRequest
    ) -> ListConversationResponse:
        return ListConversationResponse(**await self._send_request(payload))

    async def get_events(self, payload: GetEventRequest) -> GetEventResponse:
        return GetEventResponse(**await self._send_request(payload))

    async def list_messages(
        self, payload: ListMessageRequest
    ) -> ListMessageResponse:
        return ListMessageResponse(**await self._send_request(payload))

    async def get_pending_messages(
        self, payload: PendingMessageRequest
    ) -> PendingMessageResponse:
        return PendingMessageResponse(**await self._send_request(payload))

    async def list_tasks(self, payload: ListTaskRequest) -> ListTaskResponse:
        return ListTaskResponse(**await self._send_request(payload))

    async def register_agent(
        self, payload: RegisterAgentRequest
    ) -> RegisterAgentResponse:
        return RegisterAgentResponse(**await self._send_request(payload))

    async def list_agents(self, payload: ListAgentRequest) -> ListAgentResponse:
        return ListAgentResponse(**await self._send_request(payload))
