# demo/ui/state/host_agent_service.py (SOLUCIÓN EVENT LOOP)

import asyncio
import json
import os
import sys
import traceback
import uuid
from typing import Any

import httpx
from a2a.types import FileWithBytes, Message, Part, Role, Task, TaskState
from service.client.client import ConversationClient
from service.types import (Conversation, CreateConversationRequest, Event,
                           GetEventRequest, ListAgentRequest,
                           ListConversationRequest, ListMessageRequest,
                           ListTaskRequest, MessageInfo, PendingMessageRequest,
                           RegisterAgentRequest, SendMessageRequest)

from .state import (AppState, SessionTask, StateConversation, StateEvent,
                    StateMessage, StateTask)

server_url = 'http://localhost:12000'

# ============================================================================
# HTTPX CLIENT CON MANEJO DE EVENT LOOP
# ============================================================================
class EventLoopAwareHTTPXClient:
    """
    Gestiona clientes httpx por event loop para evitar el error:
    'is bound to a different event loop'
    
    Cada event loop obtiene su propio cliente httpx.
    """
    def __init__(self):
        # Dict: event_loop_id -> httpx.AsyncClient
        self._clients: dict[int, httpx.AsyncClient] = {}
    
    def get_client(self) -> httpx.AsyncClient:
        """
        Obtiene o crea un cliente httpx para el event loop actual.
        """
        try:
            # Obtener el event loop actual
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            
            # Si ya existe un cliente para este loop, reutilizarlo
            if loop_id in self._clients:
                client = self._clients[loop_id]
                # Verificar que el cliente no esté cerrado
                if not client.is_closed:
                    return client
                else:
                    # Cliente cerrado, remover del dict
                    del self._clients[loop_id]
            
            # Crear nuevo cliente para este event loop
            client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20
                )
            )
            
            self._clients[loop_id] = client
            return client
            
        except RuntimeError:
            # No hay event loop corriendo, crear cliente temporal
            # (esto no debería pasar en código async)
            return httpx.AsyncClient(timeout=30.0)
    
    async def close_all(self):
        """Cierra todos los clientes (llamar al finalizar la app)."""
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()

# Instancia global
_httpx_manager = EventLoopAwareHTTPXClient()

def get_httpx_client() -> httpx.AsyncClient:
    """Función helper para obtener el cliente del event loop actual."""
    return _httpx_manager.get_client()

# ============================================================================
# FUNCIONES DE POLLING - CON CLIENTE POR EVENT LOOP
# ============================================================================

async def ListConversations(
    client: ConversationClient | None = None,
) -> list[Conversation]:
    """Lista conversaciones usando el cliente del event loop actual."""
    if client is None:
        httpx_client = get_httpx_client()
        client = ConversationClient(server_url, httpx_client)
    
    try:
        response = await client.list_conversation(ListConversationRequest())
        return response.result if response.result else []
    except Exception as e:
        print('Failed to list conversations: ', e)
    return []


async def GetEvents(
    client: ConversationClient | None = None,
) -> list[Event]:
    """Obtiene eventos usando el cliente del event loop actual."""
    if client is None:
        httpx_client = get_httpx_client()
        client = ConversationClient(server_url, httpx_client)
    
    try:
        response = await client.get_events(GetEventRequest())
        return response.result if response.result else []
    except Exception as e:
        print('Failed to get events', e)
    return []


async def GetProcessingMessages(
    client: ConversationClient | None = None,
):
    """Obtiene mensajes en procesamiento usando el cliente del event loop actual."""
    if client is None:
        httpx_client = get_httpx_client()
        client = ConversationClient(server_url, httpx_client)
    
    try:
        response = await client.get_pending_messages(PendingMessageRequest())
        return dict(response.result) if response.result else {}
    except Exception as e:
        print('Error getting pending messages', e)
        return {}


async def GetTasks(
    client: ConversationClient | None = None,
):
    """Obtiene tareas usando el cliente del event loop actual."""
    if client is None:
        httpx_client = get_httpx_client()
        client = ConversationClient(server_url, httpx_client)
    
    try:
        response = await client.list_tasks(ListTaskRequest())
        return response.result if response.result else []
    except Exception as e:
        print('Failed to list tasks ', e)
        return []


async def ListMessages(
    conversation_id: str, client: ConversationClient | None = None
) -> list[Message]:
    """Lista mensajes usando el cliente del event loop actual."""
    if client is None:
        httpx_client = get_httpx_client()
        client = ConversationClient(server_url, httpx_client)
    
    try:
        response = await client.list_messages(
            ListMessageRequest(params=conversation_id)
        )
        return response.result if response.result else []
    except Exception as e:
        print('Failed to list messages ', e)
    return []


async def UpdateAppState(state: AppState, conversation_id: str):
    """
    Actualiza el estado de la app.
    Usa UN cliente httpx del event loop actual para todas las llamadas.
    """
    # CLAVE: Obtener el cliente del event loop actual
    httpx_client = get_httpx_client()
    client = ConversationClient(server_url, httpx_client)
    
    try:
        if conversation_id:
            state.current_conversation_id = conversation_id
            messages = await ListMessages(conversation_id, client=client)
            if not messages:
                state.messages = []
            else:
                state.messages = [convert_message_to_state(x) for x in messages]
        
        conversations = await ListConversations(client=client)
        if not conversations:
            state.conversations = []
        else:
            state.conversations = [
                convert_conversation_to_state(x) for x in conversations
            ]

        state.task_list = []
        for task in await GetTasks(client=client):
            state.task_list.append(
                SessionTask(
                    context_id=extract_conversation_id(task),
                    task=convert_task_to_state(task),
                )
            )
        
        state.background_tasks = await GetProcessingMessages(client=client)
        state.message_aliases = GetMessageAliases()
    except Exception as e:
        print('Failed to update state: ', e)
        traceback.print_exc(file=sys.stdout)


# ============================================================================
# FUNCIONES PUNTUALES
# ============================================================================

async def SendMessage(message: Message) -> Message | MessageInfo | None:
    """Envía un mensaje (acción puntual)."""
    httpx_client = get_httpx_client()
    client = ConversationClient(server_url, httpx_client)
    try:
        response = await client.send_message(SendMessageRequest(params=message))
        return response.result
    except Exception as e:
        traceback.print_exc()
        print('Failed to send message: ', e)
    return None


async def CreateConversation() -> Conversation:
    """Crea una conversación (acción puntual)."""
    httpx_client = get_httpx_client()
    client = ConversationClient(server_url, httpx_client)
    try:
        response = await client.create_conversation(CreateConversationRequest())
        return (
            response.result
            if response.result
            else Conversation(conversation_id='', is_active=False)
        )
    except Exception as e:
        print('Failed to create conversation', e)
    return Conversation(conversation_id='', is_active=False)


async def ListRemoteAgents():
    """Lista agentes remotos (acción puntual)."""
    httpx_client = get_httpx_client()
    client = ConversationClient(server_url, httpx_client)
    try:
        response = await client.list_agents(ListAgentRequest())
        return response.result
    except Exception as e:
        print('Failed to read agents', e)


async def AddRemoteAgent(path: str):
    """Registra un agente remoto (acción puntual)."""
    httpx_client = get_httpx_client()
    client = ConversationClient(server_url, httpx_client)
    try:
        await client.register_agent(RegisterAgentRequest(params=path))
    except Exception as e:
        print('Failed to register the agent', e)


async def UpdateApiKey(api_key: str):
    """Actualiza la API key (acción puntual)."""
    try:
        os.environ['GOOGLE_API_KEY'] = api_key
        
        httpx_client = get_httpx_client()
        response = await httpx_client.post(
            f'{server_url}/api_key/update', 
            json={'api_key': api_key}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print('Failed to update API key: ', e)
        return False


def GetMessageAliases():
    """Obtiene aliases de mensajes."""
    return {}


# ============================================================================
# FUNCIONES DE CONVERSIÓN (SIN CAMBIOS)
# ============================================================================

def convert_message_to_state(message: Message) -> StateMessage:
    if not message:
        return StateMessage()

    return StateMessage(
        message_id=message.message_id,
        context_id=message.context_id if message.context_id else '',
        task_id=message.task_id if message.task_id else '',
        role=message.role.name,
        content=extract_content(message.parts),
    )


def convert_conversation_to_state(
    conversation: Conversation,
) -> StateConversation:
    return StateConversation(
        conversation_id=conversation.conversation_id,
        conversation_name=conversation.name,
        is_active=conversation.is_active,
        message_ids=[extract_message_id(x) for x in conversation.messages],
    )


def convert_task_to_state(task: Task) -> StateTask:
    output = (
        [extract_content(a.parts) for a in task.artifacts]
        if task.artifacts
        else []
    )
    if not task.history:
        return StateTask(
            task_id=task.id,
            context_id=task.context_id,
            state=TaskState.failed.name,
            message=StateMessage(
                message_id=str(uuid.uuid4()),
                context_id=task.context_id,
                task_id=task.id,
                role=Role.agent.name,
                content=[('No history', 'text')],
            ),
            artifacts=output,
        )
    message = task.history[0]
    last_message = task.history[-1]
    if last_message != message:
        output = [extract_content(last_message.parts)] + output
    return StateTask(
        task_id=task.id,
        context_id=task.context_id,
        state=str(task.status.state),
        message=convert_message_to_state(message),
        artifacts=output,
    )


def convert_event_to_state(event: Event) -> StateEvent:
    return StateEvent(
        context_id=extract_message_conversation(event.content),
        actor=event.actor,
        role=event.content.role.name,
        id=event.id,
        content=extract_content(event.content.parts),
    )


def extract_content(
    message_parts: list[Part],
) -> list[tuple[str | dict[str, Any], str]]:
    parts: list[tuple[str | dict[str, Any], str]] = []
    if not message_parts:
        return []
    for part in message_parts:
        p = part.root
        if p.kind == 'text':
            parts.append((p.text, 'text/plain'))
        elif p.kind == 'file':
            if isinstance(p.file, FileWithBytes):
                parts.append((p.file.bytes, p.file.mime_type or ''))
            else:
                parts.append((p.file.uri, p.file.mime_type or ''))
        elif p.kind == 'data':
            try:
                jsonData = json.dumps(p.data)
                if 'type' in p.data and p.data['type'] == 'form':
                    parts.append((p.data, 'form'))
                else:
                    parts.append((jsonData, 'application/json'))
            except Exception as e:
                print('Failed to dump data', e)
                parts.append(('<data>', 'text/plain'))
    return parts


def extract_message_id(message: Message) -> str:
    return message.message_id


def extract_message_conversation(message: Message) -> str:
    return message.context_id if message.context_id else ''


def extract_conversation_id(task: Task) -> str:
    if task.context_id:
        return task.context_id
    if task.status.message:
        return task.status.message.context_id or ''
    return ''
