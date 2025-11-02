import asyncio
import base64
import os
import threading  # <-- Ya no usaremos esto para _send_message, pero lo dejamos por si el manager lo usa
import uuid
from typing import cast

import httpx
from a2a.types import FilePart, FileWithUri, Message, Part
# Importamos BackgroundTasks
from fastapi import BackgroundTasks, FastAPI, Request, Response
from pydantic import BaseModel
from service.types import (CreateConversationResponse, GetEventResponse,
                           ListAgentResponse, ListConversationResponse,
                           ListMessageResponse, ListTaskResponse, MessageInfo,
                           PendingMessageResponse, RegisterAgentResponse,
                           SendMessageResponse)

from .adk_host_manager import ADKHostManager, get_message_id
from .application_manager import ApplicationManager
from .in_memory_manager import InMemoryFakeAgentManager

# --- MODELOS PYDANTIC PARA LOS BODIES ---

# Modelo para /message/send
class SendMessageBody(BaseModel):
    params: Message

# Modelo para /message/list
class ListMessagesBody(BaseModel):
    params: str  # Espera el conversation_id como un string

# Modelo para /agent/register
class RegisterAgentBody(BaseModel):
    params: str  # Espera la URL del agente como un string

# Modelo para /api_key/update
class UpdateApiKeyBody(BaseModel):
    api_key: str

# --- FIN DE MODELOS ---


class ConversationServer:
    """ConversationServer is the backend to serve the agent interactions in the UI

    This defines the interface that is used by the Mesop system to interact with
    agents and provide details about the executions.
    """

    def __init__(self, app: FastAPI, http_client: httpx.AsyncClient):
        agent_manager = os.environ.get('A2A_HOST', 'ADK')
        self.manager: ApplicationManager

        # Get API key from environment
        api_key = os.environ.get('GOOGLE_API_KEY', '')
        uses_vertex_ai = (
            os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
        )

        if agent_manager.upper() == 'ADK':
            self.manager = ADKHostManager(
                http_client,
                api_key=api_key,
                uses_vertex_ai=uses_vertex_ai,
            )
        else:
            self.manager = InMemoryFakeAgentManager()
        self._file_cache = {}  # dict[str, FilePart] maps file id to message data
        self._message_to_cache = {}  # dict[str, str] maps message id to cache id

        app.add_api_route(
            '/conversation/create', self._create_conversation, methods=['POST']
        )
        app.add_api_route(
            '/conversation/list', self._list_conversation, methods=['POST']
        )
        app.add_api_route('/message/send', self._send_message, methods=['POST'])
        app.add_api_route('/events/get', self._get_events, methods=['POST'])
        app.add_api_route(
            '/message/list', self._list_messages, methods=['POST']
        )
        app.add_api_route(
            '/message/pending', self._pending_messages, methods=['POST']
        )
        app.add_api_route('/task/list', self._list_tasks, methods=['POST'])
        app.add_api_route(
            '/agent/register', self._register_agent, methods=['POST']
        )
        app.add_api_route('/agent/list', self._list_agents, methods=['POST'])
        app.add_api_route(
            '/message/file/{file_id}', self._files, methods=['GET']
        )
        app.add_api_route(
            '/api_key/update', self._update_api_key, methods=['POST']
        )

    # Update API key in manager
    def update_api_key(self, api_key: str):
        if isinstance(self.manager, ADKHostManager):
            self.manager.update_api_key(api_key)

    async def _create_conversation(self):
        c = await self.manager.create_conversation()
        return CreateConversationResponse(result=c)

    # --- MODIFICACIÓN CLAVE ---
    # Usamos BackgroundTasks en lugar de threading manual
    async def _send_message(
        self, body: SendMessageBody, background_tasks: BackgroundTasks
    ):
        message = body.params
        message = self.manager.sanitize_message(message)

        # Añadimos la tarea pesada (procesar el mensaje) al fondo.
        # FastAPI se encargará de ejecutarla después de enviar la respuesta.
        # Esto funciona para AMBOS managers (ADK y Fake) porque ambos
        # tienen un método "async def process_message"
        background_tasks.add_task(self.manager.process_message, message)

        # La API responde INMEDIATAMENTE con el ID del mensaje.
        # Ya no hay hilos, ni bucles de eventos manuales, ni bloqueos.
        return SendMessageResponse(
            result=MessageInfo(
                message_id=message.message_id,
                context_id=message.context_id if message.context_id else '',
            )
        )

    # Modificado para usar Pydantic Body
    async def _list_messages(self, body: ListMessagesBody):
        conversation_id = body.params
        conversation = self.manager.get_conversation(conversation_id)
        if conversation:
            return ListMessageResponse(
                result=self.cache_content(conversation.messages)
            )
        return ListMessageResponse(result=[])

    def cache_content(self, messages: list[Message]):
        rval = []
        for m in messages:
            message_id = get_message_id(m)
            if not message_id:
                rval.append(m)
                continue
            new_parts: list[Part] = []
            for i, p in enumerate(m.parts):
                part = p.root
                if part.kind != 'file':
                    new_parts.append(p)
                    continue
                message_part_id = f'{message_id}:{i}'
                if message_part_id in self._message_to_cache:
                    cache_id = self._message_to_cache[message_part_id]
                else:
                    cache_id = str(uuid.uuid4())
                    self._message_to_cache[message_part_id] = cache_id
                # Replace the part data with a url reference
                new_parts.append(
                    Part(
                        root=FilePart(
                            file=FileWithUri(
                                mime_type=part.file.mime_type,
                                uri=f'/message/file/{cache_id}',
                            )
                        )
                    )
                )
                if cache_id not in self._file_cache:
                    self._file_cache[cache_id] = part
            m.parts = new_parts
            rval.append(m)
        return rval

    async def _pending_messages(self):
        return PendingMessageResponse(
            result=self.manager.get_pending_messages()
        )

    def _list_conversation(self):
        return ListConversationResponse(result=self.manager.conversations)

    def _get_events(self):
        return GetEventResponse(result=self.manager.events)

    def _list_tasks(self):
        return ListTaskResponse(result=self.manager.tasks)

    # Modificado para usar Pydantic Body
    async def _register_agent(self, body: RegisterAgentBody):
        url = body.params
        self.manager.register_agent(url)
        return RegisterAgentResponse()

    async def _list_agents(self):
        return ListAgentResponse(result=self.manager.agents)

    # Modificado para añadir type hint para Swagger
    def _files(self, file_id: str):
        if file_id not in self._file_cache:
            raise Exception('file not found')
        part = self._file_cache[file_id]
        if 'image' in part.file.mime_type:
            return Response(
                content=base64.b64decode(part.file.bytes),
                media_type=part.file.mime_type,
            )
        return Response(content=part.file.bytes, media_type=part.file.mime_type)

    # Modificado para usar Pydantic Body
    async def _update_api_key(self, body: UpdateApiKeyBody):
        """Update the API key"""
        try:
            api_key = body.api_key
            if api_key:
                # Update in the manager
                self.update_api_key(api_key)
                return {'status': 'success'}
            return {'status': 'error', 'message': 'No API key provided'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
