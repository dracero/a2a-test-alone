import asyncio
import base64
import copy
import os
import uuid
from typing import Any, cast

import httpx
from a2a.types import (FilePart, FileWithBytes, FileWithUri, Message, Part,
                       TextPart)
from fastapi import BackgroundTasks, FastAPI, Request, Response
from pydantic import BaseModel, field_validator
from service.types import (CreateConversationResponse, GetEventResponse,
                           ListAgentResponse, ListConversationResponse,
                           ListMessageResponse, ListTaskResponse, MessageInfo,
                           PendingMessageResponse, RegisterAgentResponse,
                           SendMessageResponse)

from .adk_host_manager import ADKHostManager
from .application_manager import ApplicationManager
from .in_memory_manager import InMemoryFakeAgentManager
from .message_utils import get_message_id

# --- MODELOS PYDANTIC PARA LOS BODIES ---

class SendMessageBody(BaseModel):
    params: dict[str, Any]  # Cambiado a dict para parseo manual
    
    @field_validator('params', mode='before')
    @classmethod
    def parse_params(cls, v: Any) -> dict[str, Any]:
        """Validador que acepta el dict tal cual viene del frontend"""
        if isinstance(v, dict):
            return v
        return v

class ListMessagesBody(BaseModel):
    params: str
    
    class Config:
        extra = 'allow'  # Permitir campos adicionales

class RegisterAgentBody(BaseModel):
    params: str

class UpdateApiKeyBody(BaseModel):
    api_key: str

class UpdateConversationBody(BaseModel):
    conversation_id: str
    name: str

class CorrectBody(BaseModel):
    student_id: str
    tema: str
    correccion: str

class NamsConclusionsBody(BaseModel):
    student_id: str | None = None
    conversation_id: str | None = None

# --- FIN DE MODELOS ---

class ConversationServer:
    """ConversationServer is the backend to serve the agent interactions in the UI"""

    def __init__(self, app: FastAPI, http_client: httpx.AsyncClient):
        agent_manager = os.environ.get('A2A_HOST', 'ADK')
        self.manager: ApplicationManager

        # Use GROQ_API_KEY for BeeAI, Google key rotator for ADK
        if agent_manager.upper() == 'BEEAI':
            api_key = os.environ.get('GROQ_API_KEY', '')
        else:
            from .api_key_rotator import google_key_rotator
            api_key = google_key_rotator.get_key()
        uses_vertex_ai = (
            os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
        )

        if agent_manager.upper() == 'ADK':
            self.manager = ADKHostManager(
                http_client,
                api_key=api_key,
                uses_vertex_ai=uses_vertex_ai,
            )
        elif agent_manager.upper() == 'BEEAI':
            from .beeai_host_manager import BeeAIHostManager
            self.manager = BeeAIHostManager(
                http_client,
                api_key=api_key,
                uses_vertex_ai=uses_vertex_ai,
            )
        else:
            self.manager = InMemoryFakeAgentManager()

        self._file_cache = {}
        self._message_to_cache = {}

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
        app.add_api_route(
            '/agent/register/manual', self._register_agent_manual, methods=['POST']
        )
        app.add_api_route('/agent/list', self._list_agents, methods=['POST'])
        app.add_api_route(
            '/message/file/{file_id}', self._files, methods=['GET']
        )
        app.add_api_route(
            '/api_key/update', self._update_api_key, methods=['POST']
        )
        app.add_api_route(
            '/nams/conclusions', self._nams_conclusions, methods=['POST']
        )
        app.add_api_route(
            '/conversation/update', self._update_conversation, methods=['POST']
        )
        app.add_api_route(
            '/correct', self._correct_agent, methods=['POST']
        )

    def update_api_key(self, api_key: str):
        if isinstance(self.manager, ADKHostManager):
            self.manager.update_api_key(api_key)

    async def _create_conversation(self):
        c = await self.manager.create_conversation()
        return CreateConversationResponse(result=c)

    def parse_message_from_dict(self, data: dict[str, Any]) -> Message:
        """Parsea un diccionario del frontend a un objeto Message válido."""
        parts: list[Part] = []
        
        for part_data in data.get('parts', []):
            kind = part_data.get('kind')
            
            if kind == 'text':
                parts.append(Part(root=TextPart(text=part_data.get('text', ''))))
                
            elif kind == 'file':
                file_data = part_data.get('file', {})
                mime_type = file_data.get('mime_type', 'application/octet-stream')
                
                if 'bytes' in file_data:
                    parts.append(
                        Part(root=FilePart(
                            file=FileWithBytes(
                                bytes=file_data['bytes'],
                                mime_type=mime_type,
                                name=file_data.get('name')
                            )
                        ))
                    )
                elif 'uri' in file_data:
                    parts.append(
                        Part(root=FilePart(
                            file=FileWithUri(uri=file_data['uri'], mime_type=mime_type)
                        ))
                    )
        
        role_value = data.get('role', 'user')
        if isinstance(role_value, str):
            role_value = role_value.replace('Role.', '').lower()
        
        context_id_value = data.get('context_id', '')
        if isinstance(context_id_value, dict):
            context_id_value = context_id_value.get('id', '') or context_id_value.get('conversation_id', '') or ''
        elif not isinstance(context_id_value, str):
            context_id_value = str(context_id_value) if context_id_value else ''
        
        message_dict = {
            'message_id': data.get('message_id', str(uuid.uuid4())),
            'context_id': context_id_value,
            'role': role_value,
            'parts': parts,
        }
        if 'recipient' in data:
            message_dict['recipient'] = data['recipient']
        if 'metadata' in data:
            message_dict['metadata'] = data['metadata']
        
        return Message(**message_dict)

    def restore_files_from_cache(self, message: Message) -> Message:
        """Restaura FileWithBytes desde cache para FileWithUri, mantiene FileWithBytes intactos."""
        message_copy = copy.deepcopy(message)
        restored_parts: list[Part] = []
        
        for part in message_copy.parts:
            p = part.root
            if p.kind != 'file':
                restored_parts.append(part)
                continue
            
            if isinstance(p.file, FileWithBytes):
                restored_parts.append(part)
                continue
            
            if isinstance(p.file, FileWithUri):
                uri_parts = p.file.uri.split('/')
                if len(uri_parts) >= 3 and uri_parts[-2] == 'file':
                    cache_id = uri_parts[-1]
                    if cache_id in self._file_cache:
                        restored_parts.append(Part(root=self._file_cache[cache_id]))
                        continue
                restored_parts.append(part)
        
        message_copy.parts = restored_parts
        return message_copy

    async def _send_message(
        self, body: SendMessageBody, background_tasks: BackgroundTasks
    ):
        message = self.parse_message_from_dict(body.params)
        message = self.manager.sanitize_message(message)
        message = self.restore_files_from_cache(message)
        background_tasks.add_task(self.manager.process_message, message)
        return SendMessageResponse(
            result=MessageInfo(
                message_id=message.message_id,
                context_id=message.context_id if message.context_id else '',
            )
        )

    async def _list_messages(self, body: ListMessagesBody):
        conversation_id = body.params
        conversation = self.manager.get_conversation(conversation_id)
        if conversation:
            return ListMessageResponse(
                result=self.cache_content(conversation.messages)
            )
        return ListMessageResponse(result=[])

    def cache_content(self, messages: list[Message]) -> list[Message]:
        """
        Hace una copia profunda de los mensajes y reemplaza FileWithBytes
        con FileWithUri para la UI. Los archivos originales se guardan en cache.
        """
        rval = []

        for m in messages:
            # ✅ COPIA PROFUNDA del mensaje completo
            message_copy = copy.deepcopy(m)
            message_id = get_message_id(message_copy)

            if not message_id:
                rval.append(message_copy)
                continue

            new_parts: list[Part] = []
            for i, p in enumerate(message_copy.parts):
                part = p.root
                if part.kind != 'file':
                    new_parts.append(p)
                    continue

                message_part_id = f'{message_id}:{i}'

                # Verificar si ya tenemos este archivo cacheado
                if message_part_id in self._message_to_cache:
                    cache_id = self._message_to_cache[message_part_id]
                else:
                    cache_id = str(uuid.uuid4())
                    self._message_to_cache[message_part_id] = cache_id
                    # Solo cachear si no existe
                    if cache_id not in self._file_cache:
                        self._file_cache[cache_id] = part

                # Reemplazar con URI para la UI
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

            # Asignar las partes modificadas a la COPIA
            message_copy.parts = new_parts
            rval.append(message_copy)

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

    async def _register_agent(self, body: RegisterAgentBody):
        """Register a new agent"""
        url = body.params
        self.manager.register_agent(url)
        return RegisterAgentResponse()
    
    async def _register_agent_manual(self, request: Request):
        """Manually register an agent by URL (for debugging)"""
        try:
            data = await request.json()
            url = data.get('url')
            if not url:
                return {'status': 'error', 'message': 'URL is required'}
            
            self.manager.register_agent(url)
            return {'status': 'success', 'message': f'Agent registered: {url}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def _list_agents(self):
        """List all registered agents"""
        return ListAgentResponse(result=self.manager.agents)

    def _files(self, file_id: str):
        """Serve cached files"""
        if file_id not in self._file_cache:
            raise Exception('file not found')
        part = self._file_cache[file_id]
        if 'image' in part.file.mime_type:
            return Response(
                content=base64.b64decode(part.file.bytes),
                media_type=part.file.mime_type,
            )
        return Response(content=part.file.bytes, media_type=part.file.mime_type)

    async def _update_api_key(self, body: UpdateApiKeyBody):
        """Update the API key"""
        try:
            api_key = body.api_key
            if api_key:
                self.update_api_key(api_key)
                return {'status': 'success'}
            return {'status': 'error', 'message': 'No API key provided'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def _nams_conclusions(self, body: NamsConclusionsBody):
        """Fetch all user preferences/conclusions and deficiencies from NAMS for a student"""
        if not hasattr(self.manager, 'neo4j_memory') or not self.manager.neo4j_memory:
            return {'status': 'inactive', 'conclusions': [], 'deficiencies': []}
            
        try:
            await self.manager._ensure_neo4j_connected()
            if getattr(self.manager, '_neo4j_connected', False):
                student_id = body.student_id
                if not student_id and body.conversation_id:
                    conv = self.manager.get_conversation(body.conversation_id)
                    if conv:
                        student_id = conv.name or body.conversation_id
                    else:
                        student_id = body.conversation_id
                
                if not student_id:
                    return {'status': 'active', 'conclusions': [], 'deficiencies': []}

                print(f"🔍 Fetching preferences and deficiencies for user_identifier '{student_id}'")
                prefs = await self.manager.neo4j_memory.long_term.get_preferences_for(student_id)
                
                conclusions = []
                deficiencies = []
                for p in prefs:
                    pref_str = p.preference if hasattr(p, 'preference') else (p.get('preference', str(p)) if isinstance(p, dict) else str(p))
                    cat = p.category if hasattr(p, 'category') else (p.get('category', '') if isinstance(p, dict) else '')
                    
                    if cat == "falencia":
                        deficiencies.append(pref_str)
                    else:
                        conclusions.append(pref_str)
                        
                return {
                    'status': 'active', 
                    'conclusions': conclusions, 
                    'deficiencies': deficiencies
                }
            else:
                return {'status': 'inactive', 'conclusions': [], 'deficiencies': []}
        except Exception as e:
            print(f"Error fetching NAMS conclusions: {e}")
            return {'status': 'error', 'message': str(e), 'conclusions': [], 'deficiencies': []}

    async def _update_conversation(self, body: UpdateConversationBody):
        conversation = self.manager.get_conversation(body.conversation_id)
        if conversation:
            conversation.name = body.name
            if hasattr(self.manager, '_save_conversations'):
                self.manager._save_conversations()
            return {'status': 'success'}
        return {'status': 'error', 'message': 'Conversation not found'}

    async def _correct_agent(self, body: CorrectBody, request: Request):
        role = request.headers.get("X-Role", "alumno")
        if role != "profesor":
            return Response(status_code=403, content="Only professors can submit corrections.")
            
        if hasattr(self.manager, 'add_deficiency'):
            success = await self.manager.add_deficiency(
                student_id=body.student_id,
                tema=body.tema,
                correccion=body.correccion
            )
            if success:
                return {'status': 'success'}
            else:
                return {'status': 'error', 'message': 'Failed to save deficiency in database'}
        return {'status': 'error', 'message': 'Manager does not support deficiencies'}
