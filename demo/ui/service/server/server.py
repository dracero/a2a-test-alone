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

from .adk_host_manager import ADKHostManager, get_message_id
from .application_manager import ApplicationManager
from .in_memory_manager import InMemoryFakeAgentManager

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

class RegisterAgentBody(BaseModel):
    params: str

class UpdateApiKeyBody(BaseModel):
    api_key: str

# --- FIN DE MODELOS ---

class ConversationServer:
    """ConversationServer is the backend to serve the agent interactions in the UI"""

    def __init__(self, app: FastAPI, http_client: httpx.AsyncClient):
        agent_manager = os.environ.get('A2A_HOST', 'ADK')
        self.manager: ApplicationManager

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
        app.add_api_route('/agent/list', self._list_agents, methods=['POST'])
        app.add_api_route(
            '/message/file/{file_id}', self._files, methods=['GET']
        )
        app.add_api_route(
            '/api_key/update', self._update_api_key, methods=['POST']
        )

    def update_api_key(self, api_key: str):
        if isinstance(self.manager, ADKHostManager):
            self.manager.update_api_key(api_key)

    async def _create_conversation(self):
        c = await self.manager.create_conversation()
        return CreateConversationResponse(result=c)

    def parse_message_from_dict(self, data: dict[str, Any]) -> Message:
        """
        🔧 NUEVO: Parsea un diccionario del frontend a un objeto Message válido.
        Maneja los diferentes formatos de Part que puede enviar el frontend.
        """
        print(f"\n{'='*60}")
        print(f"🔍 PARSING MESSAGE FROM FRONTEND")
        print(f"Raw data keys: {data.keys()}")
        print(f"{'='*60}\n")
        
        parts: list[Part] = []
        
        for i, part_data in enumerate(data.get('parts', [])):
            print(f"📦 Part {i}: {part_data}")
            
            kind = part_data.get('kind')
            
            if kind == 'text':
                # Texto simple
                parts.append(Part(root=TextPart(text=part_data.get('text', ''))))
                print(f"  ✅ Text part added")
                
            elif kind == 'file':
                # Archivo (imagen u otro)
                file_data = part_data.get('file', {})
                mime_type = file_data.get('mime_type', 'application/octet-stream')
                
                if 'bytes' in file_data:
                    # FileWithBytes (nuevo del frontend)
                    bytes_data = file_data['bytes']
                    print(f"  ✅ File part (bytes) added: {mime_type}, {len(bytes_data)} chars")
                    parts.append(
                        Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=bytes_data,
                                    mime_type=mime_type,
                                    name=file_data.get('name')
                                )
                            )
                        )
                    )
                elif 'uri' in file_data:
                    # FileWithUri (del cache)
                    print(f"  ✅ File part (URI) added: {file_data['uri']}")
                    parts.append(
                        Part(
                            root=FilePart(
                                file=FileWithUri(
                                    uri=file_data['uri'],
                                    mime_type=mime_type
                                )
                            )
                        )
                    )
                else:
                    print(f"  ⚠️ File part without bytes or URI")
            else:
                print(f"  ⚠️ Unknown part kind: {kind}")
        
        # 🔧 CORRECCIÓN: Normalizar el role a string sin 'Role.' prefix
        role_value = data.get('role', 'user')
        if isinstance(role_value, str):
            # Limpiar si viene como 'Role.user' o 'user'
            role_value = role_value.replace('Role.', '').lower()
        
        # Construir el mensaje usando model_validate para que Pydantic maneje la conversión
        message_dict = {
            'message_id': data.get('message_id', str(uuid.uuid4())),
            'context_id': data.get('context_id', ''),
            'role': role_value,
            'parts': parts,
        }
        
        # Agregar campos opcionales solo si existen
        if 'recipient' in data:
            message_dict['recipient'] = data['recipient']
        if 'metadata' in data:
            message_dict['metadata'] = data['metadata']
        
        message = Message(**message_dict)
        
        print(f"\n✅ Message parsed successfully:")
        print(f"  • message_id: {message.message_id}")
        print(f"  • context_id: {message.context_id}")
        print(f"  • role: {message.role}")
        print(f"  • parts count: {len(message.parts)}")
        print(f"{'='*60}\n")
        
        return message

    def restore_files_from_cache(self, message: Message) -> Message:
        """
        🔧 CORREGIDO: Maneja tanto archivos nuevos (FileWithBytes) como cacheados (FileWithUri).
        - FileWithBytes (nuevo del frontend): Se mantiene tal cual
        - FileWithUri (del cache): Se restaura desde el cache
        """
        message_copy = copy.deepcopy(message)
        restored_parts: list[Part] = []
        
        for i, part in enumerate(message_copy.parts):
            p = part.root
            
            # Si NO es un archivo, mantenerlo tal cual
            if p.kind != 'file':
                restored_parts.append(part)
                continue
            
            # Si es FileWithBytes (mensaje nuevo del frontend), mantenerlo
            if isinstance(p.file, FileWithBytes):
                print(f"✅ Keeping new FileWithBytes: {p.file.mime_type}, {len(p.file.bytes)} chars")
                restored_parts.append(part)
                continue
            
            # Si es FileWithUri (mensaje del cache), restaurar desde cache
            if isinstance(p.file, FileWithUri):
                # Extraer el cache_id de la URI (formato: /message/file/{cache_id})
                uri_parts = p.file.uri.split('/')
                if len(uri_parts) >= 3 and uri_parts[-2] == 'file':
                    cache_id = uri_parts[-1]
                    
                    # Buscar en el cache
                    if cache_id in self._file_cache:
                        cached_part = self._file_cache[cache_id]
                        print(f"✅ Restored file from cache: {cache_id}")
                        restored_parts.append(Part(root=cached_part))
                        continue
                    else:
                        print(f"⚠️ Cache ID not found: {cache_id}")
                
                # Si no se pudo restaurar, mantener la URI
                print(f"⚠️ Keeping FileWithUri (not in cache): {p.file.uri}")
                restored_parts.append(part)
                continue
        
        message_copy.parts = restored_parts
        return message_copy

    async def _send_message(
        self, body: SendMessageBody, background_tasks: BackgroundTasks
    ):
        # 🔧 CORRECCIÓN CRÍTICA: Parsear el dict a Message manualmente
        message = self.parse_message_from_dict(body.params)
        
        message = self.manager.sanitize_message(message)
        
        # Restaurar archivos desde cache (solo para URIs)
        # Los FileWithBytes nuevos se mantienen intactos
        message = self.restore_files_from_cache(message)
        
        print(f"\n{'='*60}")
        print(f"📤 SEND MESSAGE - Parts after restore:")
        for i, part in enumerate(message.parts):
            p = part.root
            if p.kind == 'file':
                if isinstance(p.file, FileWithBytes):
                    print(f"  Part {i}: FileWithBytes - {p.file.mime_type}, {len(p.file.bytes)} chars base64")
                elif isinstance(p.file, FileWithUri):
                    print(f"  Part {i}: FileWithUri - {p.file.uri}")
            elif p.kind == 'text':
                print(f"  Part {i}: Text - {p.text[:50]}...")
            else:
                print(f"  Part {i}: {p.kind}")
        print(f"{'='*60}\n")
        
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
                        print(f"💾 File cached: {cache_id} ({part.file.mime_type})")

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
