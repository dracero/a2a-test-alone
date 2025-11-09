import asyncio
import base64
import json
import os
import uuid

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (AgentCard, DataPart, FilePart, FileWithBytes, Message,
                       Part, Role, Task, TaskState, TextPart,
                       TransportProtocol)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback
from timestamp_ext import TimestampExtension


class HostAgent:
    """The host agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.httpx_client = http_client
        self.timestamp_extension = TimestampExtension()
        # 🔧 NUEVO: Cache para guardar los archivos del mensaje original
        self._user_message_files: dict[str, list[Part]] = {}
        # Crear el cliente httpx con timeout personalizado
        httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0)  # 300 segundos
        )

        # Usar ese cliente en tu configuración
        config = ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[
                TransportProtocol.jsonrpc,
                TransportProtocol.http_json,
            ]
)
        client_factory = ClientFactory(config)
        client_factory = self.timestamp_extension.wrap_client_factory(
            client_factory
        )
        self.client_factory = client_factory
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''
        loop = asyncio.get_running_loop()
        loop.create_task(
            self.init_remote_agent_addresses(remote_agent_addresses)
        )

    async def init_remote_agent_addresses(
        self, remote_agent_addresses: list[str]
    ):
        async with asyncio.TaskGroup() as task_group:
            for address in remote_agent_addresses:
                task_group.create_task(self.retrieve_card(address))

    async def retrieve_card(self, address: str):
        card_resolver = A2ACardResolver(self.httpx_client, address)
        card = await card_resolver.get_agent_card()
        self.register_agent_card(card)

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(self.client_factory, card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        LITELLM_MODEL = os.getenv(
            'LITELLM_MODEL', 'gemini/gemini-2.5-flash'
        )
        return Agent(
            model=LiteLlm(model=LITELLM_MODEL),
            name='host_agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                'This agent orchestrates the decomposition of the user request into'
                ' tasks that can be performed by the child agents.'
            ),
            tools=[
                self.list_remote_agents,
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_state(context)
        return f"""You are an expert delegator that can delegate the user request to the
appropriate remote agents.

Discovery:
- You can use `list_remote_agents` to list the available remote agents you
can use to delegate the task.

Execution:
- For actionable requests, you can use `send_message` to interact with remote agents to take action.

Be sure to include the remote agent name when you respond to the user.

Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

Agents:
{self.agents}

Current agent: {current_agent['active_agent']}
"""

    def check_state(self, context: ReadonlyContext):
        state = context.state
        if (
            'context_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'agent' in state
        ):
            return {'active_agent': f'{state["agent"]}'}
        return {'active_agent': 'None'}

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request
    ):
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            state['session_active'] = True

        # 🔧 NUEVO: Guardar archivos del mensaje original
        message_id = state.get('message_id')
        if message_id and llm_request.contents:
            file_parts = []
            for content in llm_request.contents:
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        # Buscar partes con inline_data (imágenes/archivos)
                        if hasattr(part, 'inline_data') and part.inline_data:
                            print(f"🔍 HOST_AGENT before_model_callback: Archivo detectado")
                            print(f"   • mime_type: {part.inline_data.mime_type}")
                            print(f"   • data length: {len(part.inline_data.data)} bytes")
                            
                            # Convertir a base64 para guardar
                            base64_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                            
                            file_parts.append(
                                Part(
                                    root=FilePart(
                                        file=FileWithBytes(
                                            bytes=base64_data,
                                            mime_type=part.inline_data.mime_type,
                                            name=f'file_{len(file_parts)}'
                                        )
                                    )
                                )
                            )
            
            if file_parts:
                self._user_message_files[message_id] = file_parts
                print(f"✅ HOST_AGENT: Guardados {len(file_parts)} archivos para message_id={message_id}")

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_message(
        self, agent_name: str, message: str, tool_context: ToolContext
    ):
        """Sends a task either streaming (if supported) or non-streaming.

        This will send a message to the remote agent named agent_name.

        Args:
          agent_name: The name of the agent to send the task to.
          message: The message to send to the agent for the task.
          tool_context: The tool context this method runs in.

        Yields:
          A dictionary of JSON data.
        """
        print(f"\n{'='*60}")
        print(f"🔍 HOST_AGENT send_message:")
        print(f"   • agent_name: {agent_name}")
        print(f"   • message: {message[:100]}...")
        
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f'Agent {agent_name} not found')
        
        state = tool_context.state
        state['agent'] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        
        task_id = state.get('task_id', None)
        context_id = state.get('context_id', None)
        message_id = state.get('message_id', None)
        
        print(f"   • message_id: {message_id}")
        print(f"   • context_id: {context_id}")
        print(f"   • task_id: {task_id}")
        
        if not message_id:
            message_id = str(uuid.uuid4())

        # 🔧 NUEVO: Construir partes del mensaje incluyendo archivos
        message_parts = [Part(root=TextPart(text=message))]
        
        # Recuperar archivos del mensaje original si existen
        if message_id in self._user_message_files:
            file_parts = self._user_message_files[message_id]
            print(f"   ✅ Recuperados {len(file_parts)} archivos del cache")
            message_parts.extend(file_parts)
            
            # Limpiar el cache después de usar
            del self._user_message_files[message_id]
        else:
            print(f"   ⚠️ No hay archivos en cache para message_id={message_id}")

        print(f"   📤 Enviando {len(message_parts)} partes al remote agent")
        for i, part in enumerate(message_parts):
            p = part.root
            if p.kind == 'text':
                print(f"      Part {i}: Text ({len(p.text)} chars)")
            elif p.kind == 'file':
                print(f"      Part {i}: File ({p.file.mime_type})")
        print(f"{'='*60}\n")

        request_message = Message(
            role=Role.user,
            parts=message_parts,  # 🔧 Ahora incluye texto + archivos
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )
        
        response = await client.send_message(request_message)
        if isinstance(response, Message):
            return await convert_parts(response.parts, tool_context)
        
        task: Task = response
        # Assume completion unless a state returns that isn't complete
        state['session_active'] = task.status.state not in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.unknown,
        ]
        if task.context_id:
            state['context_id'] = task.context_id
        state['task_id'] = task.id
        if task.status.state == TaskState.input_required:
            # Force user input back
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.canceled:
            raise ValueError(f'Agent {agent_name} task {task.id} is cancelled')
        elif task.status.state == TaskState.failed:
            raise ValueError(f'Agent {agent_name} task {task.id} failed')
        
        response = []
        if task.status.message:
            if ts := self.timestamp_extension.get_timestamp(
                task.status.message
            ):
                response.append(f'[at {ts.astimezone().isoformat()}]')
            response.extend(
                await convert_parts(task.status.message.parts, tool_context)
            )
        if task.artifacts:
            for artifact in task.artifacts:
                if ts := self.timestamp_extension.get_timestamp(artifact):
                    response.append(f'[at {ts.astimezone().isoformat()}]')
                response.extend(
                    await convert_parts(artifact.parts, tool_context)
                )
        return response


async def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(await convert_part(p, tool_context))
    return rval


async def convert_part(part: Part, tool_context: ToolContext):
    """
    🔧 CORREGIDO: Maneja tanto string base64 como bytes directos
    """
    if part.root.kind == 'text':
        return part.root.text
    
    if part.root.kind == 'data':
        return part.root.data
    
    if part.root.kind == 'file':
        print(f"\n{'='*60}")
        print(f"🔍 HOST_AGENT convert_part (FILE):")
        print(f"  • mime_type: {part.root.file.mime_type}")
        print(f"  • name: {part.root.file.name}")
        print(f"  • bytes type: {type(part.root.file.bytes)}")
        
        file_id = part.root.file.name or str(uuid.uuid4())
        file_bytes_raw = part.root.file.bytes
        
        # 🔧 CORRECCIÓN: Normalizar el formato de bytes
        if isinstance(file_bytes_raw, str):
            # Si es string, asumimos que es base64 y lo decodificamos
            print(f"  • Decoding from base64 string ({len(file_bytes_raw)} chars)")
            try:
                file_bytes = base64.b64decode(file_bytes_raw)
                print(f"  ✅ Decoded to {len(file_bytes)} bytes")
            except Exception as e:
                print(f"  ❌ Error decoding base64: {e}")
                print(f"  • Trying UTF-8 encode as fallback")
                file_bytes = file_bytes_raw.encode('utf-8')
        elif isinstance(file_bytes_raw, bytes):
            # Si ya son bytes, usarlos directamente
            print(f"  • Already bytes ({len(file_bytes_raw)} bytes)")
            file_bytes = file_bytes_raw
        else:
            print(f"  ❌ Unsupported bytes type: {type(file_bytes_raw)}")
            print(f"{'='*60}\n")
            return f'Unknown file type: {type(file_bytes_raw)}'
        
        # Crear Part de ADK con inline_data
        file_part = types.Part(
            inline_data=types.Blob(
                mime_type=part.root.file.mime_type, 
                data=file_bytes
            )
        )
        
        # Guardar como artefacto
        await tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
        
        print(f"  ✅ File saved as artifact: {file_id}")
        print(f"{'='*60}\n")
        
        return DataPart(data={'artifact-file-id': file_id})
    
    return f'Unknown type: {part.root.kind}'
