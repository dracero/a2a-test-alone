import asyncio
import base64
import datetime
import json
import os
import uuid
from typing import Any

import httpx
from a2a.types import (AgentCard, DataPart, FilePart, FileWithBytes,
                       FileWithUri, Message, Part, Role, Task, TaskState,
                       TaskStatus, TextPart)
from beeai_framework.adapters.a2a.agents import A2AAgent
# Re-use the existing HTTP wrapper or standard HTTPX as needed,
# or define tools carefully. We will use generic Langchain model.
from beeai_framework.adapters.langchain.backend.chat import LangChainChatModel
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.message import \
    AssistantMessage as BeeAssistantMessage
from beeai_framework.backend.message import Message as BeeMessage
from beeai_framework.backend.message import UserMessage as BeeUserMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from service.server.application_manager import ApplicationManager
from service.types import Conversation, Event


class ListRemoteAgentsInput(BaseModel):
    pass

class ListRemoteAgentsTool(Tool[ListRemoteAgentsInput, Any, str]):
    name = "list_remote_agents"
    description = "List the available remote agents you can use to delegate the task."

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    @property
    def input_schema(self) -> type[BaseModel]:
        return ListRemoteAgentsInput

    def _create_emitter(self) -> Emitter:
        return Emitter()

    async def _run(self, input: ListRemoteAgentsInput, options: Any | None, context: RunContext) -> str:
        agents = []
        for card in self.manager.agents:
            agents.append({'name': card.name, 'description': card.description})
        return json.dumps(agents)


class SendMessageToAgentInput(BaseModel):
    agent_name: str = Field(description="The name of the remote agent exactly as returned by list_remote_agents")
    message: str = Field(description="The message to send to the agent. Images from the user's message will be automatically included.")

class SendMessageToAgentTool(Tool[SendMessageToAgentInput, Any, str]):
    name = "send_message_to_agent"
    description = "Send a message to a specific remote agent to execute a task or delegate work. If the user sent images, they will be automatically forwarded to the agent. Always use the agent name exactly as returned by list_remote_agents."

    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    @property
    def input_schema(self) -> type[BaseModel]:
        return SendMessageToAgentInput

    def _create_emitter(self) -> Emitter:
        return Emitter()

    async def _run(self, input: SendMessageToAgentInput, options: Any | None, context: RunContext) -> str:
        agent_name = input.agent_name
        message_text = input.message
        
        print(f"\n{'='*60}")
        print(f"🔍 SEND MESSAGE TO AGENT TOOL")
        print(f"Agent name: {agent_name}")
        print(f"Available agents: {[c.name for c in self.manager.agents]}")
        print(f"{'='*60}\n")
        
        card = next((c for c in self.manager.agents if c.name == agent_name), None)
        if not card:
            available = ", ".join([c.name for c in self.manager.agents])
            return f"Agent '{agent_name}' not found. Available agents: {available}"
        
        print(f"✅ Found agent card:")
        print(f"   Name: {card.name}")
        print(f"   URL: {card.url}")
        print(f"   Description: {card.description}")
        
        # Get the current message being processed (with images if any)
        current_message = self.manager._current_processing_message
        
        # Build the message parts for the remote agent
        parts = []
        
        # Add any file parts (images) from the original message
        if current_message:
            for part in current_message.parts:
                if part.root.kind == 'file':
                    # The file part should have bytes (base64) or uri
                    # We need to ensure we're sending the actual bytes, not a cached URI
                    file_part = part.root.file
                    
                    if isinstance(file_part, FileWithBytes):
                        # Already has bytes, just add it
                        parts.append(part)
                    elif isinstance(file_part, FileWithUri):
                        # Has URI, need to fetch from cache
                        # This shouldn't happen in the orchestrator since we process the original message
                        # but let's handle it just in case
                        print(f"⚠️ Warning: File part has URI instead of bytes: {file_part.uri}")
                        # Try to get from cache if available
                        # For now, skip it as the cache is in the server layer
                        pass
                    else:
                        # Add it anyway
                        parts.append(part)
        
        # Add the text message
        parts.append(Part(root=TextPart(text=message_text)))
        
        # Create the A2A message
        context_id = current_message.context_id if current_message else ""
        a2a_message = Message(
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            role=Role.user,
            parts=parts
        )
        
        print(f"📤 Sending to {agent_name}:")
        print(f"   Parts: {len(parts)}")
        for i, p in enumerate(parts):
            if p.root.kind == 'file':
                print(f"   Part {i}: File ({p.root.file.mime_type}), has bytes: {isinstance(p.root.file, FileWithBytes)}")
            else:
                print(f"   Part {i}: Text")
        
        # Send to remote agent via A2A protocol (JSON-RPC with streaming fallback)
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Try streaming first
                jsonrpc_payload = {
                    "jsonrpc": "2.0",
                    "method": "message/stream",
                    "params": {
                        "message": a2a_message.model_dump(mode='json')
                    },
                    "id": str(uuid.uuid4())
                }
                
                print(f"🌐 Sending JSON-RPC (streaming) to: {card.url}")
                print(f"📦 Method: message/stream")
                
                # Use streaming request
                streaming_failed = False
                try:
                    async with client.stream('POST', card.url, json=jsonrpc_payload) as response:
                        print(f"📥 Response status: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_str = error_text.decode()[:500]
                            print(f"⚠️ Streaming failed: {error_str}")
                            
                            # Check if it's an "unsupported" error
                            if 'not supported' in error_str.lower() or 'unsupported' in error_str.lower():
                                streaming_failed = True
                            else:
                                return f"❌ Error: Agent {agent_name} returned status {response.status_code}: {error_str}"
                        
                        if not streaming_failed:
                            # Collect all chunks
                            full_response = []
                            task_id = None
                            chunk_count = 0
                            
                            print(f"📡 Starting to read stream...")
                            
                            async for line in response.aiter_lines():
                                if not line or not line.strip():
                                    continue
                                
                                chunk_count += 1
                                print(f"📦 Chunk {chunk_count}: {line[:200]}...")
                                
                                # SSE format: "data: {...}"
                                if line.startswith('data: '):
                                    data_str = line[6:]  # Remove "data: " prefix
                                    
                                    try:
                                        chunk_data = json.loads(data_str)
                                        
                                        # Handle JSON-RPC response
                                        if 'result' in chunk_data:
                                            result = chunk_data['result']
                                            print(f"✅ Result chunk: {json.dumps(result, indent=2)[:300]}...")
                                            
                                            # Extract task ID
                                            if isinstance(result, dict) and 'taskId' in result:
                                                task_id = result['taskId']
                                                print(f"📋 Task ID: {task_id}")
                                            
                                            # Extract status updates and text from status messages
                                            if isinstance(result, dict) and 'status' in result:
                                                status = result['status']
                                                print(f"📊 Status: {status}")
                                                
                                                # Extract text from status message (ej: preguntas socráticas)
                                                if isinstance(status, dict):
                                                    state = status.get('state', '')
                                                    status_msg = status.get('message')
                                                    
                                                    # Gestionar sesiones activas
                                                    if state == 'input-required' and context_id:
                                                        self.manager._active_sessions[context_id] = agent_name
                                                        self.manager._save_sessions()
                                                        print(f"📌 Sesión activa guardada: {context_id[:8]}... → {agent_name}")
                                                    elif state == 'completed' and context_id:
                                                        if context_id in self.manager._active_sessions:
                                                            del self.manager._active_sessions[context_id]
                                                            self.manager._save_sessions()
                                                            print(f"🧹 Sesión activa limpiada: {context_id[:8]}...")
                                                    
                                                    if status_msg and isinstance(status_msg, dict):
                                                        parts = status_msg.get('parts', [])
                                                        for part in parts:
                                                            if isinstance(part, dict) and part.get('kind') == 'text':
                                                                text = part.get('text', '')
                                                                if text:
                                                                    full_response.append(text)
                                                                    print(f"📝 Text from status message ({state}): {text[:100]}...")
                                            
                                            # Extract response parts from 'response' field
                                            if isinstance(result, dict) and 'response' in result:
                                                response_msg = result['response']
                                                if isinstance(response_msg, dict) and 'parts' in response_msg:
                                                    for part in response_msg['parts']:
                                                        if isinstance(part, dict) and part.get('kind') == 'text':
                                                            text = part.get('text', '')
                                                            if text:
                                                                full_response.append(text)
                                                                print(f"📝 Text chunk from response: {text[:100]}...")
                                            
                                            # Extract response parts from 'artifact' field (some agents use this)
                                            if isinstance(result, dict) and 'artifact' in result:
                                                artifact = result['artifact']
                                                # Handle both dict with 'parts' and direct dict with 'kind'
                                                if isinstance(artifact, dict):
                                                    if 'parts' in artifact:
                                                        # artifact is a message with parts
                                                        for part in artifact['parts']:
                                                            if isinstance(part, dict) and part.get('kind') == 'text':
                                                                text = part.get('text', '')
                                                                if text:
                                                                    full_response.append(text)
                                                                    print(f"📝 Text chunk from artifact.parts: {text[:100]}...")
                                                    elif artifact.get('kind') == 'text':
                                                        # artifact is directly a text part
                                                        text = artifact.get('text', '')
                                                        if text:
                                                            full_response.append(text)
                                                            print(f"📝 Text chunk from artifact (direct): {text[:100]}...")
                                        
                                        elif 'error' in chunk_data:
                                            error_msg = chunk_data['error'].get('message', str(chunk_data['error']))
                                            print(f"❌ Error in chunk: {error_msg}")
                                            return f"❌ Agent error: {error_msg}"
                                            
                                    except json.JSONDecodeError as e:
                                        print(f"⚠️ Could not parse chunk as JSON: {data_str[:100]}")
                                        continue
                            
                            print(f"📡 Stream ended. Total chunks: {chunk_count}")
                            print(f"📝 Full response parts: {len(full_response)}")
                            
                            # Return the complete response
                            if full_response:
                                complete_text = '\n'.join(full_response)
                                print(f"✅ Agent completed successfully with {len(complete_text)} chars")
                                return complete_text
                            elif task_id:
                                # If we got a task ID but no response, poll for it
                                print(f"⏳ No response in stream, polling for task result: {task_id}")
                                return await self._poll_task_result(client, card.url, task_id, agent_name)
                            else:
                                print(f"⚠️ No response and no task ID from streaming")
                                streaming_failed = True
                
                except Exception as e:
                    print(f"⚠️ Streaming exception: {e}")
                    streaming_failed = True
                
                # Fallback to non-streaming message/send
                if streaming_failed:
                    print(f"🔄 Falling back to non-streaming message/send")
                    
                    jsonrpc_payload = {
                        "jsonrpc": "2.0",
                        "method": "message/send",
                        "params": {
                            "message": a2a_message.model_dump(mode='json')
                        },
                        "id": str(uuid.uuid4())
                    }
                    
                    # Use the base URL for JSON-RPC (not /message/send)
                    response = await client.post(card.url, json=jsonrpc_payload)
                    print(f"📥 Non-streaming response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Non-streaming response: {json.dumps(result, indent=2)[:500]}...")
                        
                        if 'error' in result:
                            error_msg = result['error'].get('message', str(result['error']))
                            return f"❌ Agent error: {error_msg}"
                        elif 'result' in result:
                            rpc_result = result['result']
                            if isinstance(rpc_result, dict) and 'taskId' in rpc_result:
                                task_id = rpc_result['taskId']
                                print(f"📋 Task ID: {task_id}, polling for result...")
                                return await self._poll_task_result(client, card.url, task_id, agent_name)
                            else:
                                return f"✅ Message sent to {agent_name} successfully."
                        else:
                            return f"✅ Message sent to {agent_name}."
                    else:
                        error_text = response.text[:500]
                        return f"❌ Error: Agent {agent_name} returned status {response.status_code}: {error_text}"
                        
        except httpx.ConnectError as e:
            print(f"❌ Connection error: {e}")
            return f"❌ Error: Cannot connect to agent {agent_name}. The agent may not be running at {card.url}"
        except httpx.TimeoutException as e:
            print(f"❌ Timeout error: {e}")
            return f"❌ Error: Agent {agent_name} timed out. The agent may be overloaded or stuck."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error communicating with agent {agent_name}: {str(e)}"
    
    async def _poll_task_result(self, client, agent_url, task_id, agent_name):
        """Poll for task result when streaming doesn't provide it"""
        max_attempts = 60
        for attempt in range(max_attempts):
            await asyncio.sleep(1)
            
            task_payload = {
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": str(uuid.uuid4())
            }
            
            task_response = await client.post(agent_url, json=task_payload)
            
            if task_response.status_code == 200:
                task_result = task_response.json()
                
                if 'result' in task_result:
                    task_data = task_result['result']
                    status = task_data.get('status')
                    
                    # status puede ser un dict {'state': '...', 'message': {...}} 
                    # o un string directamente
                    if isinstance(status, dict):
                        state = status.get('state', '')
                        status_message = status.get('message')
                    else:
                        state = str(status) if status else ''
                        status_message = None
                    
                    print(f"📊 Task status: {status} (attempt {attempt + 1}/{max_attempts})")
                    
                    if state == 'completed':
                        # Try to extract from status message first
                        if status_message and isinstance(status_message, dict):
                            parts = status_message.get('parts', [])
                            text_parts = [p.get('text', '') for p in parts 
                                         if isinstance(p, dict) and p.get('kind') == 'text' and p.get('text')]
                            if text_parts:
                                return '\n'.join(text_parts)
                        
                        # Try to extract from 'response' field
                        response_message = task_data.get('response', {})
                        if isinstance(response_message, dict):
                            parts = response_message.get('parts', [])
                            if parts:
                                text_parts = []
                                for part in parts:
                                    if isinstance(part, dict) and part.get('kind') == 'text':
                                        text_parts.append(part.get('text', ''))
                                
                                if text_parts:
                                    return '\n'.join(text_parts)
                        
                        # Try to extract from 'artifact' field
                        artifact = task_data.get('artifact', {})
                        if isinstance(artifact, dict):
                            if 'parts' in artifact:
                                text_parts = []
                                for part in artifact['parts']:
                                    if isinstance(part, dict) and part.get('kind') == 'text':
                                        text_parts.append(part.get('text', ''))
                                
                                if text_parts:
                                    return '\n'.join(text_parts)
                            elif artifact.get('kind') == 'text':
                                return artifact.get('text', '')
                        
                        return f"✅ Agent {agent_name} completed the task."
                    
                    elif state == 'input-required':
                        # El agente necesita input del usuario (ej: pregunta socrática)
                        # Extraer el mensaje y devolverlo como respuesta
                        print(f"🎓 Agent {agent_name} requires user input (e.g. Socratic question)")
                        
                        if status_message and isinstance(status_message, dict):
                            parts = status_message.get('parts', [])
                            text_parts = [p.get('text', '') for p in parts 
                                         if isinstance(p, dict) and p.get('kind') == 'text' and p.get('text')]
                            if text_parts:
                                return '\n'.join(text_parts)
                        
                        # Fallback: buscar en artifacts o response
                        response_message = task_data.get('response', {})
                        if isinstance(response_message, dict):
                            parts = response_message.get('parts', [])
                            text_parts = [p.get('text', '') for p in parts 
                                         if isinstance(p, dict) and p.get('kind') == 'text' and p.get('text')]
                            if text_parts:
                                return '\n'.join(text_parts)
                        
                        return f"🎓 Agent {agent_name} is waiting for your response."
                    
                    elif state == 'failed':
                        error = task_data.get('error', 'Unknown error')
                        return f"❌ Agent {agent_name} failed: {error}"
        
        return f"⏱️ Agent {agent_name} is still processing. Task ID: {task_id}"


class BeeAIHostManager(ApplicationManager):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_key: str = '',
        uses_vertex_ai: bool = False,
    ):
        self._conversations: list[Conversation] = []
        self._messages: list[Message] = []
        self._tasks: list[Task] = []
        self._events: dict[str, Event] = {}
        self._agents: list[AgentCard] = []
        self._pending_message_ids: list[str] = []
        # Mapeo context_id → agent_name para sesiones activas (ej: socrático)
        self._active_sessions: dict[str, str] = {}

        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._sessions_file = "/tmp/beeai_active_sessions.json"
        self._load_sessions()

        # Initialize the LangChain Groq Model
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=self.api_key,
            temperature=0.3,
            max_tokens=4096
        )
        # Wrap it for BeeAI
        self.chat_model = LangChainChatModel(self.llm)

    def _load_sessions(self):
        """Carga las sesiones activas desde el disco."""
        if os.path.exists(self._sessions_file):
            try:
                with open(self._sessions_file, 'r') as f:
                    self._active_sessions = json.load(f)
                print(f"📖 Sesiones cargadas: {len(self._active_sessions)}")
            except Exception as e:
                print(f"Error cargando sesiones: {e}")
                self._active_sessions = {}
        else:
            self._active_sessions = {}

    def _save_sessions(self):
        """Guarda las sesiones activas en el disco."""
        try:
            with open(self._sessions_file, 'w') as f:
                json.dump(self._active_sessions, f)
        except Exception as e:
            print(f"Error guardando sesiones: {e}")

    async def create_conversation(self) -> Conversation:
        conversation_id = str(uuid.uuid4())
        c = Conversation(conversation_id=conversation_id, is_active=True)
        self._conversations.append(c)
        # Store memory for this conversation
        # Simplified: we use UnconstrainedMemory for simplicity
        c._memory = UnconstrainedMemory() 
        return c

    def sanitize_message(self, message: Message) -> Message:
        return message

    def get_pending_messages(self) -> list[tuple[str, str]]:
        return [(msg_id, "Working...") for msg_id in self._pending_message_ids]

    def register_agent(self, url: str):
        # Fetch agent card synchronously or via a task
        try:
            # Asegurar que la URL tenga el protocolo
            if not url.startswith('http://') and not url.startswith('https://'):
                url = f'http://{url}'
            
            resp = httpx.get(f"{url}/.well-known/agent-card.json")
            if resp.status_code == 200:
                data = resp.json()
                data['url'] = url
                card = AgentCard(**data)
                self._agents.append(card)
                print(f"✅ Agent registered: {card.name} at {url}")
        except Exception as e:
            print(f"Error registering agent {url}: {e}")

    @property
    def conversations(self) -> list[Conversation]:
        return self._conversations

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    @property
    def agents(self) -> list[AgentCard]:
        return self._agents

    @property
    def events(self) -> list[Event]:
        return sorted(self._events.values(), key=lambda x: x.timestamp)

    def get_conversation(self, conversation_id: str | None) -> Conversation | None:
        if not conversation_id:
            return None
        return next((c for c in self._conversations if c.conversation_id == conversation_id), None)

    async def process_message(self, message: Message):
        self._pending_message_ids.append(message.message_id)
        
        # Store the current message being processed so tools can access it
        self._current_processing_message = message
        
        context_id = message.context_id
        conversation = self.get_conversation(context_id)
        if not conversation:
            print("Conversation not found. Creating a new one.")
            conversation = await self.create_conversation()
            context_id = conversation.conversation_id
            message.context_id = context_id

        if not hasattr(conversation, '_memory'):
            conversation._memory = UnconstrainedMemory()

        self._messages.append(message)
        conversation.messages.append(message)

        self._events[message.message_id] = Event(
            id=message.message_id,
            actor='user',
            content=message,
            timestamp=datetime.datetime.utcnow().timestamp()
        )

        # Extract text from message parts
        text_content = " ".join([p.root.text for p in message.parts if p.root.kind == 'text'])
        
        # Check if there are any images
        has_images = any(p.root.kind == 'file' for p in message.parts)

        # Use BeeAI Workflow pattern for Gemini compatibility
        try:
            # Verificar si hay una sesión activa para este contexto
            # (ej: sesión socrática en progreso)
            active_agent = self._active_sessions.get(context_id)
            
            if active_agent:
                print(f"🔄 Sesión activa detectada para contexto {context_id[:8]}... → {active_agent}")
                print(f"📤 Enviando directamente al agente (bypass del orquestador)")
                
                send_tool_instance = SendMessageToAgentTool(self)
                send_input = SendMessageToAgentInput(
                    agent_name=active_agent,
                    message=text_content
                )
                resp_text = await send_tool_instance._run(send_input, None, None)
            else:
                from service.server.beeai_orchestrator_workflow import (
                    OrchestratorState, create_orchestrator_workflow)
                
                print("🚀 Starting BeeAI Workflow orchestration...")
                
                # Create the workflow
                workflow = await create_orchestrator_workflow(
                    manager=self,
                    list_tool=ListRemoteAgentsTool(self),
                    send_tool=SendMessageToAgentTool(self),
                    llm=self.llm  # Pass the raw LangChain LLM
                )
                
                # Execute workflow with initial state
                initial_state = OrchestratorState(
                    user_message=text_content,
                    has_images=has_images
                )
                
                workflow_run = await workflow.run(initial_state)
                
                # Extract the final state from the workflow run
                final_state = workflow_run.state
                
                # Extract response from final state
                if final_state.error:
                    resp_text = f"Error: {final_state.error}"
                elif final_state.agent_response:
                    resp_text = final_state.agent_response
                else:
                    resp_text = "No response from agent."
                    
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            resp_text = f"An error occurred during orchestration: {e}"
            # Limpiar sesión activa si hubo error
            if context_id in self._active_sessions:
                del self._active_sessions[context_id]
                self._save_sessions()

        response_msg = Message(
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            role=Role.agent,
            parts=[Part(root=TextPart(text=resp_text))]
        )
        self._messages.append(response_msg)
        conversation.messages.append(response_msg)
        
        self._events[response_msg.message_id] = Event(
            id=response_msg.message_id,
            actor='host_agent',
            content=response_msg,
            timestamp=datetime.datetime.utcnow().timestamp()
        )

        if message.message_id in self._pending_message_ids:
            self._pending_message_ids.remove(message.message_id)

