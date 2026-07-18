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
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j_agent_memory import MemoryClient, MemorySettings, ExtractionConfig, ExtractorType
from neo4j_agent_memory.llm.adapters.sentence_transformers import SentenceTransformersProvider
from pydantic import BaseModel, Field, SecretStr

from .api_key_rotator import (
    google_key_rotator,
    create_google_llm,
    invoke_with_retry,
    ainvoke_with_retry,
)
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
        print(f"   context_id: {context_id[:12] if context_id else 'VACÍO'}...")
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
                                            # If the agent doesn't support streaming, fall back to message/send
                                            if 'not supported' in error_msg.lower() or 'unsupported' in error_msg.lower():
                                                print(f"⚠️ Agent doesn't support streaming, will fall back")
                                                streaming_failed = True
                                                break
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
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Non-streaming full response: {json.dumps(result, indent=2)[:2000]}")
                        
                        if 'error' in result:
                            error_msg = result['error'].get('message', str(result['error']))
                            return f"❌ Agent error: {error_msg}"
                        elif 'result' in result:
                            rpc_result = result['result']
                            print(f"🔑 rpc_result keys: {list(rpc_result.keys()) if isinstance(rpc_result, dict) else type(rpc_result)}")
                            
                            # Case 1: has a taskId → poll for result
                            task_id_key = rpc_result.get('taskId') or rpc_result.get('id') if isinstance(rpc_result, dict) else None
                            if task_id_key and isinstance(rpc_result, dict) and 'status' not in rpc_result:
                                print(f"📋 Task ID (polling): {task_id_key}")
                                return await self._poll_task_result(client, card.url, task_id_key, agent_name)
                            
                            # Case 2: result contains artifacts directly
                            if isinstance(rpc_result, dict) and 'artifacts' in rpc_result:
                                all_parts = []
                                for artifact in (rpc_result['artifacts'] or []):
                                    if isinstance(artifact, dict):
                                        all_parts.extend(artifact.get('parts', []))
                                text_parts, image_parts = self._extract_parts(all_parts)
                                print(f"📦 Artifact parts — text: {len(text_parts)}, images: {len(image_parts)}")
                                result_str = self._parts_to_marker(text_parts, image_parts)
                                if result_str:
                                    return result_str
                            
                            # Case 3: result contains status.message parts
                            if isinstance(rpc_result, dict) and 'status' in rpc_result:
                                status = rpc_result['status']
                                if isinstance(status, dict):
                                    status_msg = status.get('message')
                                    if isinstance(status_msg, dict):
                                        text_parts, image_parts = self._extract_parts(status_msg.get('parts', []))
                                        result_str = self._parts_to_marker(text_parts, image_parts)
                                        if result_str:
                                            return result_str
                            
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
    
    def _extract_parts(self, raw_parts: list) -> tuple[list[str], list[dict]]:
        """Extract text and image parts from A2A artifact parts.
        
        Handles both plain parts {'kind': 'file', 'file': {...}}
        and A2A root-wrapped parts {'root': {'kind': 'file', 'file': {...}}}.
        Returns (text_parts, image_parts) where image_parts are dicts with mime_type and bytes.
        """
        text_parts = []
        image_parts = []
        for raw in raw_parts:
            if not isinstance(raw, dict):
                continue
            # Unwrap 'root' if present (A2A SDK serialization)
            part = raw.get('root', raw)
            kind = part.get('kind', '')
            if kind == 'text':
                txt = part.get('text', '')
                if txt:
                    text_parts.append(txt)
            elif kind == 'file':
                file_data = part.get('file', {})
                if file_data.get('bytes'):
                    image_parts.append({
                        'mime_type': file_data.get('mime_type', 'image/png'),
                        'bytes': file_data['bytes']
                    })
        return text_parts, image_parts
    
    def _parts_to_marker(self, text_parts: list[str], image_parts: list[dict]) -> str | None:
        """Convert extracted parts into a response string (with __IMAGE_PARTS__ marker if needed)."""
        if image_parts:
            import json as _json
            text_prefix = '\n'.join(text_parts) if text_parts else 'Image generated successfully'
            return f"{text_prefix}\n__IMAGE_PARTS__:{_json.dumps(image_parts)}"
        if text_parts:
            return '\n'.join(text_parts)
        return None
    
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
                        # Priority 1: status message parts
                        if status_message and isinstance(status_message, dict):
                            t, i = self._extract_parts(status_message.get('parts', []))
                            r = self._parts_to_marker(t, i)
                            if r:
                                return r
                        
                        # Priority 2: response field parts
                        response_message = task_data.get('response', {})
                        if isinstance(response_message, dict):
                            t, i = self._extract_parts(response_message.get('parts', []))
                            r = self._parts_to_marker(t, i)
                            if r:
                                return r
                        
                        # Priority 3: artifacts (singular and plural)
                        all_parts = []
                        for art_key in ('artifact', 'artifacts'):
                            val = task_data.get(art_key)
                            if val:
                                art_list = val if isinstance(val, list) else [val]
                                for art in art_list:
                                    if isinstance(art, dict):
                                        all_parts.extend(art.get('parts', []))
                        if all_parts:
                            t, i = self._extract_parts(all_parts)
                            r = self._parts_to_marker(t, i)
                            if r:
                                return r
                        
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
        self.google_api_key = google_key_rotator.get_key()
        # Guardar sesiones y conversaciones localmente (no /tmp)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._sessions_file = os.path.join(base_dir, "beeai_active_sessions.json")
        self._conversations_file = os.path.join(base_dir, "beeai_conversations.json")
        self._load_sessions()
        self._load_conversations()

        # Initialize the LangChain Google Gemini Model (con key rotativa)
        self.llm = create_google_llm(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=8192
        )
        # Initialize the LangChain Google Gemini Vision Model specifically for images
        self.vision_llm = self.llm
        # Wrap it for BeeAI
        self.chat_model = LangChainChatModel(self.llm)

        # Initialize Neo4j Agent Memory direct connection to Aura DB
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE")

        if uri and username and password:
            print(f"🔗 Initializing Neo4j Agent Memory direct connection to: {uri} (db: {database})")
            try:
                # Use local SentenceTransformers BAAI/bge-small-en-v1.5 (384 dimensions)
                embedder = SentenceTransformersProvider(
                    model="BAAI/bge-small-en-v1.5",
                    device="cpu"
                )
                self.memory_settings = MemorySettings(
                    neo4j={
                        "uri": uri,
                        "username": username,
                        "password": SecretStr(password),
                        "database": database or "neo4j"
                    },
                    embedding=embedder,
                    llm="groq/llama-3.3-70b-versatile",
                    extraction=ExtractionConfig(
                        extractor_type=ExtractorType.LLM
                    )
                )
                self.neo4j_memory = MemoryClient(self.memory_settings)
                print("✅ Neo4j Agent Memory client initialized successfully with local embeddings")
            except Exception as e:
                print(f"❌ Error initializing Neo4j Agent Memory: {e}")
                self.neo4j_memory = None
        else:
            print("⚠️ Neo4j connection parameters not found in environment. Running without Neo4j Agent Memory.")
            self.neo4j_memory = None

    async def _ensure_neo4j_connected(self):
        """Idempotently ensure that the Neo4j Memory Client is connected."""
        if self.neo4j_memory:
            if not getattr(self, '_neo4j_connected', False):
                try:
                    print("🔌 Connecting Neo4j Agent Memory client...")
                    await self.neo4j_memory.connect()
                    self._neo4j_connected = True
                    print("✅ Connected to Neo4j Agent Memory")
                except Exception as e:
                    print(f"❌ Failed to connect to Neo4j Agent Memory: {e}")
                    self._neo4j_connected = False

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

    def _load_conversations(self):
        """Carga las conversaciones desde el disco para persistencia entre reinicios."""
        if os.path.exists(self._conversations_file):
            try:
                with open(self._conversations_file, 'r') as f:
                    data = json.load(f)
                for conv_data in data:
                    conv_id = conv_data.get('conversation_id', '')
                    if conv_id and not self.get_conversation(conv_id):
                        c = Conversation(
                            conversation_id=conv_id,
                            is_active=conv_data.get('is_active', True),
                            name=conv_data.get('name', ''),
                        )
                        c._memory = UnconstrainedMemory()
                        self._conversations.append(c)
                print(f"📖 Conversaciones cargadas desde disco: {len(self._conversations)}")
            except Exception as e:
                print(f"Error cargando conversaciones: {e}")
        else:
            print("ℹ️ No se encontró archivo de conversaciones previas.")

    def _save_conversations(self):
        """Guarda las conversaciones en el disco."""
        try:
            data = [
                {
                    'conversation_id': c.conversation_id,
                    'is_active': c.is_active,
                    'name': c.name,
                }
                for c in self._conversations
            ]
            with open(self._conversations_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error guardando conversaciones: {e}")

    async def _should_continue_active_session(self, user_message: str, active_agent: str) -> bool:
        """Determina si el mensaje del usuario debe ir al agente activo o al orquestador.
        
        Returns True si el mensaje es una respuesta/continuación de la sesión activa.
        Returns False si el usuario quiere hacer algo diferente.
        """
        # Frases que claramente rompen la sesión activa
        break_keywords = [
            "genera una imagen", "generá una imagen", "generar imagen",
            "dibuja", "dibujá", "crear imagen", "creá una imagen",
            "cambiar de tema", "otro tema", "hablemos de otra cosa",
            "quiero hablar con otro", "otro agente",
        ]
        msg_lower = user_message.lower().strip()
        if any(kw in msg_lower for kw in break_keywords):
            return False
        
        # Frases que claramente son continuación (respuestas de física, confusión, etc.)
        continue_keywords = [
            "no sé", "no se", "creo que", "la respuesta es", "sería",
            "no entiendo", "me parece", "pienso que", "puede ser",
        ]
        if any(kw in msg_lower for kw in continue_keywords):
            return True
        
        # Usar LLM para casos ambiguos
        try:
            from langchain_core.messages import HumanMessage
            
            prompt = f"""Eres un clasificador de intención. Un estudiante está en una sesión activa 
con el agente "{active_agent}" (un tutor de física que hace preguntas socráticas).

Determina si el siguiente mensaje del estudiante es:
- CONTINUAR: Es una respuesta a una pregunta de física, una duda, confusión, o cualquier 
  interacción relacionada con la sesión de tutoría actual. Incluye también pedidos de 
  "salir del modo socrático" o "dame la respuesta directa" (el agente de física maneja eso).
- CAMBIAR: El estudiante quiere hacer algo COMPLETAMENTE diferente, como generar una imagen,
  hablar de otro tema no relacionado con física, o usar otro servicio.

EN CASO DE DUDA, responde CONTINUAR.

Mensaje del estudiante: "{user_message}"

Responde SOLO: CONTINUAR o CAMBIAR"""
            
            response = invoke_with_retry(self.llm, [HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            print(f"🧠 Intención de sesión activa: '{user_message[:50]}...' → {result}")
            
            return "CAMBIAR" not in result
        except Exception as e:
            print(f"⚠️ Error detectando intención de sesión: {e}")
            return True  # En caso de error, continuar con la sesión activa

    async def create_conversation(self, conversation_id: str = None) -> Conversation:
        """Crea una nueva conversación, opcionalmente con un ID específico.
        
        Si conversation_id es proporcionado (ej: desde el frontend), lo reutiliza
        para mantener consistencia. Si no, genera uno nuevo.
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Verificar si ya existe (prevenir duplicados)
        existing = self.get_conversation(conversation_id)
        if existing:
            print(f"♻️ Conversación {conversation_id[:8]}... ya existe, reutilizando.")
            return existing
        
        c = Conversation(conversation_id=conversation_id, is_active=True)
        self._conversations.append(c)
        # Store memory for this conversation
        c._memory = UnconstrainedMemory()
        self._save_conversations()
        print(f"✨ Nueva conversación creada: {conversation_id[:8]}...")
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
        print(f"📨 process_message: context_id recibido del frontend = '{context_id[:8] if context_id else 'VACÍO'}...'")
        
        conversation = self.get_conversation(context_id)
        if not conversation:
            # CRÍTICO: Reutilizar el context_id del frontend en vez de generar uno nuevo.
            # Esto mantiene la sincronización frontend ↔ backend ↔ agente.
            print(f"⚠️ Conversación '{context_id[:8] if context_id else '?'}...' no encontrada. Creando con MISMO ID.")
            conversation = await self.create_conversation(conversation_id=context_id)
            context_id = conversation.conversation_id
            message.context_id = context_id
        else:
            print(f"✅ Conversación existente encontrada: {context_id[:8]}...")

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
        
        # Check if there are any images and extract their data for visual classification
        has_images = any(p.root.kind == 'file' for p in message.parts)
        image_data_list = []
        if has_images:
            for p in message.parts:
                if p.root.kind == 'file':
                    file_obj = p.root.file
                    mime_type = getattr(file_obj, 'mime_type', 'image/png') or 'image/png'
                    bytes_b64 = ""
                    if isinstance(file_obj, FileWithBytes) and file_obj.bytes:
                        bytes_b64 = file_obj.bytes
                    if bytes_b64:
                        image_data_list.append({
                            "mime_type": mime_type,
                            "bytes_b64": bytes_b64
                        })
            print(f"🖼️ Extracted {len(image_data_list)} image(s) for visual classification")

        # Retrieve Neo4j context — filtered to avoid duplicate chat history
        # The agent already maintains its own SemanticMemory.chat_history,
        # so we filter out short-term NAMS history to avoid duplicate context
        # that confuses the LLM into answering from memory instead of
        # the current turn.
        neo4j_context_text = ""
        if self.neo4j_memory:
            await self._ensure_neo4j_connected()
            if getattr(self, '_neo4j_connected', False):
                try:
                    student_id = conversation.name or context_id
                    print(f"🧠 Querying student NAMS context for student '{student_id}' (session_id={context_id})...")
                    ctx = await self.get_student_context(text_content, student_id=student_id, session_id=context_id)
                    if ctx:
                        raw_text = str(ctx)
                        # Filter out lines that look like chat history to avoid duplicates
                        # (the agent already tracks this via SemanticMemory.chat_history)
                        ignore_headers = (
                            '## conversation history',
                            '### relevant past messages',
                            'conversation history',
                            'relevant past messages'
                        )
                        filtered_lines = []
                        in_chat_history_section = False
                        for line in raw_text.split('\n'):
                            line_stripped = line.strip()
                            line_lower = line_stripped.lower()
                            if not line_lower:
                                continue
                            if any(h in line_lower for h in ignore_headers):
                                in_chat_history_section = True
                                continue
                            if '## relevant knowledge' in line_lower or '### user preferences' in line_lower:
                                in_chat_history_section = False
                                continue
                            if in_chat_history_section:
                                continue
                            # Skip lines that look like short-term chat messages
                            if any(line_lower.startswith(prefix) for prefix in [
                                'user:', 'assistant:', 'human:', 'ai:',
                                'usuario:', 'asistente:', 'q:', 'a:',
                                '- [user]', '- [assistant]', '- [human]', '- [ai]',
                                '- [usuario]', '- [asistente]'
                            ]):
                                continue
                            filtered_lines.append(line)
                        neo4j_context_text = '\n'.join(filtered_lines).strip()
                        # Truncate to avoid dominating the prompt
                        if len(neo4j_context_text) > 3000:
                            neo4j_context_text = neo4j_context_text[:3000] + "\n[... truncado]"
                        print(f"📖 Retrieved filtered Neo4j context: {len(neo4j_context_text)} chars (from {len(raw_text)} original)")
                except OSError as pipe_err:
                    # WinError 233 or other pipe/connection errors — force reconnect next time
                    print(f"⚠️ Neo4j connection error (will reconnect): {pipe_err}")
                    self._neo4j_connected = False
                except Exception as e:
                    print(f"⚠️ Error retrieving Neo4j context: {e}")

        # Save user message to Neo4j Short-Term memory
        if self.neo4j_memory and getattr(self, '_neo4j_connected', False):
            try:
                await self.neo4j_memory.short_term.add_message(
                    session_id=context_id,
                    role="user",
                    content=text_content
                )
                print("💾 Saved user message to Neo4j Short-Term memory")
            except OSError as pipe_err:
                print(f"⚠️ Neo4j pipe error saving user message (will reconnect): {pipe_err}")
                self._neo4j_connected = False
            except Exception as e:
                print(f"⚠️ Error saving user message to Neo4j: {e}")

        # Use BeeAI Workflow pattern for Gemini compatibility
        try:
            # Verificar si hay una sesión activa para este contexto
            # (ej: sesión socrática en progreso)
            active_agent = self._active_sessions.get(context_id)
            
            if active_agent:
                print(f"🔄 Sesión activa detectada para contexto {context_id[:8]}... → {active_agent}")
                
                # Verificar si el usuario quiere salir de la sesión activa
                # o hacer algo completamente diferente
                should_bypass = await self._should_continue_active_session(text_content, active_agent)
                
                if should_bypass:
                    print(f"📤 Continuando sesión activa con {active_agent}")
                    send_tool_instance = SendMessageToAgentTool(self)
                    
                    message_text = text_content
                    if neo4j_context_text:
                        message_text = f"[NAMS_CONTEXT]\n{neo4j_context_text}\n[/NAMS_CONTEXT]\n\n{text_content}"
                        print(f"🧠 Injected NAMS context into active session message sent to {active_agent}")
                        
                    send_input = SendMessageToAgentInput(
                        agent_name=active_agent,
                        message=message_text
                    )
                    resp_text = await send_tool_instance._run(send_input, None, None)
                else:
                    print(f"🔀 Usuario quiere cambiar de tema/agente. Limpiando sesión activa.")
                    del self._active_sessions[context_id]
                    self._save_sessions()
                    active_agent = None  # Caer al flujo del orquestador abajo
            
            if not active_agent:
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
                
                # Format history for the orchestrator
                history_texts = []
                # Get last 5 messages excluding the current one
                recent_messages = conversation.messages[:-1][-5:]
                for m in recent_messages:
                    role = m.role.name if hasattr(m.role, 'name') else str(m.role)
                    text = " ".join([p.root.text for p in m.parts if p.root.kind == 'text'])
                    if text:
                        history_texts.append(f"{role.upper()}: {text}")
                
                history_text = "\n".join(history_texts)
                
                # Execute workflow with initial state (including image data for visual classification)
                initial_state = OrchestratorState(
                    user_message=text_content,
                    has_images=has_images,
                    image_data_list=image_data_list,
                    history_text=history_text,
                    neo4j_context_text=neo4j_context_text
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

        # Save assistant response to Neo4j Short-Term memory
        if self.neo4j_memory and resp_text:
            await self._ensure_neo4j_connected()
            if getattr(self, '_neo4j_connected', False):
                try:
                    # Strip visual/binary markers from saved message for clean text history
                    clean_resp_text = resp_text.split("__IMAGE_PARTS__:")[0].strip()
                    await self.neo4j_memory.short_term.add_message(
                        session_id=context_id,
                        role="assistant",
                        content=clean_resp_text
                    )
                    print("💾 Saved assistant response to Neo4j Short-Term memory")
                except OSError as pipe_err:
                    print(f"⚠️ Neo4j pipe error saving assistant response (will reconnect): {pipe_err}")
                    self._neo4j_connected = False
                except Exception as e:
                    print(f"⚠️ Error saving assistant response to Neo4j: {e}")

            # Run Self-Learning in background to extract and persist user preferences/facts
            asyncio.create_task(self._learn_user_preferences(text_content, context_id))

        # Build response parts — detect image marker from image agent
        import json as _json
        response_parts: list[Part] = []
        IMAGE_MARKER = "__IMAGE_PARTS__:"
        if IMAGE_MARKER in resp_text:
            lines = resp_text.split('\n')
            non_image_lines = []
            for line in lines:
                if line.startswith(IMAGE_MARKER):
                    try:
                        image_parts_data = _json.loads(line[len(IMAGE_MARKER):])
                        for img in image_parts_data:
                            fp = FilePart(
                                file=FileWithBytes(
                                    bytes=img['bytes'],
                                    mime_type=img.get('mime_type', 'image/png'),
                                    name='generated_image.png'
                                )
                            )
                            response_parts.append(Part(root=fp))
                    except Exception as e:
                        print(f"⚠️ Could not parse image marker: {e}")
                else:
                    non_image_lines.append(line)
            text_without_marker = '\n'.join(non_image_lines).strip()
            if text_without_marker:
                response_parts.insert(0, Part(root=TextPart(text=text_without_marker)))
        else:
            response_parts = [Part(root=TextPart(text=resp_text))]

        response_msg = Message(
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            role=Role.agent,
            parts=response_parts
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

    async def get_student_context(self, query: str, student_id: str, session_id: str, max_items: int = 10) -> str:
        """Get combined context from memory, but filter long-term preferences strictly by student_id."""
        parts = []

        # 1. Short-term memory (session-scoped conversation history)
        short_term_context = await self.neo4j_memory.short_term.get_context(
            query,
            session_id=session_id,
            max_messages=max_items,
        )
        if short_term_context:
            parts.append(f"## Conversation History\n{short_term_context}")

        # 2. Long-term memory - filtered strictly to user_identifier = student_id
        embedding = None
        if self.neo4j_memory.long_term._embedder is not None:
            try:
                embedding = await self.neo4j_memory.long_term._embedder.embed(query)
            except Exception as e:
                print(f"⚠️ Error generating embedding for student context search: {e}")

        preferences = []
        if embedding is not None:
            try:
                # Custom cypher query to enforce User relationship
                cypher_query = """
                CALL db.index.vector.queryNodes('preference_embedding_idx', $limit, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                MATCH (u:User {identifier: $user_identifier})-[:HAS_PREFERENCE]->(node)
                RETURN node AS p, score
                ORDER BY score DESC
                """
                results = await self.neo4j_memory.long_term._client.execute_read(
                    cypher_query,
                    {
                        "embedding": embedding,
                        "limit": max_items,
                        "threshold": 0.7,
                        "user_identifier": student_id
                    }
                )
                for row in results:
                    pref_data = dict(row["p"])
                    pref = self.neo4j_memory.long_term._parse_preference(pref_data)
                    preferences.append(pref)
            except Exception as e:
                print(f"⚠️ Vector search failed, falling back to direct preference query: {e}")

        # Fallback: if vector search failed or returned nothing, fetch student preferences directly
        if not preferences:
            try:
                preferences = await self.neo4j_memory.long_term.get_preferences_for(student_id)
            except Exception as e:
                print(f"⚠️ Failed to fetch preferences for user {student_id}: {e}")

        if preferences:
            parts.append("## Relevant Knowledge")
            for pref in preferences:
                line = f"- [{pref.category}] {pref.preference}"
                if pref.context:
                    line += f" (context: {pref.context})"
                parts.append(line)

        # 3. Entities
        try:
            entities = await self.neo4j_memory.long_term.search_entities(query, limit=max_items)
            if entities:
                entity_parts = []
                for entity in entities:
                    type_str = entity.full_type
                    line = f"- {entity.display_name} ({type_str})"
                    if entity.description:
                        line += f": {entity.description}"
                    entity_parts.append(line)
                if entity_parts:
                    parts.append("## Relevant Entities\n" + "\n".join(entity_parts))
        except Exception as e:
            print(f"⚠️ Failed to search entities: {e}")

        return "\n\n".join(parts)

    async def add_deficiency(self, student_id: str, tema: str, correccion: str):
        """Save a deficiency verified by the teacher both semantically and structurally in Neo4j."""
        if not self.neo4j_memory:
            print("⚠️ Neo4j Memory Client not initialized. Cannot save deficiency.")
            return False

        try:
            await self._ensure_neo4j_connected()
            if not getattr(self, '_neo4j_connected', False):
                print("❌ Neo4j not connected. Cannot save deficiency.")
                return False

            # 1. Save semantically as a Preference node scoped to the student
            pref_text = f"El alumno tiene una falencia en '{tema}': {correccion}"
            await self.neo4j_memory.long_term.add_preference(
                category="falencia",
                preference=pref_text,
                user_identifier=student_id
            )
            print(f"✅ Saved semantic deficiency preference for student '{student_id}'")

            # 2. Save structurally as custom entities and relationships
            # Get or create the Student entity
            student_entity, _ = await self.neo4j_memory.long_term.add_entity(
                name=student_id,
                entity_type="Student",
                description=f"Perfil del estudiante {student_id}",
                resolve=False,
                deduplicate=True
            )

            # Get or create the Concept entity
            concept_entity, _ = await self.neo4j_memory.long_term.add_entity(
                name=tema,
                entity_type="Concept",
                description=f"Concepto de física: {tema}",
                resolve=False,
                deduplicate=True
            )

            # Add TIENE_FALENCIA relationship between Student and Concept
            await self.neo4j_memory.long_term.add_relationship(
                source=student_entity.id,
                target=concept_entity.id,
                relationship_type="TIENE_FALENCIA",
                description=correccion
            )
            print(f"✅ Saved structural relationship (Student {student_id}) -[:TIENE_FALENCIA]-> (Concept {tema})")
            return True

        except Exception as e:
            print(f"❌ Error adding deficiency: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _learn_user_preferences(self, user_message: str, context_id: str):
        """Extract and persist user preferences and facts to Neo4j Agent Memory in background."""
        if not self.neo4j_memory or not user_message:
            return
            
        try:
            print("🧠 Running self-learning extractor in background...")
            from langchain_core.messages import HumanMessage
            
            prompt = f"""Analiza el siguiente mensaje enviado por un usuario:
"{user_message}"

Determina si el usuario está expresando alguna preferencia personal persistente, hábito, estilo de comunicación preferido o dato biográfico relevante (por ejemplo: prefiere respuestas cortas, le gustan las explicaciones con analogías, se llama Diego, estudia ingeniería, etc.).

IMPORTANTE: No extraigas conclusiones de aprendizaje, correcciones ni conceptos físicos o matemáticos (por ejemplo, NO extraigas afirmaciones sobre leyes físicas, fórmulas o resolución de problemas). Solo debes extraer preferencias de estilo, formato o datos personales.

Si encuentras alguna preferencia o dato relevante, descríbelo en una frase corta y directa en tercera persona (ejemplo: "El usuario prefiere explicaciones con el método socrático", "El usuario prefiere respuestas concisas").
Si no encuentras nada relevante o es una pregunta general, responde únicamente con la palabra "NONE".

Responde en el formato:
Preferencia: <frase corta>
(O "NONE" si no hay nada)."""


            response = await ainvoke_with_retry(self.llm, [HumanMessage(content=prompt)])
            result = response.content.strip()
            
            if "NONE" in result.upper() or not result:
                print("🧠 Self-learning: No new persistent preferences detected.")
                return
                
            # Parse preferences
            preferences = []
            for line in result.split('\n'):
                if ":" in line:
                    pref_val = line.split(":", 1)[1].strip()
                    if pref_val and pref_val.upper() != "NONE":
                        preferences.append(pref_val)
                elif line.strip() and line.strip().upper() != "NONE":
                    preferences.append(line.strip())
            
            conversation = self.get_conversation(context_id)
            student_id = conversation.name or context_id if conversation else context_id
            for pref in preferences:
                print(f"💾 Self-learning: Extracted preference -> '{pref}' for student '{student_id}'")
                # Store preference in Neo4j Long-Term memory
                await self.neo4j_memory.long_term.add_preference(
                    category="user_preference",
                    preference=pref,
                    user_identifier=student_id
                )
                print(f"✅ Preference persisted in Neo4j Graph")
                
        except Exception as e:
            print(f"⚠️ Error in self-learning extractor: {e}")

