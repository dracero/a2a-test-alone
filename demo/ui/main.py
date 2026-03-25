"""A UI solution and host service to interact with the agent framework.
run:
  uv main.py
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import mesop as me
from components.api_key_dialog import api_key_dialog
from components.page_scaffold import page_scaffold
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pages.agent_list import agent_list_page
from pages.conversation import conversation_page
from pages.event_list import event_list_page
from pages.home import home_page_content
from pages.settings import settings_page_content
from pages.task_list import task_list_page
from service.server.server import ConversationServer
from state import host_agent_service
from state.state import AppState

# Load .env from project root (2 levels up: ui -> demo -> root)
root_dir = Path(__file__).resolve().parents[2]
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)



def on_load(e: me.LoadEvent):  # pylint: disable=unused-argument
    """On load event"""
    state = me.state(AppState)
    me.set_theme_mode(state.theme_mode)
    if 'conversation_id' in me.query_params:
        state.current_conversation_id = me.query_params['conversation_id']
    else:
        state.current_conversation_id = ''

    # check if the API key is set in the environment
    # and if the user is using Vertex AI
    uses_vertex_ai = (
        os.getenv('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
    )
    # Use GROQ_API_KEY for the orchestrator
    api_key = os.getenv('GROQ_API_KEY', '')

    if uses_vertex_ai:
        state.uses_vertex_ai = True
    elif api_key:
        state.api_key = api_key
    else:
        # Show the API key dialog if both are not set
        state.api_key_dialog_open = True


# Policy to allow the lit custom element to load
security_policy = me.SecurityPolicy(
    allowed_script_srcs=[
        'https://cdn.jsdelivr.net',
    ]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ['A2A_HOST'] = 'BEEAI'
    httpx_client_wrapper.start()
    
    server = ConversationServer(app, httpx_client_wrapper())
    
    # Auto-register local sample agents with retry logic
    async def register_with_retry(url: str, max_retries: int = 30, delay: float = 5.0):
        """Intenta registrar un agente con reintentos (hasta 2.5 minutos)"""
        for attempt in range(max_retries):
            try:
                server.manager.register_agent(url)
                print(f"✅ Successfully registered agent at {url}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    if attempt == 0:
                        print(f"⏳ Waiting for agent at {url} to be ready...")
                    if attempt % 5 == 0 and attempt > 0:
                        print(f"   Still waiting... ({attempt}/{max_retries} attempts)")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ Failed to register agent at {url} after {max_retries} attempts")
                    print(f"   The agent may still be processing PDFs. You can register it manually later using:")
                    print(f"   curl -X POST http://localhost:12000/agent/register/manual -H 'Content-Type: application/json' -d '{{\"url\": \"{url}\"}}'")
    
    print("🚀 Starting agent registration (this may take a few minutes if agents are processing PDFs)...")
    print("   Agents will be registered as they become available...")
    print()
    
    # Registrar agentes en paralelo con reintentos (solo 3 agentes ahora)
    await asyncio.gather(
        register_with_retry("http://localhost:10001"),
        register_with_retry("http://localhost:10002"),
        register_with_retry("http://localhost:10003"),
        return_exceptions=True
    )
    
    # Crear conversación inicial si no existe
    try:
        if not server.manager.conversations:
            await server.manager.create_conversation()
    except Exception as e:
        print(f"Error creating initial conversation: {e}")

    app.openapi_schema = None
    app.mount(
        '/',
        WSGIMiddleware(
            me.create_wsgi_app(
                debug_mode=os.environ.get('DEBUG_MODE', '') == 'true'
            )
        ),
    )
    app.setup()
    yield
    await httpx_client_wrapper.stop()


@me.page(
    path='/',
    title='Chat',
    on_load=on_load,
    security_policy=security_policy,
)
def home_page():
    """Main Page"""
    state = me.state(AppState)
    api_key_dialog()
    conversation_page(state)


@me.page(
    path='/agents',
    title='Agents',
    on_load=on_load,
    security_policy=security_policy,
)
def another_page():
    """Another Page"""
    api_key_dialog()
    agent_list_page(me.state(AppState))


@me.page(
    path='/conversation',
    title='Conversation',
    on_load=on_load,
    security_policy=security_policy,
)
def chat_page():
    """Conversation Page."""
    api_key_dialog()
    conversation_page(me.state(AppState))


@me.page(
    path='/event_list',
    title='Event List',
    on_load=on_load,
    security_policy=security_policy,
)
def event_page():
    """Event List Page."""
    api_key_dialog()
    event_list_page(me.state(AppState))


@me.page(
    path='/settings',
    title='Settings',
    on_load=on_load,
    security_policy=security_policy,
)
def settings_page():
    """Settings Page."""
    api_key_dialog()
    settings_page_content()


@me.page(
    path='/task_list',
    title='Task List',
    on_load=on_load,
    security_policy=security_policy,
)
def task_page():
    """Task List Page."""
    api_key_dialog()
    task_list_page(me.state(AppState))


class HTTPXClientWrapper:
    """Wrapper to return the singleton client where needed."""

    async_client: httpx.AsyncClient = None

    def start(self):
        """Instantiate the client. Call from the FastAPI startup hook."""
        self.async_client = httpx.AsyncClient(timeout=30)

    async def stop(self):
        """Gracefully shutdown. Call from FastAPI shutdown hook."""
        await self.async_client.aclose()
        self.async_client = None

    def __call__(self):
        """Calling the instantiated HTTPXClientWrapper returns the wrapped singleton."""
        # Ensure we don't use it if not started / running
        assert self.async_client is not None
        return self.async_client


# Setup the server global objects
httpx_client_wrapper = HTTPXClientWrapper()
agent_server = None


if __name__ == '__main__':
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(lifespan=lifespan)
    
    # Add CORS middleware to allow frontend requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup the connection details, these should be set in the environment
    host = os.environ.get('A2A_UI_HOST', '0.0.0.0')
    port = int(os.environ.get('A2A_UI_PORT', '12000'))

    # For client connections, resolve '0.0.0.0' to 'localhost'.
    # The server will still bind to the original host address.
    connect_host = 'localhost' if host == '0.0.0.0' else host

    # Set the client to talk to the server
    host_agent_service.server_url = f'http://{connect_host}:{port}'

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=0,
    )
