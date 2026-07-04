"""
Backend server for the A2A multi-agent system.
Provides REST API for the Next.js frontend.

Run:
  uv run main.py
"""

import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.server.server import ConversationServer

# Load .env from project root (2 levels up: ui -> demo -> root)
root_dir = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=root_dir / '.env', override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ['A2A_HOST'] = 'BEEAI'
    httpx_client_wrapper.start()

    server = ConversationServer(app, httpx_client_wrapper())

    async def register_with_retry(url: str, max_retries: int = 30, delay: float = 5.0):
        """Try to register an agent with retries (up to 2.5 minutes)."""
        for attempt in range(max_retries):
            try:
                server.manager.register_agent(url)
                print(f"✅ Registered agent at {url}")
                return
            except Exception:
                if attempt == 0:
                    print(f"⏳ Waiting for agent at {url}...")
                elif attempt % 5 == 0:
                    print(f"   Still waiting... ({attempt}/{max_retries})")
                await asyncio.sleep(delay)
        print(f"❌ Failed to register agent at {url} after {max_retries} attempts")

    print("🚀 Registering agents...")
    await asyncio.gather(
        register_with_retry("http://localhost:10001"),
        register_with_retry("http://localhost:10002"),
        register_with_retry("http://localhost:10003"),
        return_exceptions=True,
    )

    try:
        if not server.manager.conversations:
            await server.manager.create_conversation()
    except Exception as e:
        print(f"Warning: could not create initial conversation: {e}")

    app.openapi_schema = None
    app.setup()
    yield
    
    # Close Neo4j memory if active
    if hasattr(server.manager, 'neo4j_memory') and server.manager.neo4j_memory:
        try:
            await server.manager.neo4j_memory.close()
            print("🛑 Closed Neo4j Agent Memory client connection")
        except Exception as e:
            print(f"Error closing Neo4j memory client: {e}")

    await httpx_client_wrapper.stop()


class HTTPXClientWrapper:
    async_client: httpx.AsyncClient = None

    def start(self):
        self.async_client = httpx.AsyncClient(timeout=30)

    async def stop(self):
        await self.async_client.aclose()
        self.async_client = None

    def __call__(self):
        assert self.async_client is not None
        return self.async_client


httpx_client_wrapper = HTTPXClientWrapper()


if __name__ == '__main__':
    import uvicorn

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    host = os.environ.get('A2A_UI_HOST', '0.0.0.0')
    port = int(os.environ.get('A2A_UI_PORT', '12000'))

    uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=0, access_log=False)
