"""This file serves as the main entry point for the application.

It initializes the A2A server, defines the agent's capabilities,
and starts the server to handle incoming requests with LangSmith monitoring.
"""

import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import ImageGenerationAgent
from app.agent_executor import ImageGenerationAgentExecutor
from app.langsmith_config import LANGSMITH_ENABLED, get_langsmith_status
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (6 levels up: __main__.py -> app -> images -> agents -> python -> samples -> root)
root_dir = Path(__file__).resolve().parents[5]
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""
@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    """Entry point for the A2A + CrewAI Image generation sample with LangSmith."""
    try:
        # Check Groq API configuration
        if not os.getenv('GROQ_API_KEY'):
            raise MissingAPIKeyError(
                'GROQ_API_KEY environment variable not set.'
            )
        
        # También necesitamos Google API Key para generación de imágenes
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variable not set (needed for image generation).'
            )

        # Display LangSmith status
        logger.info("=" * 70)
        logger.info("🎭 CrewAI Image Generator - A2A Protocol + LangSmith")
        logger.info("=" * 70)
        
        langsmith_status = get_langsmith_status()
        if langsmith_status["enabled"]:
            logger.info("📊 LangSmith Monitoring: ENABLED")
            logger.info(f"   Project: {langsmith_status['project']}")
            logger.info(f"   Endpoint: {langsmith_status['endpoint']}")
        else:
            logger.info("📊 LangSmith Monitoring: DISABLED")
            logger.info("   Add LANGCHAIN_API_KEY to enable monitoring")
        logger.info("=" * 70)

        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='image_generator',
            name='Image Generator',
            description=(
                'Generate stunning, high-quality images on demand and leverage'
                ' powerful editing capabilities to modify, enhance, or completely'
                ' transform visuals.'
            ),
            tags=['generate image', 'edit image'],
            examples=['Generate a photorealistic image of raspberry lemonade'],
        )

        agent_host_url = (
            os.getenv('HOST_OVERRIDE')
            if os.getenv('HOST_OVERRIDE')
            else f'http://{host}:{port}/'
        )
        agent_card = AgentCard(
            name='Image Generator Agent',
            description=(
                'Generate stunning, high-quality images on demand and leverage'
                ' powerful editing capabilities to modify, enhance, or completely'
                ' transform visuals. All operations monitored with LangSmith.'
            ),
            url=agent_host_url,
            version='1.0.0',
            default_input_modes=ImageGenerationAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=ImageGenerationAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=ImageGenerationAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        
        logger.info(f"🚀 Starting A2A server on {host}:{port}")
        logger.info(f"🔗 Agent URL: {agent_host_url}")
        
        import uvicorn

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
