"""Orchestrator Agent - Entry Point (LangChain Based)

Este agente orquestador usa LangChain para coordinar múltiples agentes A2A:
- Image Generator: Generación y edición de imágenes
- Medical Images: Análisis de imágenes médicas

El orquestador lee las tarjetas de los agentes al iniciar y usa LangChain
para enrutar inteligentemente las solicitudes.
"""

import logging
import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.orchestrator_executor import OrchestratorExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MissingConfigError(Exception):
    """Excepción para configuración faltante."""


@click.command()
@click.option('--host', 'host', default='localhost', help='Host del servidor')
@click.option('--port', 'port', default=10003, help='Puerto del servidor')
@click.option(
    '--image-agent-url', 
    'image_agent_url',
    default='http://localhost:10001',
    help='URL del agente de generación de imágenes'
)
@click.option(
    '--medical-agent-url',
    'medical_agent_url', 
    default='http://localhost:10002',
    help='URL del agente de imágenes médicas'
)
def main(host, port, image_agent_url, medical_agent_url):
    """Inicia el agente orquestador basado en LangChain."""
    try:
        logger.info("=" * 80)
        logger.info("🎭 ORCHESTRATOR AGENT - LangChain + A2A Protocol")
        logger.info("=" * 80)
        
        # Verificar API Keys necesarias
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingConfigError(
                'GOOGLE_API_KEY environment variable not set (required for LangChain LLM)'
            )
        
        # Configuración de agentes
        agent_urls = {
            'image_generator': os.getenv('IMAGE_AGENT_URL', image_agent_url),
            'medical_images': os.getenv('MEDICAL_AGENT_URL', medical_agent_url),
        }
        
        logger.info("🔍 Configuración de agentes especializados:")
        logger.info(f"   • Image Generator: {agent_urls['image_generator']}")
        logger.info(f"   • Medical Images: {agent_urls['medical_images']}")
        logger.info("")
        logger.info("🧠 LangChain LLM: Google Gemini (gemini-2.0-flash)")
        
        # Definir capacidades del orquestador
        capabilities = AgentCapabilities(
            streaming=True,
            push_notifications=False
        )
        
        # Definir habilidades del orquestador
        skills = [
            AgentSkill(
                id='intelligent_orchestration',
                name='Intelligent Agent Orchestration',
                description=(
                    'Usa LangChain para analizar consultas y enrutarlas '
                    'inteligentemente a los agentes especializados apropiados: '
                    'generación de imágenes creativas o análisis médico de imágenes.'
                ),
                tags=[
                    'orchestration',
                    'langchain',
                    'multi-agent',
                    'intelligent routing'
                ],
                examples=[
                    'Generate a beautiful sunset image',
                    'Analyze this chest X-ray for me',
                    'Create an artistic portrait',
                    '¿Qué observas en esta tomografía?'
                ],
            ),
        ]
        
        # Crear tarjeta del orquestador
        orchestrator_url = (
            os.getenv('HOST_OVERRIDE')
            if os.getenv('HOST_OVERRIDE')
            else f'http://{host}:{port}/'
        )
        
        agent_card = AgentCard(
            name='LangChain Orchestrator Agent',
            description=(
                'Agente orquestador inteligente basado en LangChain que coordina '
                'agentes especializados en generación de imágenes y análisis médico. '
                'Analiza las consultas del usuario y las enruta automáticamente '
                'al agente apropiado.'
            ),
            url=orchestrator_url,
            version='1.0.0',
            default_input_modes=['text', 'text/plain', 'image/jpeg', 'image/png'],
            default_output_modes=['text', 'text/plain', 'image/png', 'image/jpeg'],
            capabilities=capabilities,
            skills=skills,
        )
        
        # Inicializar el executor (que usará LangChain internamente)
        request_handler = DefaultRequestHandler(
            agent_executor=OrchestratorExecutor(agent_urls),
            task_store=InMemoryTaskStore(),
        )
        
        # Crear aplicación
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        logger.info("=" * 80)
        logger.info(f"🚀 Starting Orchestrator on http://{host}:{port}")
        logger.info(f"🔗 Agent URL: {orchestrator_url}")
        logger.info("=" * 80)
        
        # Ejecutar servidor
        uvicorn.run(server.build(), host=host, port=port)
    
    except MissingConfigError as e:
        logger.error(f'❌ Configuration Error: {e}')
        logger.error('Please set the following environment variables:')
        logger.error('  - GOOGLE_API_KEY (for LangChain LLM)')
        exit(1)
    except Exception as e:
        logger.error(f'❌ Error during server startup: {e}')
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main()
