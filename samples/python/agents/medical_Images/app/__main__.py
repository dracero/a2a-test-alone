import logging
import os
import sys

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (BasePushNotificationSender,
                              InMemoryPushNotificationConfigStore,
                              InMemoryTaskStore)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import MedicalAgent
from app.agent_executor import MedicalAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Excepción para API keys faltantes."""


@click.command()
@click.option('--host', 'host', default='localhost', help='Host del servidor')
@click.option('--port', 'port', default=10002, help='Puerto del servidor')
def main(host, port):
    """Inicia el servidor del Asistente Médico."""
    try:
        # Verificar Google API Key
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variable not set.'
            )
        
        # Verificar Tavily API Key
        if not os.getenv('TAVILY_API_KEY'):
            raise MissingAPIKeyError(
                'TAVILY_API_KEY environment variable not set.'
            )
        
        # Definir capacidades del agente
        capabilities = AgentCapabilities(
            streaming=True, 
            push_notifications=True
        )
        
        # Definir habilidad principal
        skill = AgentSkill(
            id='medical_analysis',
            name='Análisis Médico con Imágenes',
            description='Analiza imágenes médicas y proporciona evaluaciones clínicas profesionales basadas en hallazgos visuales y búsqueda médica',
            tags=[
                'análisis médico', 
                'imágenes médicas', 
                'diagnóstico asistido',
                'radiología',
                'búsqueda médica'
            ],
            examples=[
                '¿Qué observas en esta radiografía de tórax?',
                'Analiza estas imágenes de resonancia magnética',
                '¿Qué hallazgos puedes identificar en esta tomografía?',
                'Necesito una segunda opinión sobre estos estudios'
            ],
        )
        
        # Crear tarjeta del agente
        agent_card = AgentCard(
            name='Asistente Médico',
            description='Asistente médico especializado en análisis de imágenes médicas con búsqueda de información clínica y memoria conversacional',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=MedicalAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=['text', 'text/plain'],
            capabilities=capabilities,
            skills=[skill],
        )
        
        # Inicializar componentes del servidor
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )
        
        request_handler = DefaultRequestHandler(
            agent_executor=MedicalAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        
        # Crear aplicación
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )
        
        logger.info(f"🏥 Iniciando Asistente Médico en http://{host}:{port}")
        logger.info(f"📋 Capacidades: Análisis de imágenes, Búsqueda médica, Memoria conversacional")
        
        # Ejecutar servidor
        uvicorn.run(server.build(), host=host, port=port)
    
    except MissingAPIKeyError as e:
        logger.error(f'❌ Error: {e}')
        logger.error('Por favor, configura las siguientes variables de entorno:')
        logger.error('  - GOOGLE_API_KEY (para Gemini)')
        logger.error('  - TAVILY_API_KEY (para búsqueda médica)')
        sys.exit(1)
    except Exception as e:
        logger.error(f'❌ Error durante el inicio del servidor: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
