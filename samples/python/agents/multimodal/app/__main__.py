import asyncio
import logging
import os
import sys
from pathlib import Path

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (BasePushNotificationSender,
                              InMemoryPushNotificationConfigStore,
                              InMemoryTaskStore)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import PhysicsMultimodalAgent
from app.agent_executor import PhysicsAgentExecutor
from app.custom_request_handler import PhysicsAgentExecutorWrapper
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Excepción para API keys faltantes."""


async def inicializar_agente_con_pdfs(qdrant_url: str, qdrant_api_key: str, pdf_dir: str = None):
    """
    Inicializar agente y procesar PDFs si es la primera vez.
    
    Args:
        qdrant_url: URL de Qdrant
        qdrant_api_key: API Key de Qdrant
        pdf_dir: Directorio con PDFs (opcional, lee de variable de entorno)
    
    Returns:
        PhysicsAgentExecutor configurado
    """
    # Crear executor
    executor = PhysicsAgentExecutor(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    
    # Verificar si ya existen las colecciones en Qdrant
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    colecciones_existen = False
    try:
        await client.get_collection(executor.agent.text_collection)
        colecciones_existen = True
        logger.info("✅ Colecciones ya existen en Qdrant, saltando procesamiento de PDFs")
    except Exception:
        logger.info("⚠️ Colecciones no encontradas, se procesarán los PDFs")
    
    # Si no existen colecciones, procesar PDFs
    if not colecciones_existen:
        # Obtener directorio de PDFs
        if pdf_dir is None:
            pdf_dir = os.getenv('PDF_DIR', '/content')  # Default para compatibilidad con Colab
        
        pdf_path = Path(pdf_dir)
        
        # Buscar PDFs
        if pdf_path.exists():
            # Opción 1: Buscar archivos con patrón arch*.pdf
            pdf_files = list(pdf_path.glob("arch*.pdf"))
            
            # Opción 2: Si no hay arch*.pdf, buscar todos los PDFs
            if not pdf_files:
                pdf_files = list(pdf_path.glob("*.pdf"))
            
            pdf_files = [str(f) for f in pdf_files]
            
            if pdf_files:
                logger.info(f"📚 Encontrados {len(pdf_files)} archivos PDF")
                for f in pdf_files:
                    logger.info(f"   • {Path(f).name}")
                
                # Procesar PDFs (esto también extrae el temario)
                logger.info("🔄 Procesando PDFs y extrayendo temario...")
                temario = await executor.agent.procesar_y_almacenar_pdfs(pdf_files)
                
                logger.info(f"✅ PDFs procesados y temario extraído")
                logger.info(f"📋 Temario preview:\n{temario[:500]}...\n")
            else:
                logger.warning(f"⚠️ No se encontraron archivos PDF en {pdf_dir}")
                logger.info("💡 El agente funcionará sin documentos pre-cargados")
        else:
            logger.warning(f"⚠️ Directorio de PDFs no existe: {pdf_dir}")
            logger.info("💡 Configura PDF_DIR en .env o el agente funcionará sin documentos")
    
    return executor


@click.command()
@click.option('--host', 'host', default='localhost', help='Host del servidor')
@click.option('--port', 'port', default=10003, help='Puerto del servidor')
@click.option('--pdf-dir', 'pdf_dir', default=None, help='Directorio con PDFs a procesar')
def main(host, port, pdf_dir):
    """Inicia el servidor del Asistente de Física Multimodal."""
    
    async def startup():
        try:
            # Verificar Google API Key
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set.'
                )
            
            # Verificar Qdrant
            if not os.getenv('QDRANT_URL'):
                raise MissingAPIKeyError(
                    'QDRANT_URL environment variable not set.'
                )
            
            if not os.getenv('QDRANT_KEY'):
                raise MissingAPIKeyError(
                    'QDRANT_KEY environment variable not set.'
                )
            
            # Definir capacidades del agente
            capabilities = AgentCapabilities(
                streaming=True, 
                push_notifications=True
            )
            
            # Definir habilidad principal
            skill = AgentSkill(
                id='physics_multimodal_analysis',
                name='Análisis de Física Multimodal',
                description='Asistente de Física I que analiza consultas, imágenes de experimentos y busca en documentos académicos para proporcionar explicaciones didácticas',
                tags=[
                    'física', 
                    'análisis multimodal', 
                    'búsqueda vectorial',
                    'imágenes científicas',
                    'educación universitaria',
                    'UBA'
                ],
                examples=[
                    '¿Qué es la cantidad de movimiento y cómo se conserva?',
                    'Analiza este diagrama de un péndulo físico',
                    'Explica el principio de Bernoulli con ejemplos',
                    '¿Qué fenómeno físico se observa en esta imagen?',
                    'Dame ejemplos de colisiones elásticas e inelásticas'
                ],
            )
            
            # Crear tarjeta del agente
            agent_card = AgentCard(
                name='Asistente de Física Multimodal',
                description='Profesor virtual de Física I especializado en análisis multimodal (texto + imágenes), búsqueda en documentos académicos y memoria conversacional',
                url=f'http://{host}:{port}/',
                version='1.0.0',
                default_input_modes=PhysicsMultimodalAgent.SUPPORTED_CONTENT_TYPES,
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
            
            # Inicializar agente con procesamiento automático de PDFs
            logger.info("🔧 Inicializando agente con procesamiento automático de PDFs...")
            
            real_executor = await inicializar_agente_con_pdfs(
                qdrant_url=os.getenv('QDRANT_URL'),
                qdrant_api_key=os.getenv('QDRANT_KEY'),
                pdf_dir=pdf_dir or os.getenv('PDF_DIR')
            )
            
            logger.info("✅ Agente inicializado correctamente")
            
            # Wrapper que pre-procesa inline_data → FilePart
            wrapped_executor = PhysicsAgentExecutorWrapper(real_executor)
            
            # Request handler con el executor envuelto
            request_handler = DefaultRequestHandler(
                agent_executor=wrapped_executor,
                task_store=InMemoryTaskStore(),
                push_config_store=push_config_store,
                push_sender=push_sender
            )
            
            # Crear aplicación
            server = A2AStarletteApplication(
                agent_card=agent_card, 
                http_handler=request_handler
            )
            
            logger.info(f"📚 Iniciando Asistente de Física en http://{host}:{port}")
            logger.info(f"📋 Capacidades:")
            logger.info(f"   • Análisis de imágenes de física")
            logger.info(f"   • Búsqueda vectorial en documentos (Qdrant)")
            logger.info(f"   • Memoria conversacional")
            logger.info(f"   • Clasificación según temario UBA")
            logger.info(f"   • Procesamiento automático de PDFs")
            logger.info(f"🔧 Modo: Procesamiento automático + Wrapper inline_data")
            
            return server
        
        except MissingAPIKeyError as e:
            logger.error(f'❌ Error: {e}')
            logger.error('Por favor, configura las siguientes variables de entorno:')
            logger.error('  - GOOGLE_API_KEY (para Gemini)')
            logger.error('  - QDRANT_URL (URL de tu instancia Qdrant)')
            logger.error('  - QDRANT_KEY (API Key de Qdrant)')
            logger.error('Opcional:')
            logger.error('  - PDF_DIR (directorio con PDFs, default: /content)')
            sys.exit(1)
        except Exception as e:
            logger.error(f'❌ Error durante el inicio: {e}', exc_info=True)
            sys.exit(1)
    
    # Ejecutar startup asíncrono y obtener servidor
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = loop.run_until_complete(startup())
    
    # Ejecutar servidor
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    main()
