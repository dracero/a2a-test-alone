# samples/python/agents/palimodal/app/__main__.py

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
from app.agent import PhysicsPalimodalAgent
from app.agent_executor import PhysicsPalimodalExecutor
from app.custom_request_handler import PhysicsAgentExecutorWrapper
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Excepción para API keys faltantes."""


async def inicializar_agente_con_pdfs(qdrant_url: str, qdrant_api_key: str, pdf_dir: str = None):
    """
    Inicializar agente Palimodal y procesar PDFs si es la primera vez.
    
    Args:
        qdrant_url: URL de Qdrant
        qdrant_api_key: API Key de Qdrant
        pdf_dir: Directorio con PDFs (opcional)
    
    Returns:
        PhysicsPalimodalExecutor configurado
    """
    # Crear executor
    executor = PhysicsPalimodalExecutor(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    
    # Verificar si ya existen las colecciones en Qdrant
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    colecciones_existen = False
    try:
        collection_info = await client.get_collection(executor.agent.muvera_collection)
        if collection_info.points_count and collection_info.points_count > 0:
            colecciones_existen = True
            logger.info(f"✅ Colecciones Palimodal ya existen ({collection_info.points_count} docs)")
        else:
            logger.info(f"⚠️ Colección existe pero está VACÍA (0 docs)")
            colecciones_existen = False
    except Exception as e:
        logger.info(f"⚠️ Colecciones no encontradas: {e}")
        logger.info("🔄 Se procesarán los PDFs con ColPali + MUVERA")
    
    # Si no existen colecciones, procesar PDFs en segundo plano
    if not colecciones_existen:
        # Obtener directorio de PDFs
        if pdf_dir is None:
            pdf_dir = os.getenv('PDF_DIR', '/content')  # Default Colab
        
        pdf_path = Path(pdf_dir)
        logger.info(f"📂 Buscando PDFs en: {pdf_path.absolute()}")
        
        # Buscar PDFs
        if pdf_path.exists():
            # Opción 1: Buscar arch*.pdf
            pdf_files = list(pdf_path.glob("arch*.pdf"))
            
            # Opción 2: Si no hay arch*.pdf, buscar todos los PDFs
            if not pdf_files:
                pdf_files = list(pdf_path.glob("*.pdf"))
            
            pdf_files = [str(f) for f in pdf_files]
            
            async def procesar_background(files):
                logger.info("🔄 INICIANDO PROCESAMIENTO DE PDFs EN SEGUNDO PLANO...")
                try:
                    temario = await executor.agent.procesar_y_almacenar_pdfs(files)
                    logger.info(f"✅ PDFs procesados con ColPali + MUVERA (Background Task)")
                    logger.info(f"📋 Temario preview:\n{temario[:500]}...\n")
                except Exception as e:
                    logger.error(f"❌ Error en procesamiento background: {e}", exc_info=True)

            if pdf_files:
                logger.info(f"📚 Encontrados {len(pdf_files)} archivos PDF para procesar.")
                # Lanzar tarea en background
                # Nota: En a2a el loop principal corre el server, esto debería funcionar
                # si se crea la tarea en el loop correcto.
                asyncio.create_task(procesar_background(pdf_files))
                logger.info("🚀 Tarea de procesamiento lanzada en segundo plano. El servidor iniciará de inmediato.")
            else:
                logger.warning(f"⚠️ No se encontraron archivos PDF en {pdf_dir}")
                logger.info("💡 El agente funcionará sin documentos pre-cargados")
        else:
            logger.warning(f"⚠️ Directorio de PDFs no existe: {pdf_dir}")
            logger.info("💡 Configura PDF_DIR en .env o el agente funcionará sin documentos")
    
    return executor


@click.command()
@click.option('--host', 'host', default='localhost', help='Host del servidor')
@click.option('--port', 'port', default=10004, help='Puerto del servidor')
@click.option('--pdf-dir', 'pdf_dir', default=None, help='Directorio con PDFs a procesar')
def main(host, port, pdf_dir):
    """Inicia el servidor del Asistente de Física Palimodal (ColPali + MUVERA)."""
    
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
                id='physics_palimodal_analysis',
                name='Resolución de Problemas e Imágenes de Física',
                description='Agente especializado en Física que resuelve problemas teóricos y prácticos, y analiza imágenes de experimentos, diagramas, gráficos y ecuaciones. Utiliza ColPali para embeddings multi-vector y MUVERA para búsqueda visual avanzada.',
                tags=[
                    'física', 
                    'problemas de física',
                    'análisis de imágenes',
                    'experimentos',
                    'ecuaciones',
                    'diagramas',
                    'ColPali',
                    'MUVERA',
                    'UBA'
                ],
                examples=[
                    'Resuelve este problema de movimiento parabólico',
                    '¿Qué fenómeno físico muestra esta imagen de un péndulo?',
                    'Analiza este diagrama de fuerzas y calcula la aceleración',
                    'Explica qué ocurre en esta imagen de una colisión',
                    '¿Cómo se aplica la conservación de energía en este problema?',
                    'Identifica los errores en este gráfico de velocidad vs tiempo'
                ],
            )
            
            # Crear tarjeta del agente
            agent_card = AgentCard(
                name='Agente de Física - Problemas e Imágenes',
                description='Agente inteligente especializado en Física que puede resolver problemas teóricos y prácticos, analizar imágenes de experimentos, diagramas, gráficos y ecuaciones. Envía tus consultas de física o imágenes de problemas/experimentos y recibirás explicaciones detalladas con el fundamento teórico correspondiente.',
                url=f'http://{host}:{port}/',
                version='1.0.0',
                default_input_modes=PhysicsPalimodalAgent.SUPPORTED_CONTENT_TYPES,
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
            logger.info("🔧 Inicializando agente Palimodal (ColPali + MUVERA)...")
            
            real_executor = await inicializar_agente_con_pdfs(
                qdrant_url=os.getenv('QDRANT_URL'),
                qdrant_api_key=os.getenv('QDRANT_KEY'),
                pdf_dir=pdf_dir or os.getenv('PDF_DIR')
            )
            
            logger.info("✅ Agente Palimodal inicializado correctamente")
            
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
            
            logger.info(f"📚 Iniciando Asistente Palimodal en http://{host}:{port}")
            logger.info(f"📋 Capacidades:")
            logger.info(f"   • ColPali multi-vector embeddings")
            logger.info(f"   • MUVERA búsqueda rápida + re-ranking")
            logger.info(f"   • Análisis visual de documentos")
            logger.info(f"   • Memoria conversacional")
            logger.info(f"   • Clasificación según temario UBA")
            logger.info(f"🔧 Modelo: {os.getenv('COLPALI_MODEL', 'vidore/colqwen2-v1.0')}")
            
            return server
        
        except MissingAPIKeyError as e:
            logger.error(f'❌ Error: {e}')
            logger.error('Por favor, configura las siguientes variables de entorno:')
            logger.error('  - GOOGLE_API_KEY (para Gemini)')
            logger.error('  - QDRANT_URL (URL de tu instancia Qdrant)')
            logger.error('  - QDRANT_KEY (API Key de Qdrant)')
            logger.error('Opcional:')
            logger.error('  - PDF_DIR (directorio con PDFs, default: /content)')
            logger.error('  - COLPALI_MODEL (modelo a usar, default: vidore/colqwen2-v1.0)')
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
