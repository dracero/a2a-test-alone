import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root at the very beginning (6 levels up)
root_dir = Path(__file__).resolve().parents[5]
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

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
from app.custom_request_handler import MedicalAgentExecutorWrapper
from app.langsmith_config import setup_langsmith_environment, get_langsmith_status

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
        # Configurar e informar estado de LangSmith
        setup_langsmith_environment("a2a-medical-assistant")
        ls_status = get_langsmith_status()
        if ls_status.get("enabled"):
            logger.info(f"📊 LangSmith Monitoring: ENABLED (Project: {ls_status.get('project')})")
        else:
            logger.info("📊 LangSmith Monitoring: DISABLED")

        # Verificar Groq API Key
        if not os.getenv('GROQ_API_KEY'):
            raise MissingAPIKeyError(
                'GROQ_API_KEY environment variable not set.'
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
        
        # ✨ NUEVO: Crear executor con wrapper para manejar inline_data
        logger.info("🔧 Inicializando executor con soporte de inline_data...")
        
        # Executor real
        real_executor = MedicalAgentExecutor()
        
        # Wrapper que pre-procesa inline_data → FilePart
        wrapped_executor = MedicalAgentExecutorWrapper(real_executor)
        
        # Request handler con el executor envuelto
        request_handler = DefaultRequestHandler(
            agent_executor=wrapped_executor,
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        
        logger.info("✅ Executor wrapper configurado correctamente")
        
        # Mount static directory for images
        os.makedirs("histopatologia_data", exist_ok=True)
        
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def lifespan_handler(app):
            logger.info("🚀 Iniciando backend y cargando modelos para MedicalAgent...")
            
            # Limpiar uploads al inicio
            uploads_dir = Path("uploads")
            if uploads_dir.exists():
                logger.info(f"🧹 Limpiando directorio {uploads_dir}...")
                for file in uploads_dir.glob("*"):
                    if file.is_file():
                        try:
                            file.unlink()
                        except:
                            pass
            
            # Inicializar componentes del agente
            real_executor.agent.inicializar_componentes()
            logger.info("✅ Modelos del agente cargados.")
            
            # Indexación automática de PDFs si la colección está vacía
            try:
                pdf_dir = Path("./pdfs")
                if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
                    # Fallback a multimodal PDF usando root_dir
                    multi_pdfs = root_dir / "samples" / "python" / "agents" / "multimodal" / "PDF"
                    if multi_pdfs.exists() and list(multi_pdfs.glob("*.pdf")):
                        pdf_dir = multi_pdfs
                        
                pdfs = list(pdf_dir.glob("*.pdf"))
                if pdfs:
                    logger.info(f"📦 Se encontraron {len(pdfs)} PDFs en {pdf_dir}. Verificando indexación...")
                    client = real_executor.agent.qdrant_client
                    col_name = real_executor.agent.gestor_qdrant.content_mv_collection
                    
                    try:
                        count = await client.count(col_name)
                        if count.count == 0:
                            logger.info("⚠️ Colección vacía. Iniciando indexación automática de PDFs...")
                            await real_executor.agent.procesar_pdfs([str(f) for f in pdfs])
                        else:
                            logger.info(f"✅ Colección {col_name} tiene {count.count} documentos. Saltando indexación.")
                    except Exception as e:
                        logger.info(f"⚠️ Colección no encontrada o error ({e}). Iniciando indexación automática...")
                        await real_executor.agent.procesar_pdfs([str(f) for f in pdfs])
                else:
                    logger.warning("⚠️ No se encontraron PDFs para indexar. Por favor, coloca PDFs en './pdfs' o en 'samples/python/agents/multimodal/PDF'.")
            except Exception as e:
                logger.error(f"❌ Error en auto-indexación: {e}", exc_info=True)
            
            yield
            logger.info("🛑 Deteniendo backend...")

        # Crear aplicación con lifespan
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )
        app = server.build(lifespan=lifespan_handler)
        
        # Mount static directory for images
        from starlette.staticfiles import StaticFiles
        app.mount("/histopatologia_data", StaticFiles(directory="histopatologia_data"), name="histopatologia_data")
        
        logger.info(f"🏥 Iniciando Asistente Médico en http://{host}:{port}")
        logger.info(f"📋 Capacidades: Colpali RAG, Fallback Búsqueda médica, Memoria conversacional")
        logger.info(f"🔧 Modo: Executor con wrapper para inline_data y StaticFiles montado")
        
        # Ejecutar servidor
        uvicorn.run(app, host=host, port=port)
    
    except MissingAPIKeyError as e:
        logger.error(f'❌ Error: {e}')
        logger.error('Por favor, configura las siguientes variables de entorno:')
        logger.error('  - GROQ_API_KEY (para Llama 4)')
        logger.error('  - TAVILY_API_KEY (para búsqueda médica)')
        sys.exit(1)
    except Exception as e:
        logger.error(f'❌ Error durante el inicio del servidor: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
