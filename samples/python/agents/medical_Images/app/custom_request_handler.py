"""
Custom Request Handler para el Asistente Médico.

Este handler extiende el DefaultRequestHandler para manejar correctamente
las imágenes que vienen desde el host ADK en formato inline_data.
"""

import base64
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import FilePart, FileWithBytes, Message, Part, TextPart
from app.langsmith_config import traceable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalAgentExecutorWrapper(AgentExecutor):
    """
    Wrapper que pre-procesa mensajes antes de pasar al executor real.
    
    Este wrapper intercepta los mensajes que vienen del host ADK y convierte
    las partes con 'inline_data' (formato del ADK) a 'FilePart' (formato A2A)
    que el agente médico puede procesar.
    """

    def __init__(self, wrapped_executor: AgentExecutor):
        """
        Inicializa el wrapper.
        
        Args:
            wrapped_executor: El executor real (MedicalAgentExecutor) que procesará el mensaje
        """
        self.wrapped_executor = wrapped_executor

    def _preprocess_message(self, message: Message) -> Message:
        """
        Pre-procesa el mensaje para convertir inline_data a FilePart.
        
        El host ADK envía imágenes como Part con inline_data:
        {
            'inline_data': {
                'data': bytes,
                'mime_type': 'image/png'
            }
        }
        
        Pero el agente espera FilePart:
        Part(root=FilePart(file=FileWithBytes(bytes=base64_str, mime_type='image/png')))
        
        Args:
            message: Mensaje original del host
            
        Returns:
            Mensaje con inline_data convertido a FilePart
        """
        if not message or not message.parts:
            logger.warning("⚠️ Mensaje vacío o sin partes")
            return message

        logger.info(f"🔍 Pre-procesando mensaje con {len(message.parts)} partes")

        new_parts: list[Part] = []

        for idx, part in enumerate(message.parts):
            # Intentar obtener el dict de la parte
            if hasattr(part, 'model_dump'):
                part_dict = part.model_dump()
            elif hasattr(part, 'dict'):
                part_dict = part.dict()
            elif isinstance(part, dict):
                part_dict = part
            else:
                part_dict = {}

            logger.info(f"📦 Parte {idx} dict keys: {part_dict.keys() if part_dict else 'empty'}")

            # CASO 1: inline_data (viene del host ADK con imágenes)
            if 'inline_data' in part_dict:
                inline_data = part_dict['inline_data']
                logger.info(f"🖼️ Detectado inline_data: {type(inline_data)}")

                try:
                    # Extraer bytes y mime_type
                    if isinstance(inline_data, dict):
                        image_bytes = inline_data.get('data')
                        mime_type = inline_data.get('mime_type', 'image/png')
                    else:
                        image_bytes = getattr(inline_data, 'data', None)
                        mime_type = getattr(inline_data, 'mime_type', 'image/png')

                    if image_bytes is None:
                        logger.warning(f"⚠️ inline_data sin 'data'")
                        continue

                    # Convertir a base64 si son bytes
                    if isinstance(image_bytes, bytes):
                        image_data_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        logger.info(f"✅ Bytes convertidos a base64: {len(image_data_b64)} chars")
                    elif isinstance(image_bytes, str):
                        # Ya es base64
                        image_data_b64 = image_bytes
                        logger.info(f"✅ Base64 string mantenido: {len(image_data_b64)} chars")
                    else:
                        logger.warning(f"⚠️ Tipo de bytes desconocido: {type(image_bytes)}")
                        continue

                    # Crear FilePart con FileWithBytes
                    file_part = FilePart(
                        file=FileWithBytes(
                            bytes=image_data_b64,
                            mime_type=mime_type,
                            name=f'image_{idx}.{mime_type.split("/")[-1]}'
                        )
                    )

                    new_parts.append(Part(root=file_part))
                    logger.info(f"✅ inline_data → FilePart: {mime_type}")

                except Exception as e:
                    logger.error(f"❌ Error procesando inline_data: {e}", exc_info=True)
                    # Mantener la parte original si falla
                    new_parts.append(part)

            # CASO 2: text (TextPart estándar)
            elif 'text' in part_dict:
                new_parts.append(part)
                logger.info(f"✅ TextPart mantenido")

            # CASO 3: root (Part estándar con root.kind)
            elif 'root' in part_dict:
                root = part_dict['root']
                kind = root.get('kind') if isinstance(root, dict) else getattr(root, 'kind', None)
                logger.info(f"✅ Part con root.kind='{kind}' mantenido")
                new_parts.append(part)

            # CASO 4: Part válido (objeto Part)
            elif isinstance(part, Part):
                new_parts.append(part)
                logger.info(f"✅ Part objeto mantenido")

            # CASO 5: Desconocido
            else:
                logger.warning(f"⚠️ Parte {idx} desconocida: {part_dict}")
                new_parts.append(part)

        # Crear nuevo mensaje con partes procesadas
        # Solo incluir campos que existen en la clase Message
        processed_message = Message(
            role=message.role,
            parts=new_parts,
            message_id=message.message_id,
            context_id=message.context_id,
            task_id=message.task_id,
            metadata=message.metadata,
        )

        logger.info(f"✅ Mensaje pre-procesado: {len(message.parts)} → {len(new_parts)} partes")
        return processed_message

    @traceable(name="medical_a2a_wrapper_execution", run_type="chain", tags=["agent_type:medical_assistant", "medical_assistant", "a2a-agent"])
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Intercepta execute() para pre-procesar el mensaje antes de pasarlo al executor real.
        
        Args:
            context: Contexto de la request con el mensaje original
            event_queue: Cola de eventos para streaming
        """
        logger.info("="*80)
        logger.info("🔧 WRAPPER: Interceptando mensaje antes de executor real")
        logger.info("="*80)

        # Pre-procesar el mensaje si existe
        if context.message:
            original_parts = len(context.message.parts) if context.message.parts else 0
            processed_message = self._preprocess_message(context.message)
            new_parts = len(processed_message.parts) if processed_message.parts else 0
            logger.info(f"📊 Transformación: {original_parts} → {new_parts} partes")
            
            # CRÍTICO: Modificar el mensaje directamente en el contexto usando setattr
            # Esto es un hack, pero RequestContext no permite crear instancias nuevas
            try:
                # Intentar modificar el mensaje usando object.__setattr__ (bypass de property)
                object.__setattr__(context, '_message', processed_message)
                logger.info("✅ Mensaje modificado exitosamente en el contexto")
            except Exception as e:
                logger.error(f"❌ Error modificando mensaje: {e}")
                logger.warning("⚠️ Usando contexto original sin modificar")
        else:
            logger.warning("⚠️ Contexto sin mensaje")

        # Llamar al executor real con el contexto (modificado o no)
        logger.info("📤 Pasando al executor real...")
        try:
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            logger.info("🧠 VRAM pre-cleaned before agent execution")
        except Exception as cleanup_err:
            logger.warning(f"⚠️ Error cleaning VRAM before agent execution: {cleanup_err}")

        try:
            await self.wrapped_executor.execute(context, event_queue)
        finally:
            logger.info("✅ Executor real completado")
            try:
                import gc
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                logger.info("🧠 VRAM post-cleaned after agent execution")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Error cleaning VRAM after agent execution: {cleanup_err}")

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Delegar cancel al executor real.
        
        Args:
            context: Contexto de la request
            event_queue: Cola de eventos
        """
        logger.info("⚠️ Cancelación solicitada, delegando al executor real")
        await self.wrapped_executor.cancel(context, event_queue)
