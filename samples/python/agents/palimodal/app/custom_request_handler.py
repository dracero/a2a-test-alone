# samples/python/agents/palimodal/app/custom_request_handler.py

"""
Custom Request Handler para el Asistente de Física Palimodal.

Este handler maneja correctamente las imágenes que vienen desde el host ADK 
en formato inline_data.
"""


import base64
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import FilePart, FileWithBytes, Message, Part, TextPart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsAgentExecutorWrapper(AgentExecutor):
    """
    Wrapper que pre-procesa mensajes antes de pasar al executor real.
    
    Convierte 'inline_data' (formato del ADK) a 'FilePart' (formato A2A).
    """

    def __init__(self, wrapped_executor: AgentExecutor):
        """
        Inicializa el wrapper.
        
        Args:
            wrapped_executor: El executor real (PhysicsAgentExecutor)
        """
        self.wrapped_executor = wrapped_executor

    def _preprocess_message(self, message: Message) -> Message:
        """
        Pre-procesa el mensaje para convertir inline_data a FilePart.
        
        El host ADK envía imágenes como:
        {
            'inline_data': {
                'data': bytes,
                'mime_type': 'image/png'
            }
        }
        
        El agente espera:
        Part(root=FilePart(file=FileWithBytes(bytes=base64_str, mime_type='image/png')))
        """
        if not message or not message.parts:
            logger.warning("⚠️ Mensaje vacío o sin partes")
            return message

        logger.info(f"🔍 Pre-procesando mensaje con {len(message.parts)} partes")

        new_parts: list[Part] = []

        for idx, part in enumerate(message.parts):
            # Obtener dict de la parte
            if hasattr(part, 'model_dump'):
                part_dict = part.model_dump()
            elif hasattr(part, 'dict'):
                part_dict = part.dict()
            elif isinstance(part, dict):
                part_dict = part
            else:
                part_dict = {}

            logger.info(f"📦 Parte {idx} dict keys: {list(part_dict.keys()) if part_dict else 'empty'}")

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
                        logger.info(f"✅ inline_data: Bytes → base64 ({len(image_data_b64)} chars)")
                    elif isinstance(image_bytes, str):
                        image_data_b64 = image_bytes
                        logger.info(f"✅ inline_data: Base64 ya es string ({len(image_data_b64)} chars)")
                    else:
                        logger.warning(f"⚠️ Tipo de bytes desconocido: {type(image_bytes)}")
                        continue

                    # Crear FilePart con FileWithBytes
                    file_part = FilePart(
                        file=FileWithBytes(
                            bytes=image_data_b64,
                            mime_type=mime_type,
                            name=f'physics_image_{idx}.{mime_type.split("/")[-1]}'
                        )
                    )

                    new_parts.append(Part(root=file_part))
                    logger.info(f"✅ inline_data → FilePart: {mime_type}")

                except Exception as e:
                    logger.error(f"❌ Error procesando inline_data: {e}", exc_info=True)
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

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Intercepta execute() para pre-procesar el mensaje.
        
        IMPORTANTE: No modificamos el contexto porque es inmutable.
        En su lugar, procesamos el mensaje y dejamos que el wrapped_executor
        lo maneje directamente.
        """
        logger.info("="*80)
        logger.info("🔧 WRAPPER: Interceptando mensaje")
        logger.info("="*80)

        # Pre-procesar mensaje si existe
        if context.message:
            original_parts = len(context.message.parts) if context.message.parts else 0
            processed_message = self._preprocess_message(context.message)
            new_parts = len(processed_message.parts) if processed_message.parts else 0
            logger.info(f"📊 Transformación: {original_parts} → {new_parts} partes")
            
            # 🔧 SOLUCIÓN: Modificar las partes in-place
            # El contexto es inmutable, pero la lista de partes sí es mutable
            context.message.parts.clear()
            context.message.parts.extend(processed_message.parts)
            logger.info("✅ Partes del mensaje actualizadas in-place")
        else:
            logger.warning("⚠️ Contexto sin mensaje")

        # Llamar al executor real
        logger.info("📤 Pasando al executor real...")
        await self.wrapped_executor.execute(context, event_queue)
        logger.info("✅ Executor real completado")

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Delegar cancel al executor real."""
        logger.info("⚠️ Cancelación solicitada")
        await self.wrapped_executor.cancel(context, event_queue)
