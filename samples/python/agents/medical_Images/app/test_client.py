import asyncio
import base64
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (AgentCard, MessageSendParams, SendMessageRequest,
                       SendStreamingMessageRequest)
from a2a.utils.constants import (AGENT_CARD_WELL_KNOWN_PATH,
                                 EXTENDED_AGENT_CARD_PATH)


def encode_image_file(image_path: str) -> tuple[str, str]:
    """
    Codifica una imagen desde un archivo.
    
    Returns:
        Tupla de (base64_data, mime_type)
    """
    path = Path(image_path)
    
    # Determinar MIME type
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }
    mime_type = mime_types.get(path.suffix.lower(), 'image/png')
    
    # Leer y codificar
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    return image_data, mime_type


async def main() -> None:
    """Cliente de prueba para el Asistente Médico."""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    base_url = 'http://localhost:10001'
    
    async with httpx.AsyncClient() as httpx_client:
        # Inicializar resolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        # Obtener tarjeta del agente
        final_agent_card_to_use: AgentCard | None = None
        
        try:
            logger.info(
                f'Obteniendo tarjeta del agente desde: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = await resolver.get_agent_card()
            logger.info('✅ Tarjeta del agente obtenida exitosamente')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\n✅ Usando tarjeta pública del agente.'
            )
            
            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        '\nLa tarjeta pública soporta tarjeta extendida autenticada. '
                        'Intentando obtener desde: '
                        f'{base_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Tarjeta extendida autenticada obtenida exitosamente:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = _extended_card
                    logger.info(
                        '\n✅ Usando tarjeta EXTENDIDA autenticada.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'No se pudo obtener tarjeta extendida: {e_extended}. '
                        'Usando tarjeta pública.',
                        exc_info=True,
                    )
            elif _public_card:
                logger.info(
                    '\nLa tarjeta pública no indica soporte para tarjeta extendida. '
                    'Usando tarjeta pública.'
                )
        
        except Exception as e:
            logger.error(
                f'❌ Error crítico obteniendo tarjeta pública: {e}', 
                exc_info=True
            )
            raise RuntimeError(
                'No se pudo obtener la tarjeta pública del agente. No se puede continuar.'
            ) from e
        
        # Inicializar cliente
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=final_agent_card_to_use
        )
        logger.info('✅ Cliente A2A inicializado.')
        
        # --- EJEMPLO 1: Consulta solo con texto ---
        logger.info('\n' + '='*80)
        logger.info('EJEMPLO 1: Consulta médica solo texto')
        logger.info('='*80)
        
        text_only_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': '¿Cuáles son los síntomas comunes de la neumonía y cómo se diferencia de un resfriado común?'
                    }
                ],
                'message_id': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**text_only_payload)
        )
        
        try:
            response = await client.send_message(request)
            logger.info('📩 Respuesta recibida (texto):')
            print(response.model_dump(mode='json', exclude_none=True))
        except Exception as e:
            logger.error(f'❌ Error en consulta de texto: {e}', exc_info=True)
        
        # --- EJEMPLO 2: Consulta con imagen ---
        logger.info('\n' + '='*80)
        logger.info('EJEMPLO 2: Consulta médica con imagen')
        logger.info('='*80)
        
        # Ruta a tu imagen de ejemplo (ajusta según tu archivo)
        image_path = '/media/dracero/08c67654-6ed7-4725-b74e-50f29ea60cb2/pythonAI-Others/a2a-samples/samples/python/agents/medical_Images/app/imagen.png'
        
        if Path(image_path).exists():
            try:
                image_data, mime_type = encode_image_file(image_path)
                
                image_payload: dict[str, Any] = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {
                                'kind': 'text',
                                'text': 'Analiza esta imagen médica y proporciona tus hallazgos principales.'
                            },
                            {
                                'kind': 'image',
                                'data': image_data,
                                'mime_type': mime_type
                            }
                        ],
                        'message_id': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**image_payload)
                )
                
                response = await client.send_message(request)
                logger.info('📩 Respuesta con imagen recibida:')
                print(response.model_dump(mode='json', exclude_none=True))
                        
            except Exception as e:
                logger.error(f'❌ Error en consulta con imagen: {e}', exc_info=True)
        else:
            logger.warning(f'⚠️ Archivo de imagen no encontrado: {image_path}')
            logger.info('💡 Coloca una imagen médica en el directorio con ese nombre para probar esta funcionalidad.')
        
        # --- EJEMPLO 3: Consulta con streaming ---
        logger.info('\n' + '='*80)
        logger.info('EJEMPLO 3: Consulta con streaming')
        logger.info('='*80)
        
        streaming_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {
                        'kind': 'text',
                        'text': '¿Cuáles son las indicaciones para solicitar una tomografía de tórax?'
                    }
                ],
                'message_id': uuid4().hex,
            },
        }
        
        streaming_request = SendStreamingMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**streaming_payload)
        )
        
        try:
            logger.info('📡 Iniciando streaming...')
            stream_response = client.send_message_streaming(streaming_request)
            
            async for chunk in stream_response:
                logger.info(f'📦 Chunk recibido:')
                print(chunk.model_dump(mode='json', exclude_none=True))
        except Exception as e:
            logger.error(f'❌ Error en streaming: {e}', exc_info=True)
        
        logger.info('\n' + '='*80)
        logger.info('✅ Todas las pruebas completadas')
        logger.info('='*80)


if __name__ == '__main__':
    asyncio.run(main())
