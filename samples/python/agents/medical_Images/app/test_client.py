import base64
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (AgentCard, MessageSendParams, SendMessageRequest,
                       SendStreamingMessageRequest)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


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
        try:
            logger.info(f'Obteniendo tarjeta del agente desde: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}')
            agent_card = await resolver.get_agent_card()
            logger.info('✅ Tarjeta del agente obtenida exitosamente')
            logger.info(f'Agente: {agent_card.name}')
            logger.info(f'Descripción: {agent_card.description}')
        except Exception as e:
            logger.error(f'❌ Error obteniendo tarjeta del agente: {e}')
            raise
        
        # Inicializar cliente
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=agent_card
        )
        logger.info('✅ Cliente A2A inicializado')
        
        # --- EJEMPLO 1: Consulta solo con texto ---
        logger.info('\n' + '='*80)
        logger.info('EJEMPLO 1: Consulta médica solo texto')
        logger.info('='*80)
        
        text_only_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts':[
                    {
                        'type': 'text',
                        'text': '¿Cuáles son los síntomas comunes de la gripe?'
                    }
                ]
            }
        }
        
        try:
            response = await client.send_message(text_only_payload)
            logger.info('📩 Respuesta recibida:')
            for part in response.message.parts:
                if part.get('type') == 'text':
                    logger.info(f'{part.get("text")}')
        except Exception as e:
            logger.error(f'❌ Error en consulta de texto: {e}')
        
        # --- EJEMPLO 2: Consulta con imagen ---
        logger.info('\n' + '='*80)
        logger.info('EJEMPLO 2: Consulta médica con imagen')
        logger.info('='*80)
        
        # Ruta a tu imagen de ejemplo
        image_path = 'ejemplo_sintoma.jpg'  # Cambiar por tu archivo
        
        try:
            image_data, mime_type = encode_image_file(image_path)
            
            image_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {
                            'type': 'text',
                            'text': '¿Qué puedes decirme sobre esta imagen médica?'
                        },
                        {
                            'type': 'image',
                            'image': {
                                'format': mime_type,
                                'data': image_data
                            }
                        }
                    ]
                }
            }
            
            response = await client.send_message(image_payload)
            logger.info('📩 Respuesta con imagen recibida:')
            for part in response.message.parts:
                if part.get('type') == 'text':
                    logger.info(f'{part.get("text")}')
                    
        except FileNotFoundError:
            logger.warning(f'⚠️ Archivo de imagen no encontrado: {image_path}')
        except Exception as e:
            logger.error(f'❌ Error en consulta con imagen: {e}')
        
        logger.info('\n' + '='*80)
        logger.info('✅ Pruebas completadas')
        logger.info('='*80)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
