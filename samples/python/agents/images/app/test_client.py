"""Test client para el Image Generator Agent.

Este cliente prueba la generación de imágenes y guarda el resultado en base64.
Uso:
    python test_client.py
    
O con uv:
    uv run app/test_client.py
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.client.errors import A2AClientTimeoutError
from a2a.types import (AgentCard, MessageSendParams, SendMessageRequest,
                       TextPart)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


def save_base64_image(base64_data: str, output_path: str, mime_type: str = 'image/png'):
    """
    Guarda una imagen desde base64 a un archivo.
    
    Args:
        base64_data: String con los datos en base64
        output_path: Ruta donde guardar la imagen
        mime_type: Tipo MIME de la imagen
    """
    try:
        # Decodificar base64
        image_bytes = base64.b64decode(base64_data)
        
        # Determinar extensión
        extensions = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/webp': '.webp',
            'image/gif': '.gif'
        }
        
        ext = extensions.get(mime_type, '.png')
        output_file = Path(output_path)
        
        # Agregar extensión si no la tiene
        if not output_file.suffix:
            output_file = output_file.with_suffix(ext)
        
        # Guardar archivo
        output_file.write_bytes(image_bytes)
        
        file_size = len(image_bytes)
        print(f'✅ Imagen guardada: {output_file}')
        print(f'   • Tamaño: {file_size:,} bytes ({file_size / 1024:.2f} KB)')
        print(f'   • Tipo: {mime_type}')
        
        return str(output_file)
        
    except Exception as e:
        print(f'❌ Error guardando imagen: {e}')
        return None


async def main() -> None:
    """Cliente de prueba para el Image Generator Agent."""
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    base_url = 'http://localhost:10001'
    
    # Crear cliente httpx con timeout largo
    httpx_client = httpx.AsyncClient(timeout=120.0)
    
    async with httpx_client:
        # Obtener tarjeta del agente
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        try:
            logger.info(f'🔍 Obteniendo tarjeta del agente desde: {base_url}')
            agent_card = await resolver.get_agent_card()
            logger.info('✅ Tarjeta del agente obtenida')
            logger.info(f'   • Nombre: {agent_card.name}')
            logger.info(f'   • Versión: {agent_card.version}')
            logger.info(f'   • Skills: {[s.name for s in agent_card.skills]}')
            
        except Exception as e:
            logger.error(f'❌ Error obteniendo tarjeta: {e}')
            raise
        
        # Inicializar cliente A2A
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=agent_card
        )
        logger.info('✅ Cliente A2A inicializado\n')
        
        # --- TEST 1: Generar estudiante UBA millonario ---
        logger.info('='*80)
        logger.info('TEST 1: Genera una imagen de un estudiante de la UBA que se hizo millonario')
        logger.info('='*80)
        
        prompt = "Generate a photorealistic image of a successful young Argentine university student who became a millionaire entrepreneur. The student should be wearing casual business attire, standing in front of the iconic UBA (Universidad de Buenos Aires) building with its neoclassical columns. The person should look confident and ambitious, holding a laptop or smartphone. Background shows the historic faculty building in Buenos Aires. Professional lighting, modern and aspirational atmosphere."
        
        text_part = TextPart(
            kind='text',
            text=prompt
        )
        
        payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [text_part.model_dump()],
                'message_id': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload)
        )
        
        try:
            logger.info(f'📤 Enviando prompt: {prompt}')
            logger.info('⏳ Generando imagen... (esto puede tomar un momento)')
            
            response = await client.send_message(request)
            
            logger.info('✅ Respuesta recibida')
            
            # Guardar respuesta completa para debugging
            response_data = response.model_dump(mode='json', exclude_none=True)
            
            # Buscar la imagen en los artifacts
            image_saved = False
            if 'result' in response_data:
                result = response_data['result']
                
                if 'artifacts' in result:
                    for idx, artifact in enumerate(result['artifacts']):
                        if 'parts' in artifact:
                            for part_idx, part in enumerate(artifact['parts']):
                                # Buscar FilePart con imagen
                                if part.get('kind') == 'file':
                                    file_data = part.get('file', {})
                                    
                                    if 'bytes' in file_data:
                                        base64_data = file_data['bytes']
                                        mime_type = file_data.get('mime_type', 'image/png')
                                        
                                        # Guardar imagen
                                        output_path = f'estudiante_uba_millonario_{idx}_{part_idx}'
                                        saved_path = save_base64_image(
                                            base64_data, 
                                            output_path, 
                                            mime_type
                                        )
                                        
                                        if saved_path:
                                            image_saved = True
                                            logger.info(f'🖼️  Imagen guardada en: {saved_path}')
                                        
                                        # También guardar el base64 en un archivo de texto
                                        base64_file = Path(saved_path).with_suffix('.txt')
                                        base64_file.write_text(base64_data)
                                        logger.info(f'📝 Base64 guardado en: {base64_file}')
            
            if not image_saved:
                logger.warning('⚠️  No se encontró imagen en la respuesta')
                # Guardar respuesta completa para debugging
                debug_file = Path('debug_response.json')
                debug_file.write_text(json.dumps(response_data, indent=2))
                logger.info(f'📋 Respuesta completa guardada en: {debug_file}')
            
            # Mostrar estructura de la respuesta
            logger.info('\n📊 Estructura de la respuesta:')
            print(json.dumps(response_data, indent=2))
            
        except A2AClientTimeoutError as e:
            logger.error(f'❌ Timeout: {e}')
        except Exception as e:
            logger.error(f'❌ Error: {e}', exc_info=True)
        
        # --- TEST 2: Editar la imagen anterior ---
        logger.info('\n' + '='*80)
        logger.info('TEST 2: Editar imagen - agregar elementos de éxito empresarial')
        logger.info('='*80)
        
        edit_prompt = "Add a luxury sports car in the background and make the lighting more dramatic and cinematic, emphasizing success and achievement"
        
        text_part_edit = TextPart(
            kind='text',
            text=edit_prompt
        )
        
        payload_edit: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [text_part_edit.model_dump()],
                'message_id': uuid4().hex,
            },
        }
        
        request_edit = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload_edit)
        )
        
        try:
            logger.info(f'📤 Enviando edición: {edit_prompt}')
            logger.info('⏳ Editando imagen...')
            
            response_edit = await client.send_message(request_edit)
            
            logger.info('✅ Imagen editada recibida')
            
            # Guardar imagen editada
            response_edit_data = response_edit.model_dump(mode='json', exclude_none=True)
            
            if 'result' in response_edit_data:
                result = response_edit_data['result']
                
                if 'artifacts' in result:
                    for idx, artifact in enumerate(result['artifacts']):
                        if 'parts' in artifact:
                            for part_idx, part in enumerate(artifact['parts']):
                                if part.get('kind') == 'file':
                                    file_data = part.get('file', {})
                                    
                                    if 'bytes' in file_data:
                                        base64_data = file_data['bytes']
                                        mime_type = file_data.get('mime_type', 'image/png')
                                        
                                        output_path = f'estudiante_uba_millonario_edited_{idx}_{part_idx}'
                                        saved_path = save_base64_image(
                                            base64_data, 
                                            output_path, 
                                            mime_type
                                        )
                                        
                                        if saved_path:
                                            logger.info(f'🖼️  Imagen editada guardada en: {saved_path}')
            
            print(json.dumps(response_edit_data, indent=2))
            
        except Exception as e:
            logger.error(f'⚠️  Edición no disponible o error: {e}')
        
        logger.info('\n' + '='*80)
        logger.info('✅ Tests completados')
        logger.info('='*80)


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════╗
║         🎨 IMAGE GENERATOR AGENT - TEST CLIENT 🎨             ║
║                                                                ║
║  Este cliente prueba la generación de imágenes y las          ║
║  guarda en archivos para visualización.                       ║
║                                                                ║
║  Test: Estudiante de la UBA que se hizo millonario 💰🎓      ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
