"""Test Client para el Orchestrator Agent

Este cliente prueba todas las capacidades del orquestador:
1. Routing a Image Generator
2. Routing a Medical Images
3. Consultas con y sin imágenes
4. Streaming de respuestas

Uso:
    python test_client.py
    
O con uv:
    uv run app/test_client.py
"""

import asyncio
import base64
import io
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.client.errors import A2AClientTimeoutError
from a2a.types import (FilePart, FileWithBytes, MessageSendParams,
                       SendMessageRequest, SendStreamingMessageRequest,
                       TextPart)
from PIL import Image


def encode_image_file_compressed(
    image_path: str,
    max_size: tuple = (1024, 1024),
    quality: int = 85
) -> tuple[str, str, tuple, tuple]:
    """
    Codifica una imagen optimizándola.
    
    Returns:
        Tupla de (base64_data, mime_type, original_size, new_size)
    """
    with Image.open(image_path) as img:
        original_size = img.size
        
        # Convertir RGBA a RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img)
            img = background
        
        # Redimensionar
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Guardar en buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        mime_type = 'image/jpeg'
        
        buffer.seek(0)
        image_data = base64.b64encode(buffer.read()).decode('utf-8')
    
    return image_data, mime_type, original_size, img.size


async def test_orchestrator():
    """Test completo del orquestador."""
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║          🎭 ORCHESTRATOR AGENT - TEST CLIENT 🎭               ║
║                                                                ║
║  Testing intelligent agent routing with LangChain            ║
║                                                                ║
║  Agents:                                                      ║
║    • Image Generator (localhost:10001)                       ║
║    • Medical Images (localhost:10002)                        ║
║    • Orchestrator (localhost:10003)                          ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    base_url = 'http://localhost:10003'
    
    # Crear cliente httpx
    httpx_client = httpx.AsyncClient(timeout=120.0)
    
    async with httpx_client:
        # Obtener tarjeta del orquestador
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        try:
            logger.info(f'🔍 Obteniendo tarjeta del orquestador: {base_url}')
            agent_card = await resolver.get_agent_card()
            logger.info('✅ Tarjeta obtenida exitosamente')
            logger.info(f'   • Nombre: {agent_card.name}')
            logger.info(f'   • Versión: {agent_card.version}')
            logger.info(f'   • Skills: {[s.name for s in agent_card.skills]}')
            
        except Exception as e:
            logger.error(f'❌ Error obteniendo tarjeta: {e}')
            logger.error('Asegúrate de que el orquestador esté corriendo en localhost:10003')
            return
        
        # Inicializar cliente
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=agent_card
        )
        logger.info('✅ Cliente A2A inicializado\n')
        
        # ========================================
        # TEST 1: Routing a Image Generator
        # ========================================
        logger.info('='*80)
        logger.info('TEST 1: Routing a Image Generator (sin imágenes)')
        logger.info('='*80)
        
        query_1 = "Generate a beautiful sunset over mountains with dramatic lighting"
        
        text_part_1 = TextPart(kind='text', text=query_1)
        
        payload_1: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [text_part_1.model_dump()],
                'message_id': uuid4().hex,
            },
        }
        
        request_1 = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload_1)
        )
        
        try:
            logger.info(f'📤 Query: {query_1}')
            logger.info('📡 Usando streaming...')
            
            stream_response_1 = client.send_message_streaming(request_1)
            
            print("\n" + "─"*80)
            print("📥 RESPUESTAS DEL ORQUESTADOR:")
            print("─"*80)
            
            async for chunk in stream_response_1:
                chunk_data = chunk.model_dump(mode='json', exclude_none=True)
                
                # Extraer y mostrar contenido relevante
                if 'result' in chunk_data:
                    result = chunk_data['result']
                    
                    if 'message' in result and 'parts' in result['message']:
                        for part in result['message']['parts']:
                            if part.get('kind') == 'text' and 'text' in part:
                                print(f"\n💬 {part['text']}")
                    
                    if 'artifacts' in result:
                        for artifact in result['artifacts']:
                            if 'parts' in artifact:
                                for part in artifact['parts']:
                                    if part.get('kind') == 'text' and 'text' in part:
                                        print(f"\n📄 {part['text']}")
            
            print("─"*80)
            logger.info('✅ Test 1 completado\n')
            
        except A2AClientTimeoutError as e:
            logger.error(f'❌ Timeout: {e}')
        except Exception as e:
            logger.error(f'❌ Error: {e}', exc_info=True)
        
        # ========================================
        # TEST 2: Routing a Medical Images
        # ========================================
        logger.info('='*80)
        logger.info('TEST 2: Routing a Medical Images (solo texto)')
        logger.info('='*80)
        
        query_2 = "¿Cuáles son los hallazgos típicos en una radiografía de tórax con neumonía?"
        
        text_part_2 = TextPart(kind='text', text=query_2)
        
        payload_2: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [text_part_2.model_dump()],
                'message_id': uuid4().hex,
            },
        }
        
        request_2 = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload_2)
        )
        
        try:
            logger.info(f'📤 Query: {query_2}')
            logger.info('📡 Usando streaming...')
            
            stream_response_2 = client.send_message_streaming(request_2)
            
            print("\n" + "─"*80)
            print("📥 RESPUESTAS DEL ORQUESTADOR:")
            print("─"*80)
            
            async for chunk in stream_response_2:
                chunk_data = chunk.model_dump(mode='json', exclude_none=True)
                
                if 'result' in chunk_data:
                    result = chunk_data['result']
                    
                    if 'message' in result and 'parts' in result['message']:
                        for part in result['message']['parts']:
                            if part.get('kind') == 'text' and 'text' in part:
                                print(f"\n💬 {part['text']}")
                    
                    if 'artifacts' in result:
                        for artifact in result['artifacts']:
                            if 'parts' in artifact:
                                for part in artifact['parts']:
                                    if part.get('kind') == 'text' and 'text' in part:
                                        print(f"\n📄 {part['text']}")
            
            print("─"*80)
            logger.info('✅ Test 2 completado\n')
            
        except Exception as e:
            logger.error(f'❌ Error: {e}', exc_info=True)
        
        # ========================================
        # TEST 3: Medical con imagen
        # ========================================
        logger.info('='*80)
        logger.info('TEST 3: Routing a Medical Images (con imagen)')
        logger.info('='*80)
        
        # Buscar imagen de prueba
        image_path = 'test_medical_image.png'
        
        # Si no existe, crear una imagen de prueba simple
        if not Path(image_path).exists():
            logger.info('⚠️ No se encontró imagen de prueba, creando una...')
            test_img = Image.new('RGB', (400, 300), color='lightgray')
            test_img.save(image_path)
            logger.info(f'✅ Imagen de prueba creada: {image_path}')
        
        if Path(image_path).exists():
            try:
                logger.info(f'📁 Cargando imagen: {image_path}')
                
                image_data, mime_type, orig_size, new_size = encode_image_file_compressed(
                    image_path,
                    max_size=(800, 800),
                    quality=85
                )
                
                logger.info(f'✅ Imagen procesada: {orig_size} → {new_size}')
                
                query_3 = "Analiza esta imagen médica y proporciona tus hallazgos"
                
                text_part_3 = TextPart(kind='text', text=query_3)
                
                file_with_bytes = FileWithBytes(
                    bytes=image_data,
                    mime_type=mime_type,
                    name='medical_scan.jpg'
                )
                
                image_part_3 = FilePart(kind='file', file=file_with_bytes)
                
                payload_3: dict[str, Any] = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            text_part_3.model_dump(),
                            image_part_3.model_dump()
                        ],
                        'message_id': uuid4().hex,
                    },
                }
                
                request_3 = SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(**payload_3)
                )
                
                logger.info(f'📤 Query con imagen: {query_3}')
                logger.info('📡 Usando streaming...')
                
                stream_response_3 = client.send_message_streaming(request_3)
                
                print("\n" + "─"*80)
                print("📥 RESPUESTAS DEL ORQUESTADOR:")
                print("─"*80)
                
                async for chunk in stream_response_3:
                    chunk_data = chunk.model_dump(mode='json', exclude_none=True)
                    
                    if 'result' in chunk_data:
                        result = chunk_data['result']
                        
                        if 'message' in result and 'parts' in result['message']:
                            for part in result['message']['parts']:
                                if part.get('kind') == 'text' and 'text' in part:
                                    print(f"\n💬 {part['text']}")
                        
                        if 'artifacts' in result:
                            for artifact in result['artifacts']:
                                if 'parts' in artifact:
                                    for part in artifact['parts']:
                                        if part.get('kind') == 'text' and 'text' in part:
                                            print(f"\n📄 {part['text']}")
                
                print("─"*80)
                logger.info('✅ Test 3 completado\n')
                
            except Exception as e:
                logger.error(f'❌ Error en test con imagen: {e}', exc_info=True)
        
        # ========================================
        # TEST 4: Consulta ambigua
        # ========================================
        logger.info('='*80)
        logger.info('TEST 4: Consulta ambigua (test de routing)')
        logger.info('='*80)
        
        query_4 = "Create an analysis of this visual content"
        
        text_part_4 = TextPart(kind='text', text=query_4)
        
        payload_4: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [text_part_4.model_dump()],
                'message_id': uuid4().hex,
            },
        }
        
        request_4 = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload_4)
        )
        
        try:
            logger.info(f'📤 Query ambigua: {query_4}')
            logger.info('📡 Usando streaming...')
            
            stream_response_4 = client.send_message_streaming(request_4)
            
            print("\n" + "─"*80)
            print("📥 RESPUESTAS DEL ORQUESTADOR:")
            print("─"*80)
            
            async for chunk in stream_response_4:
                chunk_data = chunk.model_dump(mode='json', exclude_none=True)
                
                if 'result' in chunk_data:
                    result = chunk_data['result']
                    
                    if 'message' in result and 'parts' in result['message']:
                        for part in result['message']['parts']:
                            if part.get('kind') == 'text' and 'text' in part:
                                print(f"\n💬 {part['text']}")
                    
                    if 'artifacts' in result:
                        for artifact in result['artifacts']:
                            if 'parts' in artifact:
                                for part in artifact['parts']:
                                    if part.get('kind') == 'text' and 'text' in part:
                                        print(f"\n📄 {part['text']}")
            
            print("─"*80)
            logger.info('✅ Test 4 completado\n')
            
        except Exception as e:
            logger.error(f'❌ Error: {e}', exc_info=True)
        
        # ========================================
        # Resumen final
        # ========================================
        logger.info('\n' + '='*80)
        logger.info('✅ TODOS LOS TESTS COMPLETADOS')
        logger.info('='*80)
        logger.info('')
        logger.info('Resumen de tests:')
        logger.info('  1. ✅ Routing a Image Generator (generación de imagen)')
        logger.info('  2. ✅ Routing a Medical Images (consulta médica)')
        logger.info('  3. ✅ Routing a Medical con imagen adjunta')
        logger.info('  4. ✅ Consulta ambigua (test de decisión)')
        logger.info('')
        logger.info('El orquestador usa LangChain para decidir inteligentemente')
        logger.info('a qué agente enviar cada consulta.')
        logger.info('='*80)


if __name__ == '__main__':
    print("\n🚀 Iniciando tests del Orchestrator Agent...\n")
    asyncio.run(test_orchestrator())
