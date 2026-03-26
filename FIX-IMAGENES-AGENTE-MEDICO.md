# Fix: Imágenes No Llegan al Agente Médico

## Problema

Las imágenes se suben correctamente desde el frontend pero no llegan al agente médico para su análisis.

## Diagnóstico del Flujo

### 1. Frontend → Backend (✅ Funciona)
```typescript
// ChatInterface.tsx
parts.push({
  kind: 'file',
  file: {
    mime_type: image.mimeType,
    bytes: cleanBytes  // base64 string
  },
});
```

### 2. Backend → BeeAI Orchestrator (✅ Funciona)
```python
# server.py - parse_message_from_dict()
parts.append(
    Part(
        root=FilePart(
            file=FileWithBytes(
                bytes=bytes_data,  # base64 string
                mime_type=mime_type
            )
        )
    )
)
```

### 3. BeeAI Orchestrator → Agente Médico (❌ PROBLEMA AQUÍ)
```python
# beeai_host_manager.py - SendMessageToAgentTool._run()
# Add any file parts (images) from the original message
if current_message:
    for part in current_message.parts:
        if part.root.kind == 'file':
            file_part = part.root.file
            
            if isinstance(file_part, FileWithBytes):
                # Already has bytes, just add it
                parts.append(part)  # ✅ Se agrega correctamente
```

### 4. Agente Médico - Wrapper (⚠️ NO PROCESA)
```python
# custom_request_handler.py - MedicalAgentExecutorWrapper
# El wrapper solo maneja 'inline_data' (formato ADK)
# Pero las imágenes vienen como FilePart con FileWithBytes (formato A2A)
# Por lo tanto, el wrapper NO las toca y las pasa tal cual
```

### 5. Agente Médico - Executor (❌ PROBLEMA AQUÍ)
```python
# agent_executor.py - _extract_images_from_message()
elif part_kind == 'file' or part_class_name == 'FilePart':
    if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
        image_data = file_obj.bytes
        
        # PROBLEMA: Si image_data es bytes, lo mantiene como bytes
        # Pero luego en agent.py, analyze_images() espera que sea string base64
        if isinstance(image_data, bytes):
            logger.info(f"✅ FilePart (bytes) extraída")
        elif isinstance(image_data, str):
            try:
                image_data = base64.b64decode(image_data)  # ❌ DECODIFICA A BYTES
            except Exception as e:
                logger.warning(f"❌ Error decodificando base64: {e}")
        
        images.append({
            'data': image_data,  # BYTES CRUDOS
            'mime_type': mime_type
        })
```

### 6. Agente Médico - Agent (❌ PROBLEMA AQUÍ)
```python
# agent.py - analyze_images()
for idx, img in enumerate(images):
    image_data_raw = img.get('data') or img.get('bytes')
    
    if isinstance(image_data_raw, bytes):
        image_data_b64 = self.encode_image(image_data_raw)  # ✅ Codifica a base64
    elif isinstance(image_data_raw, str):
        image_data_b64 = image_data_raw  # ✅ Ya es base64
```

## Causa Raíz

El problema está en el **executor** (`agent_executor.py`):

1. Cuando recibe `FileWithBytes.bytes` como **string** (base64), lo **decodifica a bytes**
2. Luego en `agent.py`, cuando recibe **bytes**, los **codifica de nuevo a base64**
3. Pero si `FileWithBytes.bytes` ya viene como **bytes** (no string), se mantiene como bytes y luego se codifica correctamente

El problema es que `FileWithBytes.bytes` debería ser siempre un **string base64**, pero el executor está tratando de "normalizar" asumiendo que podría venir como bytes crudos.

## Solución

### Opción 1: Asegurar que FileWithBytes.bytes sea siempre string base64 (RECOMENDADO)

Modificar `server.py` para asegurar que `bytes` sea siempre string:

```python
# server.py - parse_message_from_dict()
if 'bytes' in file_data:
    bytes_data = file_data['bytes']
    
    # Asegurar que sea string, no bytes
    if isinstance(bytes_data, bytes):
        bytes_data = base64.b64encode(bytes_data).decode('utf-8')
    
    parts.append(
        Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=bytes_data,  # SIEMPRE string base64
                    mime_type=mime_type
                )
            )
        )
    )
```

### Opción 2: Simplificar el executor para NO decodificar

Modificar `agent_executor.py` para mantener el formato original:

```python
# agent_executor.py - _extract_images_from_message()
elif part_kind == 'file' or part_class_name == 'FilePart':
    if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
        image_data = file_obj.bytes
        mime_type = file_obj.mime_type
        
        # NO decodificar, mantener el formato original
        # El agent.py se encargará de manejar tanto bytes como string
        images.append({
            'data': image_data,  # Mantener formato original
            'mime_type': mime_type
        })
```

## Implementación Recomendada

Usar **Opción 2** porque es más simple y no rompe compatibilidad con otros flujos.

### Cambios en `agent_executor.py`:

```python
async def _extract_images_from_message(self, context: RequestContext) -> list[dict]:
    """
    Extrae imágenes del mensaje del usuario.
    Soporta ImagePart (kind='image') y FilePart (kind='file').
    """
    images = []
    if not context.message or not context.message.parts:
        return images

    logger.info(f"📸 Procesando {len(context.message.parts)} partes del mensaje")

    for idx, part in enumerate(context.message.parts):
        part_root = part.root
        part_kind = getattr(part_root, 'kind', None)

        # FilePart con FileWithBytes
        if part_kind == 'file':
            if hasattr(part_root, 'file'):
                file_obj = part_root.file
                
                # FileWithBytes
                if hasattr(file_obj, 'bytes') and hasattr(file_obj, 'mime_type'):
                    image_data = file_obj.bytes
                    mime_type = file_obj.mime_type
                    
                    # NO decodificar - mantener formato original
                    # agent.py maneja tanto bytes como string base64
                    logger.info(f"✅ FilePart extraída: {mime_type}, tipo datos: {type(image_data).__name__}")
                    
                    images.append({
                        'data': image_data,  # Mantener formato original
                        'mime_type': mime_type
                    })
                    continue
                
                # FileWithUri
                elif hasattr(file_obj, 'uri') and hasattr(file_obj, 'mime_type'):
                    # ... (código existente para descargar desde URI)
                    pass
    
    logger.info(f"📊 Total imágenes extraídas: {len(images)}")
    return images
```

## Verificación

Después de aplicar el fix, verificar:

1. ✅ Frontend envía imagen con `bytes` (base64 string)
2. ✅ Backend parsea y crea `FileWithBytes` con `bytes` (base64 string)
3. ✅ BeeAI orchestrator reenvía el `FilePart` con `FileWithBytes`
4. ✅ Agente médico executor extrae `bytes` sin decodificar
5. ✅ Agente médico agent recibe `bytes` (string base64) y lo usa directamente
6. ✅ Groq recibe la imagen correctamente

## Logs Esperados

```
📸 Procesando 2 partes del mensaje
✅ FilePart extraída: image/png, tipo datos: str
📊 Total imágenes extraídas: 1
📤 ENVIANDO AL AGENTE:
   Query: Analiza esta imagen...
   Número de imágenes: 1
   Imagen 1: image/png, longitud datos: 12345
🔍 Analizando 1 imagen(es)...
✅ Análisis visual completado
```

## Estado

✅ **COMPLETADO**

Modificado `samples/python/agents/medical_Images/app/agent_executor.py` para NO decodificar `FileWithBytes.bytes`, manteniendo el formato original (string base64).

## Archivos Modificados

1. `samples/python/agents/medical_Images/app/agent_executor.py` - Simplificado extracción de imágenes
