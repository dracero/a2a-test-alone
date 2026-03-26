# Cambio de Gemini a Hugging Face para Generación de Imágenes

## Problema

El agente de generación de imágenes usaba Gemini 2.5 Flash Image, pero Groq no tiene capacidad de generación de imágenes.

## Solución

Migrar la generación de imágenes a **Hugging Face Inference API** con **Stable Diffusion 2.1**.

## Ventajas de Hugging Face

1. **Gratuito**: API gratuita con límites generosos
2. **Sin necesidad de Gemini**: No requiere Google API Key para generación de imágenes
3. **Stable Diffusion 2.1**: Modelo de alta calidad y código abierto
4. **Fácil integración**: API REST simple

## Cambios Realizados

### 1. Modificado `samples/python/agents/images/app/agent.py`

**Antes (Gemini)**:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

response = client.models.generate_content(
    model='gemini-2.5-flash-image',
    contents=enhanced_prompt,
    config=types.GenerateContentConfig(
        temperature=0.7,
        response_modalities=['image'],
    )
)
```

**Después (Hugging Face)**:
```python
import requests

hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HF_TOKEN')

api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

payload = {
    "inputs": enhanced_prompt,
    "parameters": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": "blurry, bad quality, watermark, text, signature, low resolution"
    }
}

response = requests.post(api_url, headers=headers, json=payload, timeout=60)
image_data_bytes = response.content
```

### 2. Actualizado `samples/python/agents/images/pyproject.toml`

Agregado:
```toml
"requests>=2.31.0",
```

Removido (ya no necesario para generación de imágenes):
- `google-genai` (se mantiene por si se usa en otro lugar)

### 3. Imports actualizados

Removido:
```python
from google import genai
from google.genai import types
```

Agregado:
```python
import requests
```

## Variables de Entorno Requeridas

### Antes:
```bash
GOOGLE_API_KEY=AIza...  # Para generación de imágenes
GROQ_API_KEY=gsk_...    # Para razonamiento del agente
```

### Después:
```bash
HUGGINGFACEHUB_API_TOKEN=hf_...  # Para generación de imágenes
# o
HF_TOKEN=hf_...                   # Alternativa

GROQ_API_KEY=gsk_...              # Para razonamiento del agente
```

## Cómo Obtener Hugging Face API Token

1. Ir a https://huggingface.co/
2. Crear cuenta (gratis)
3. Ir a Settings → Access Tokens
4. Crear un nuevo token con permisos de "Read"
5. Copiar el token (empieza con `hf_`)
6. Agregarlo al `.env`:
   ```bash
   HUGGINGFACEHUB_API_TOKEN=hf_tu_token_aqui
   ```

## Modelo Usado

**Stable Diffusion 2.1** (`stabilityai/stable-diffusion-2-1`)
- Modelo de código abierto
- Alta calidad de imágenes
- Resolución: 768x768 por defecto
- Formato de salida: PNG

## Parámetros de Generación

```python
{
    "num_inference_steps": 50,      # Calidad (más pasos = mejor calidad)
    "guidance_scale": 7.5,          # Adherencia al prompt
    "negative_prompt": "blurry, bad quality, watermark, text, signature, low resolution"
}
```

## Prompt Enhancement

El prompt del usuario se mejora automáticamente:
```python
enhanced_prompt = f"{prompt}, high quality, detailed, professional, 4k, masterpiece"
```

## Manejo de Errores

### Timeout
Si el modelo está "cargando" (cold start), puede tardar hasta 60 segundos:
```
ERROR: Hugging Face API timeout - model may be loading, try again in a minute
```

### Sin Token
```
ERROR: HUGGINGFACEHUB_API_TOKEN or HF_TOKEN not set
```

### API Error
```
ERROR: Hugging Face API error: 503 - Model is loading
```

## Instalación de Dependencias

```bash
cd samples/python/agents/images
uv sync
```

Esto instalará `requests>=2.31.0` automáticamente.

## Comparación: Gemini vs Hugging Face

| Característica | Gemini 2.5 Flash Image | Hugging Face SD 2.1 |
|----------------|------------------------|---------------------|
| Costo | De pago | Gratis |
| Calidad | Excelente | Muy buena |
| Velocidad | Rápido (~2-5s) | Medio (~10-30s) |
| Resolución | Variable | 768x768 |
| Formato | JPEG | PNG |
| Cold Start | No | Sí (primera vez) |
| Límites | Por cuota | Por hora |

## Limitaciones

1. **Cold Start**: La primera generación puede tardar hasta 60 segundos mientras el modelo se carga
2. **Límites de Rate**: Hugging Face tiene límites por hora (generoso para uso gratuito)
3. **Resolución fija**: 768x768 por defecto (se puede cambiar con parámetros adicionales)
4. **Sin edición de imágenes**: Solo generación nueva (igual que antes)

## Ventajas

1. **Completamente gratuito**: No requiere tarjeta de crédito
2. **Sin dependencia de Google**: Funciona sin Google API Key
3. **Código abierto**: Modelo Stable Diffusion es open source
4. **Buena calidad**: Resultados profesionales
5. **Fácil de usar**: API REST simple

## Testing

Para probar la generación de imágenes:

1. Asegurar que `HUGGINGFACEHUB_API_TOKEN` esté en `.env`
2. Reiniciar el agente de imágenes:
   ```bash
   cd samples/python/agents/images
   uv run python -m app
   ```
3. Enviar un prompt desde el frontend:
   ```
   Generate an image of a cute cat playing with a ball
   ```

## Logs Esperados

```
🎨 Generating image for session: abc123...
🆕 Generating new image with Hugging Face Stable Diffusion
🚀 Calling Hugging Face Inference API...
📝 Enhanced prompt: a cute cat playing with a ball, high quality, detailed, professional, 4k, masterpiece
📥 Response status: 200
✅ Image data received: 245678 bytes
✅ Image generated with ID: 1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p
```

## Rollback a Gemini

Si necesitas volver a Gemini:

1. Revertir cambios en `agent.py`:
   - Restaurar imports de `google.genai`
   - Restaurar código de generación con Gemini
2. Asegurar que `GOOGLE_API_KEY` esté en `.env`
3. Reiniciar el agente

## Estado

✅ **COMPLETADO**

- ✅ Código migrado a Hugging Face
- ✅ Dependencias actualizadas
- ✅ Documentación creada
- ⏳ Pendiente: Testing con usuario

## Archivos Modificados

1. `samples/python/agents/images/app/agent.py` - Migrado a Hugging Face
2. `samples/python/agents/images/pyproject.toml` - Agregado `requests`
3. `CAMBIO-GENERACION-IMAGENES-HF.md` - Documentación (este archivo)

## Próximos Pasos

1. Agregar `HUGGINGFACEHUB_API_TOKEN` al `.env`
2. Ejecutar `uv sync` en el directorio del agente
3. Reiniciar el agente de imágenes
4. Probar generación de imágenes desde el frontend
