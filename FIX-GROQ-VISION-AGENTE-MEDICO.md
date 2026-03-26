# Fix: Groq Vision para Agente Médico

## Problema

El agente médico intentaba analizar imágenes con Groq Llama 4 Scout, pero el formato de mensaje no era correcto, causando error 400:

```
Error code: 400 - {'error': {'message': "'messages.0' : for 'role:user' the following must be satisfied..."}}
```

## Causa

El formato de `image_url` en el mensaje no era el correcto para Groq. Groq requiere que `image_url` sea un objeto con una propiedad `url`, no un string directo.

## Solución

Actualizado el formato del mensaje para que sea compatible con Groq Vision API.

### Antes (Incorrecto):
```python
content.append({
    "type": "image_url",
    "image_url": f"data:{mime_type};base64,{image_data_b64}"  # ❌ String directo
})
```

### Después (Correcto):
```python
content.append({
    "type": "image_url",
    "image_url": {  # ✅ Objeto con propiedad 'url'
        "url": f"data:{mime_type};base64,{image_data_b64}"
    }
})
```

## Modelo Usado

**Groq Llama 4 Scout 17B** (`meta-llama/llama-4-scout-17b-16e-instruct`)
- Modelo multimodal con capacidad de visión
- Soporta texto e imágenes
- Límites:
  - Máximo 5 imágenes por request
  - Máximo 20MB por imagen (URL)
  - Máximo 4MB por imagen (base64)
  - Máximo 33 megapixels por imagen

## Formato Correcto de Mensaje

```python
from langchain_core.messages import HumanMessage

content = [
    {
        "type": "text",
        "text": "Analiza esta imagen médica..."
    },
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_string}"
        }
    }
]

message = HumanMessage(content=content)
response = llm.invoke([message])
```

## Cambios Realizados

### 1. Actualizado `samples/python/agents/medical_Images/app/agent.py`

**Método `analyze_images()`**:
- Cambiado formato de `image_url` de string a objeto
- Actualizado comentarios para reflejar uso de Groq
- Removida referencia a Gemini

### 2. Removido `__init__` con dos modelos

**Antes**:
```python
self.llm = ChatGroq(...)  # Para razonamiento
self.vision_llm = ChatGoogleGenerativeAI(...)  # Para visión
```

**Después**:
```python
self.llm = ChatGroq(...)  # Para todo (razonamiento + visión)
```

### 3. Actualizado `pyproject.toml`

Removido:
- `langchain-google-genai>=2.0.8`

Mantenido:
- `langchain-groq>=0.2.0`

### 4. Actualizado `__main__.py`

Removida validación de `GOOGLE_API_KEY` (ya no necesaria).

## Variables de Entorno Requeridas

```bash
# Groq API Key (para razonamiento Y análisis de imágenes)
GROQ_API_KEY=gsk_...

# Tavily API Key (para búsqueda médica)
TAVILY_API_KEY=tvly_...
```

## Ventajas

✅ **Un solo modelo**: Groq Llama 4 Scout para todo
✅ **Sin Google API Key**: No requiere cuenta de Google
✅ **Más rápido**: Groq es muy rápido
✅ **Gratis**: Groq tiene tier gratuito generoso
✅ **Multimodal nativo**: Soporta texto e imágenes en el mismo modelo

## Limitaciones

⚠️ **Calidad de visión**: Llama 4 Scout puede ser menos preciso que Gemini para análisis médico complejo
⚠️ **Límite de imágenes**: Máximo 5 imágenes por request
⚠️ **Tamaño de imagen**: Máximo 4MB por imagen en base64

## Testing

Para probar el análisis de imágenes:

1. Asegurar que `GROQ_API_KEY` esté en `.env`
2. Reiniciar el agente médico:
   ```bash
   cd samples/python/agents/medical_Images
   uv sync
   uv run python -m app
   ```
3. Enviar una imagen médica desde el frontend
4. Verificar logs:
   ```
   ✅ Imagen 0 agregada para Groq: image/png
   ✅ Análisis de imágenes completado con Groq Llama 4 Scout
   ```

## Logs Esperados

```
📸 Analizando 1 imagen(es)...
✅ Imagen 0 agregada para Groq: image/png
✅ Análisis de imágenes completado con Groq Llama 4 Scout
✅ Análisis visual completado
🔍 Clasificando consulta...
```

## Comparación: Gemini vs Groq

| Aspecto | Gemini 2.0 Flash | Groq Llama 4 Scout |
|---------|------------------|---------------------|
| Visión | Excelente | Muy buena |
| Velocidad | Rápido | Muy rápido |
| Costo | De pago | Gratis |
| Precisión médica | Excelente | Buena |
| Límite imágenes | 16 | 5 |
| Tamaño imagen | 20MB | 4MB (base64) |

## Rollback a Gemini

Si necesitas volver a Gemini para mejor precisión médica:

1. Agregar `langchain-google-genai>=2.0.8` a `pyproject.toml`
2. Crear `self.vision_llm` con `ChatGoogleGenerativeAI`
3. Usar `self.vision_llm.invoke()` en `analyze_images()`
4. Agregar validación de `GOOGLE_API_KEY` en `__main__.py`
5. Ejecutar `uv sync`

## Estado

✅ **COMPLETADO**

- ✅ Formato de mensaje corregido
- ✅ Usando solo Groq (sin Gemini)
- ✅ Dependencias actualizadas
- ✅ Validaciones actualizadas
- ⏳ Pendiente: Testing con usuario

## Archivos Modificados

1. `samples/python/agents/medical_Images/app/agent.py` - Formato de imagen corregido
2. `samples/python/agents/medical_Images/app/__main__.py` - Removida validación de Google API Key
3. `samples/python/agents/medical_Images/pyproject.toml` - Removida dependencia de Gemini
4. `FIX-GROQ-VISION-AGENTE-MEDICO.md` - Documentación (este archivo)

## Próximos Pasos

1. Ejecutar `uv sync` en el directorio del agente médico
2. Reiniciar el agente médico
3. Probar con una imagen médica
4. Verificar que el análisis funcione correctamente
5. Comparar calidad con Gemini (si es necesario)
