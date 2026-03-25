# Cambio de Gemini a Groq (Llama 4) - COMPLETADO ✅

## Resumen
Todos los agentes Y el backend orchestrator han sido actualizados para usar Groq con el modelo `meta-llama/llama-4-scout-17b-16e-instruct` en lugar de Google Gemini.

## Cambios Realizados

### 1. Agente Médico (`medical_Images`)

**Archivo**: `samples/python/agents/medical_Images/app/agent.py`
- ✅ Cambiado de `ChatGoogleGenerativeAI` a `ChatGroq`
- ✅ Modelo: `meta-llama/llama-4-scout-17b-16e-instruct`
- ✅ API Key: `GROQ_API_KEY`

**Archivo**: `samples/python/agents/medical_Images/app/__main__.py`
- ✅ Validación cambiada de `GOOGLE_API_KEY` a `GROQ_API_KEY`
- ✅ Mensajes de error actualizados

**Archivo**: `samples/python/agents/medical_Images/pyproject.toml`
- ✅ Agregado: `langchain-groq>=0.2.0`
- ✅ Removido: `langchain-google-genai`

### 2. Agente Multimodal (`multimodal`)

**Archivo**: `samples/python/agents/multimodal/app/agent.py`
- ✅ Constante cambiada: `GEMINI_MODEL` → `GROQ_MODEL`
- ✅ Cambiado de `ChatGoogleGenerativeAI` a `ChatGroq`
- ✅ Modelo: `meta-llama/llama-4-scout-17b-16e-instruct`

**Archivo**: `samples/python/agents/multimodal/app/__main__.py`
- ✅ Validación cambiada de `GOOGLE_API_KEY` a `GROQ_API_KEY`
- ✅ Mensajes de error actualizados

**Archivo**: `samples/python/agents/multimodal/pyproject.toml`
- ✅ Agregado: `langchain-groq>=0.2.0`
- ✅ Removido: `langchain-google-genai`

### 3. Agente de Imágenes (`images`)

**Archivo**: `samples/python/agents/images/app/agent.py`
- ✅ LLM del agente CrewAI cambiado a Groq
- ✅ Modelo: `groq/llama-3.3-70b-versatile` (compatible con LiteLLM)
- ⚠️ **Nota**: La generación de imágenes sigue usando Gemini 2.5 Flash Image porque Groq no tiene capacidad de generación de imágenes

**Archivo**: `samples/python/agents/images/app/__main__.py`
- ✅ Validación actualizada para requerir tanto `GROQ_API_KEY` (razonamiento) como `GOOGLE_API_KEY` (generación de imágenes)

**Archivo**: `samples/python/agents/images/pyproject.toml`
- ✅ Agregado: `litellm>=1.0.0` (requerido por CrewAI para usar Groq)

### 4. Backend Orchestrator (`demo/ui/`)

**Archivo**: `demo/ui/service/server/beeai_host_manager.py`
- ✅ Cambiado de `ChatGoogleGenerativeAI` a `ChatGroq`
- ✅ Modelo: `meta-llama/llama-4-scout-17b-16e-instruct`
- ✅ API Key: `GROQ_API_KEY`

**Archivo**: `demo/ui/service/server/beeai_orchestrator_workflow.py`
- ✅ Actualizado comentarios de "Gemini" a "Groq"
- ✅ Actualizado mensajes de log para reflejar uso de Groq

**Archivo**: `demo/ui/service/server/server.py`
- ✅ Actualizado para usar `GROQ_API_KEY` cuando `A2A_HOST=BEEAI`
- ✅ Mantiene `GOOGLE_API_KEY` cuando `A2A_HOST=ADK`

**Archivo**: `demo/ui/service/server/adk_host_manager.py`
- ✅ Actualizado para usar `GROQ_API_KEY` en lugar de `GOOGLE_API_KEY`
- ✅ Actualizado método `update_api_key()` para usar `GROQ_API_KEY`

**Archivo**: `demo/ui/main.py`
- ✅ Actualizado para leer `GROQ_API_KEY` del entorno
- ✅ Comentarios actualizados

**Archivo**: `demo/ui/components/api_key_dialog.py`
- ✅ Actualizado para solicitar "Groq API Key" en lugar de "Google API Key"
- ✅ Actualizado para guardar en `GROQ_API_KEY`

**Archivo**: `demo/ui/pyproject.toml`
- ✅ Agregado: `langchain-groq>=0.2.0`
- ✅ Cambiado: `langchain-google-genai` → `langchain-groq`

## Variables de Entorno Requeridas

Actualiza tu archivo `.env` en el root del proyecto:

```bash
# Groq API Key (para todos los agentes Y el orchestrator)
GROQ_API_KEY=tu_groq_api_key_aqui

# Google API Key (solo para agente de imágenes - generación)
GOOGLE_API_KEY=tu_google_api_key_aqui

# Tavily API Key (para agente médico - búsqueda)
TAVILY_API_KEY=tu_tavily_api_key_aqui

# Qdrant (para agente multimodal)
QDRANT_KEY=tu_qdrant_key_aqui
QDRANT_URL=tu_qdrant_url_aqui

# Opcional
LANGSMITH_API_KEY=tu_langsmith_key_aqui
HF_TOKEN=tu_huggingface_token_aqui
```

## Instalación de Dependencias

Las dependencias ya fueron actualizadas. Para reinstalar:

```bash
# Backend orchestrator
cd demo/ui
uv sync

# Agente médico
cd samples/python/agents/medical_Images
uv sync

# Agente multimodal
cd samples/python/agents/multimodal
uv sync

# Agente de imágenes
cd samples/python/agents/images
uv sync
```

## Configuración del Modelo

### Modelos Usados

**Backend Orchestrator, Agentes Médico y Multimodal**:
- **Nombre**: `meta-llama/llama-4-scout-17b-16e-instruct`
- **Proveedor**: Groq (vía langchain-groq)
- **Parámetros**:
  - Temperature: 0.3
  - Max tokens: 4096

**Agente de Imágenes**:
- **LLM (razonamiento)**: `groq/llama-3.3-70b-versatile`
- **Proveedor**: Groq (vía LiteLLM en CrewAI)
- **Generación de imágenes**: Gemini 2.5 Flash Image (Google)

### Comparación con Gemini

| Característica | Gemini 2.5 Flash | Llama 4 Scout (Groq) |
|----------------|------------------|----------------------|
| Velocidad | Rápido | Muy rápido |
| Contexto | 1M tokens | 128K tokens |
| Multimodal | Sí (visión) | No |
| Costo | Medio | Bajo |
| Latencia | ~1-2s | ~0.3-0.5s |

## Limitaciones

### Agente de Imágenes
- **Razonamiento**: Usa Groq (Llama 4)
- **Generación de imágenes**: Sigue usando Gemini 2.5 Flash Image
- **Razón**: Groq no tiene capacidad de generación de imágenes

### Agente Multimodal
- **Análisis de imágenes**: Limitado (Llama 4 no tiene visión nativa)
- **Solución**: Las imágenes se procesan con modelos de embeddings locales (CLIP, etc.)

### Agente Médico
- **Análisis de imágenes médicas**: Limitado
- **Recomendación**: Para análisis de imágenes médicas complejas, considera mantener Gemini

## Verificación

Para verificar que todo funciona:

```bash
# Probar backend orchestrator
cd demo/ui
uv run python -c "from service.server.beeai_host_manager import BeeAIHostManager; print('✅ Backend Orchestrator OK')"

# Probar agente médico
cd samples/python/agents/medical_Images
uv run python -c "from app.agent import MedicalAgent; print('✅ Medical Agent OK')"

# Probar agente multimodal
cd samples/python/agents/multimodal
uv run python -c "from app.agent import PhysicsMultimodalAgent; print('✅ Multimodal Agent OK')"

# Probar agente de imágenes
cd samples/python/agents/images
uv run python -c "from app.agent import ImageGenerationAgent; print('✅ Images Agent OK')"
```

## Ejecución

Reinicia todos los servicios:

```bash
# Detener todo
pkill -f "uv run"
pkill -f "npm"

# Reiniciar backend
cd demo/ui
uv run python main.py &

# Reiniciar agentes (en terminales separadas)
cd samples/python/agents/medical_Images
uv run python -m app &

cd samples/python/agents/multimodal
uv run python -m app &

cd samples/python/agents/images
uv run python -m app &

# Reiniciar frontend
cd frontend
npm run dev
```

## Beneficios de la Migración

1. **Resuelve el error 429 (quota exceeded)**: El orchestrator ya no usa Gemini, eliminando el problema de cuota
2. **Mayor velocidad**: Groq es significativamente más rápido que Gemini
3. **Menor costo**: Groq tiene precios más competitivos
4. **Centralización**: Todos los componentes usan la misma API key (excepto generación de imágenes)

## Notas Importantes

1. **Groq es más rápido**: Espera respuestas mucho más rápidas que con Gemini
2. **Sin visión nativa**: Llama 4 no procesa imágenes directamente como Gemini
3. **Límites de rate**: Groq tiene límites de requests por minuto (verifica tu plan)
4. **Calidad**: Llama 4 Scout es excelente para razonamiento y texto, pero diferente a Gemini en estilo
5. **Orchestrator migrado**: El backend BeeAI ahora usa Groq, resolviendo el problema de cuota de Gemini

## Rollback a Gemini

Si necesitas volver a Gemini, los cambios están documentados en este archivo. Básicamente:
1. Revertir los cambios en `agent.py` de cada agente y en `beeai_host_manager.py`
2. Cambiar `langchain-groq` por `langchain-google-genai` en `pyproject.toml`
3. Actualizar validaciones de API keys en `__main__.py` y `main.py`
4. Actualizar `api_key_dialog.py` para solicitar Google API Key
5. Ejecutar `uv sync` en cada directorio

## Estado Final

✅ **MIGRACIÓN COMPLETADA**
- ✅ Agente médico migrado a Groq
- ✅ Agente multimodal migrado a Groq
- ✅ Agente de imágenes migrado a Groq (razonamiento)
- ✅ Backend orchestrator (BeeAI) migrado a Groq
- ✅ Backend orchestrator (ADK) actualizado para usar GROQ_API_KEY
- ✅ UI actualizada para solicitar Groq API Key
- ✅ Dependencias instaladas
- ✅ Documentación actualizada

**Próximo paso**: Reiniciar todos los servicios y probar la aplicación completa.
