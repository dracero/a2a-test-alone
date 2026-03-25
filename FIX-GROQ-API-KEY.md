# Fix: Error "Invalid API Key" en Orchestrator

## Problema

Al enviar un mensaje al orchestrator, se producía el siguiente error:

```
❌ Error during classification: Error code: 401 - {'error': {'message': 'Invalid API Key', 'type': 'invalid_request_error', 'code': 'invalid_api_key'}}
```

## Causa Raíz

El archivo `demo/ui/service/server/server.py` estaba leyendo `GOOGLE_API_KEY` para todos los managers (ADK y BeeAI), pero el BeeAI Host Manager necesita `GROQ_API_KEY` porque usa Groq en lugar de Gemini.

### Código Problemático

```python
# server.py (línea 58)
api_key = os.environ.get('GOOGLE_API_KEY', '')  # ❌ Siempre usa GOOGLE_API_KEY
```

Esto causaba que el BeeAI Host Manager se inicializara con una API key de Google en lugar de Groq, resultando en un error 401 cuando intentaba hacer llamadas a Groq.

## Solución

Actualizar `server.py` para usar la API key correcta según el tipo de manager:

```python
# server.py (línea 58)
# Use GROQ_API_KEY for BeeAI, GOOGLE_API_KEY for ADK
api_key = os.environ.get('GROQ_API_KEY' if agent_manager.upper() == 'BEEAI' else 'GOOGLE_API_KEY', '')
```

## Archivos Modificados

1. **`demo/ui/service/server/server.py`**
   - Línea 58: Actualizada lógica de selección de API key

2. **`demo/ui/service/server/beeai_orchestrator_workflow.py`**
   - Actualizados comentarios de "Gemini" a "Groq"
   - Actualizados mensajes de log para reflejar uso de Groq

## Verificación

Después del fix, el orchestrator debería:

1. ✅ Leer correctamente `GROQ_API_KEY` del `.env`
2. ✅ Inicializar el LLM de Groq con la API key correcta
3. ✅ Clasificar mensajes sin errores 401
4. ✅ Enrutar correctamente a los agentes especializados

## Cómo Probar

1. Asegúrate de que `GROQ_API_KEY` esté en tu `.env`:
   ```bash
   GROQ_API_KEY=tu_groq_api_key_aqui
   ```

2. Reinicia el backend:
   ```bash
   ./restart-all.sh
   ```

3. Envía un mensaje de prueba desde el frontend:
   ```
   Hola
   ```

4. Verifica en los logs que no hay errores 401:
   ```bash
   tail -f logs/backend.log
   ```

## Logs Esperados (Correcto)

```
🤔 Step 2: Classifying request and choosing agent...
🔍 Sending classification prompt to Groq...
📥 Groq response type: <class 'langchain_core.messages.ai.AIMessage'>
📥 Groq response content: DIRECT
✅ Responding directly (no agent needed)
✅ Direct response generated: Hola! ¿En qué puedo ayudarte hoy?...
```

## Logs Anteriores (Error)

```
🤔 Step 2: Classifying request and choosing agent...
🔍 Sending classification prompt to Gemini...
❌ Error during classification: Error code: 401 - {'error': {'message': 'Invalid API Key', 'type': 'invalid_request_error', 'code': 'invalid_api_key'}}
⚠️ No agent chosen, generating fallback response
```

## Contexto Adicional

Este fix es parte de la migración completa de Gemini a Groq. Otros componentes ya habían sido migrados:

- ✅ BeeAI Host Manager (usa ChatGroq)
- ✅ Agentes Python (usan ChatGroq)
- ✅ UI (solicita Groq API Key)

El único componente que faltaba era el `server.py`, que es el punto de entrada que instancia el manager.

## Documentación Relacionada

- `CAMBIO-A-GROQ.md` - Migración completa a Groq
- `RESUMEN-COMPLETO.md` - Resumen de todos los cambios
- `INICIO-RAPIDO.md` - Guía de inicio rápido

## Estado

✅ **RESUELTO** - El orchestrator ahora usa correctamente la API key de Groq.
