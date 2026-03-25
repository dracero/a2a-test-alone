# Cómo Reiniciar el Sistema

## Problema Actual
El backend necesita reiniciarse para aplicar los cambios en el código que corrigen el error de validación de `context_id`.

## Solución Rápida

### Opción 1: Reiniciar Todo el Sistema

1. Detén todos los procesos con `Ctrl+C` en la terminal donde ejecutaste `npm run dev`

2. Vuelve a iniciar todo:
   ```bash
   npm run dev
   ```

### Opción 2: Reiniciar Solo el Backend

1. Encuentra el proceso del backend:
   ```bash
   ps aux | grep "uv run main.py"
   ```

2. Mata el proceso (reemplaza PID con el número que encontraste):
   ```bash
   kill PID
   ```

3. O usa el script de reinicio:
   ```bash
   bash restart-backend.sh
   ```

### Opción 3: Usar pkill (Más Fácil)

```bash
# Matar el backend
pkill -f "uv run main.py"

# Reiniciar el backend
cd demo/ui && uv run main.py
```

## Verificar que Funciona

Después de reiniciar, deberías ver en los logs del backend:

```
🔍 PARSING MESSAGE FROM FRONTEND
Raw data keys: dict_keys([...])
Raw data: {...}
```

Y cuando envíes un mensaje, debería procesarse sin el error de validación.

## Cambios Aplicados

Los cambios que corrigen el error ya están guardados en:
- `demo/ui/service/server/server.py`

El método `parse_message_from_dict` ahora:
- ✅ Convierte `context_id` de dict a string automáticamente
- ✅ Maneja múltiples formatos de entrada
- ✅ Tiene mejor logging para debug

## Si el Error Persiste

1. Verifica que el backend se reinició correctamente:
   ```bash
   curl http://localhost:12000/agent/list -X POST
   ```

2. Revisa los logs del backend para ver si hay otros errores

3. Verifica que el frontend esté enviando el formato correcto:
   - Abre las DevTools del navegador (F12)
   - Ve a la pestaña Network
   - Envía un mensaje
   - Revisa el payload del request a `/message/send`

## Estructura Esperada del Mensaje

El frontend debería enviar:

```json
{
  "params": {
    "message_id": "msg-123",
    "context_id": "conv-456",  // ← Debe ser string, no objeto
    "role": "user",
    "parts": [
      {
        "kind": "text",
        "text": "Hola"
      }
    ]
  }
}
```

Si `context_id` viene como objeto, el backend ahora lo convierte automáticamente a string.
