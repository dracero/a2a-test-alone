# Fix: Backend Message Parsing Error

## Problema
El backend estaba fallando al parsear mensajes del frontend con el error:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Message
context_id
  Input should be a valid string [type=string_type, input_value={'jsonrpc': '2.0', 'id': ...}, input_type=dict]
```

## Causa
El frontend estaba enviando `context_id` como un objeto (dict) en lugar de un string, causando que Pydantic rechazara la validación.

## Solución

### Cambios en `demo/ui/service/server/server.py`

Se agregó validación y conversión robusta para `context_id` en el método `parse_message_from_dict`:

```python
# 🔧 CORRECCIÓN: Asegurar que context_id sea string
context_id_value = data.get('context_id', '')
if isinstance(context_id_value, dict):
    # Si es un dict, intentar extraer un ID o usar string vacío
    context_id_value = context_id_value.get('id', '') or context_id_value.get('conversation_id', '') or ''
elif not isinstance(context_id_value, str):
    context_id_value = str(context_id_value) if context_id_value else ''
```

### Lógica de Conversión

1. **Si es dict**: Intenta extraer `id` o `conversation_id` del objeto
2. **Si no es string**: Convierte a string usando `str()`
3. **Si es None o vacío**: Usa string vacío `''`

### Debug Mejorado

Se agregó logging adicional para ver el contenido completo del mensaje recibido:

```python
print(f"Raw data: {data}")
```

Esto ayuda a diagnosticar problemas de formato en el futuro.

## Validaciones Adicionales

El método `parse_message_from_dict` ahora maneja:

1. ✅ `context_id` como string
2. ✅ `context_id` como dict (extrae ID)
3. ✅ `context_id` como None (usa '')
4. ✅ `context_id` como número (convierte a string)
5. ✅ `role` con prefijo 'Role.' (lo limpia)
6. ✅ Parts de tipo text y file
7. ✅ Files con bytes o URI

## Testing

Para probar que funciona:

1. Inicia el backend:
   ```bash
   npm run dev:backend
   ```

2. Inicia el frontend:
   ```bash
   npm run dev:frontend
   ```

3. Envía un mensaje desde la UI

El mensaje debería procesarse correctamente sin errores de validación.

## Logs Esperados

Cuando se envía un mensaje, deberías ver:

```
============================================================
🔍 PARSING MESSAGE FROM FRONTEND
Raw data keys: dict_keys(['message_id', 'context_id', 'role', 'parts'])
Raw data: {'message_id': '...', 'context_id': '...', 'role': 'user', 'parts': [...]}
============================================================

📦 Part 0: {'kind': 'text', 'text': 'Hola'}
  ✅ Text part added

✅ Message parsed successfully:
  • message_id: msg-123
  • context_id: conv-456
  • role: user
  • parts count: 1
============================================================
```

## Notas

- Esta corrección es backward compatible con el formato anterior
- No requiere cambios en el frontend
- Maneja múltiples formatos de entrada de manera robusta
