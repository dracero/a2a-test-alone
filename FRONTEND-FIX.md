# Fix: Frontend Missing API Module

## Problema
El frontend de Next.js no podía compilar porque faltaban los módulos:
- `@/lib/api` - Cliente API para comunicarse con el backend
- `@/lib/utils` - Utilidades para clases CSS (cn helper)

## Solución

### Archivos Creados

1. **`frontend/lib/api.ts`**
   - Cliente API completo para comunicarse con el backend
   - Tipos TypeScript para todas las entidades (Message, Conversation, Agent, Task, Event)
   - Métodos para todas las operaciones:
     - Conversaciones: crear, listar
     - Mensajes: enviar, listar, obtener pendientes
     - Agentes: registrar, listar
     - Tareas: listar
     - Eventos: obtener
     - API Key: actualizar
   - Singleton `chatAPI` exportado para uso global

2. **`frontend/lib/utils.ts`**
   - Función `cn()` para combinar clases CSS con Tailwind
   - Usa `clsx` y `tailwind-merge` para manejo inteligente de clases

3. **`frontend/.env.local`**
   - Variable de entorno `NEXT_PUBLIC_API_URL` configurada
   - Apunta al backend en `http://localhost:12000`

4. **`frontend/.env.local.example`**
   - Plantilla de ejemplo para otros desarrolladores

## Estructura de la API

### Tipos Principales

```typescript
interface Message {
  message_id: string;
  context_id: string;
  role: 'user' | 'model';
  parts: Part[];
  recipient?: string;
  metadata?: Record<string, any>;
}

interface Part {
  kind: 'text' | 'file';
  text?: string;
  file?: {
    mime_type: string;
    uri?: string;
    bytes?: string;
    name?: string;
  };
}

interface Conversation {
  conversation_id: string;
  messages: Message[];
  created_at?: string;
}

interface Agent {
  name: string;
  url: string;
  description?: string;
  version?: string;
  capabilities?: Record<string, any>;
}
```

### Uso del Cliente API

```typescript
import { chatAPI, Message, Conversation } from '@/lib/api';

// Crear conversación
const { result: conversation } = await chatAPI.createConversation();

// Enviar mensaje
const message: Message = {
  message_id: 'msg-123',
  context_id: conversation.conversation_id,
  role: 'user',
  parts: [{ kind: 'text', text: 'Hello!' }]
};
await chatAPI.sendMessage(message);

// Listar mensajes
const { result: messages } = await chatAPI.listMessages(conversation.conversation_id);

// Listar agentes
const { result: agents } = await chatAPI.listAgents();
```

## Configuración TypeScript

El archivo `tsconfig.json` ya tenía configurado el alias de path:

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./*"]
    }
  }
}
```

Esto permite importar desde `@/lib/api` en lugar de rutas relativas como `../../lib/api`.

## Endpoints del Backend

El cliente API se conecta a estos endpoints en el backend (puerto 12000):

- `POST /conversation/create` - Crear nueva conversación
- `POST /conversation/list` - Listar conversaciones
- `POST /message/send` - Enviar mensaje
- `POST /message/list` - Listar mensajes de una conversación
- `POST /message/pending` - Obtener mensajes pendientes
- `POST /agent/register` - Registrar agente por URL
- `POST /agent/register/manual` - Registrar agente manualmente
- `POST /agent/list` - Listar agentes registrados
- `POST /task/list` - Listar tareas
- `POST /events/get` - Obtener eventos
- `POST /api_key/update` - Actualizar API key
- `GET /message/file/{file_id}` - Obtener archivo cacheado

## Verificación

Para verificar que todo funciona:

```bash
# En el directorio frontend
npm run dev
```

El frontend debería compilar sin errores y conectarse al backend en el puerto 12000.

## Notas

- El backend debe estar corriendo en el puerto 12000 (configurado en `demo/ui/main.py`)
- Los agentes deben estar corriendo en sus puertos respectivos (10001, 10002, 10003)
- La variable `NEXT_PUBLIC_API_URL` puede ser modificada en `.env.local` si el backend está en otra URL
