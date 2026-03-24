# Requirements Document

## Introduction

Este documento especifica los requisitos para integrar el frontend completo del repositorio a2a-frontend (Next.js + TypeScript + Tailwind) en el proyecto actual. La integración debe permitir ejecutar tanto el backend Python existente como el nuevo frontend Next.js con un solo comando, manteniendo toda la funcionalidad del frontend original incluyendo chat multimodal, admin dashboard, y comunicación con la API del backend.

## Glossary

- **Frontend_Application**: La aplicación Next.js del repositorio a2a-frontend que incluye interfaz de chat y admin dashboard
- **Backend_Server**: El servidor Python FastAPI ubicado en demo/ui que expone la API REST en el puerto 12000
- **Agent_Process**: Cada uno de los agentes Python ubicados en samples/python/agents/ que se ejecutan en puertos individuales
- **Dev_Command**: El comando npm run dev que orquesta el inicio de todos los servicios
- **Chat_Interface**: Componente del frontend que permite interacción multimodal (texto + imágenes) con los agentes
- **Admin_Dashboard**: Componente del frontend con 7 pestañas (Chat, Conversations, Messages, Events, Tasks, Agents, Settings)
- **API_Client**: Cliente HTTP del frontend que se comunica con el Backend_Server
- **Package_Manager**: npm como gestor de paquetes para el proyecto integrado
- **Process_Orchestrator**: Herramienta concurrently que ejecuta múltiples procesos simultáneamente

## Requirements

### Requirement 1: Frontend Repository Integration

**User Story:** Como desarrollador, quiero integrar el código del frontend a2a-frontend en este proyecto, para que todo el código esté en un solo repositorio.

#### Acceptance Criteria

1. THE Frontend_Application SHALL be located in a frontend/ directory at the root of the project
2. THE Frontend_Application SHALL preserve all original files from a2a-frontend including components, pages, styles, and configuration
3. THE Frontend_Application SHALL maintain its original package.json dependencies (Next.js 13.5.1, React 18.2.0, TypeScript, Tailwind CSS, shadcn/ui)
4. THE Frontend_Application SHALL include all UI components (ChatInterface, AdminDashboard, MessageManager, etc.)
5. THE Frontend_Application SHALL preserve the API client configuration that communicates with localhost:12000

### Requirement 2: Development Command Configuration

**User Story:** Como desarrollador, quiero ejecutar todo el sistema con un solo comando, para que sea fácil iniciar el entorno de desarrollo.

#### Acceptance Criteria

1. WHEN the developer executes npm run dev, THE Dev_Command SHALL start the Backend_Server on port 12000
2. WHEN the developer executes npm run dev, THE Dev_Command SHALL start all Agent_Process instances from samples/python/agents/
3. WHEN the developer executes npm run dev, THE Dev_Command SHALL start the Frontend_Application
4. THE Process_Orchestrator SHALL display output from all processes in a single terminal with labeled prefixes
5. IF any process fails to start, THEN THE Process_Orchestrator SHALL display the error and continue running other processes

### Requirement 3: Backend API Compatibility

**User Story:** Como usuario del frontend, quiero que el frontend se comunique correctamente con el backend existente, para que todas las funcionalidades funcionen sin errores.

#### Acceptance Criteria

1. THE API_Client SHALL send HTTP requests to http://localhost:12000
2. WHEN the Frontend_Application sends a message, THE Backend_Server SHALL process it using the existing /message/send endpoint
3. WHEN the Frontend_Application requests conversations, THE Backend_Server SHALL respond using the existing /conversation/list endpoint
4. WHEN the Frontend_Application requests messages, THE Backend_Server SHALL respond using the existing /message/list endpoint
5. WHEN the Frontend_Application uploads an image, THE Backend_Server SHALL handle FileWithBytes format correctly
6. THE Backend_Server SHALL maintain CORS configuration to allow requests from the Frontend_Application origin

### Requirement 4: Chat Interface Functionality

**User Story:** Como usuario final, quiero usar la interfaz de chat con soporte multimodal, para que pueda enviar texto e imágenes a los agentes.

#### Acceptance Criteria

1. THE Chat_Interface SHALL display a text input field for user messages
2. THE Chat_Interface SHALL provide an image upload button for attaching images
3. WHEN a user sends a text message, THE Chat_Interface SHALL display it in the conversation view
4. WHEN a user uploads an image, THE Chat_Interface SHALL display a preview before sending
5. WHEN a user sends a message with an image, THE Chat_Interface SHALL encode the image as base64 and include it in the request
6. THE Chat_Interface SHALL poll the Backend_Server for new messages in real-time
7. WHEN the Backend_Server responds, THE Chat_Interface SHALL display agent messages in the conversation view

### Requirement 5: Admin Dashboard Functionality

**User Story:** Como administrador, quiero acceder al dashboard con todas sus pestañas, para que pueda monitorear y gestionar el sistema.

#### Acceptance Criteria

1. THE Admin_Dashboard SHALL display 7 navigation tabs: Chat, Conversations, Messages, Events, Tasks, Agents, Settings
2. WHEN a user clicks the Chat tab, THE Admin_Dashboard SHALL display the Chat_Interface
3. WHEN a user clicks the Conversations tab, THE Admin_Dashboard SHALL fetch and display all conversations from /conversation/list
4. WHEN a user clicks the Messages tab, THE Admin_Dashboard SHALL fetch and display all messages from /message/list
5. WHEN a user clicks the Events tab, THE Admin_Dashboard SHALL fetch and display all events from /events/get
6. WHEN a user clicks the Tasks tab, THE Admin_Dashboard SHALL fetch and display all tasks from /task/list
7. WHEN a user clicks the Agents tab, THE Admin_Dashboard SHALL fetch and display all registered agents from /agent/list
8. WHEN a user clicks the Settings tab, THE Admin_Dashboard SHALL display configuration options

### Requirement 6: Agent Process Management

**User Story:** Como desarrollador, quiero que todos los agentes se inicien automáticamente, para que estén disponibles cuando el frontend los necesite.

#### Acceptance Criteria

1. THE Dev_Command SHALL start the images agent on port 10001
2. THE Dev_Command SHALL start the medical_Images agent on port 10002
3. THE Dev_Command SHALL start the multimodal agent on port 10003
4. WHEN all Agent_Process instances start, THE Backend_Server SHALL auto-register them using their respective URLs
5. IF an Agent_Process fails to start, THEN THE Dev_Command SHALL log the error but continue running other processes

### Requirement 7: Package Manager Configuration

**User Story:** Como desarrollador, quiero que el proyecto use npm como gestor de paquetes principal, para que la gestión de dependencias sea consistente.

#### Acceptance Criteria

1. THE Package_Manager SHALL be npm for the root project
2. THE root package.json SHALL include scripts for dev, dev:ui, dev:frontend, and dev:agent:\*
3. THE root package.json SHALL include concurrently as a devDependency
4. THE Frontend_Application SHALL have its own package.json in the frontend/ directory
5. WHEN a developer runs npm install at the root, THE Package_Manager SHALL install root dependencies
6. THE developer SHALL run npm install separately in the frontend/ directory for frontend dependencies

### Requirement 8: Environment Configuration

**User Story:** Como desarrollador, quiero que las variables de entorno estén correctamente configuradas, para que todos los servicios se conecten correctamente.

#### Acceptance Criteria

1. THE Backend_Server SHALL read A2A_UI_HOST from environment with default value '0.0.0.0'
2. THE Backend_Server SHALL read A2A_UI_PORT from environment with default value '12000'
3. THE Backend_Server SHALL read A2A_HOST from environment with default value 'BEEAI'
4. THE Frontend_Application SHALL connect to the Backend_Server at http://localhost:12000
5. WHERE GOOGLE_API_KEY is set, THE Backend_Server SHALL use it for API authentication
6. WHERE GOOGLE_GENAI_USE_VERTEXAI is 'TRUE', THE Backend_Server SHALL use Vertex AI instead of API key

### Requirement 9: Real-time Message Polling

**User Story:** Como usuario final, quiero ver los mensajes de los agentes en tiempo real, para que la conversación sea fluida.

#### Acceptance Criteria

1. THE Chat_Interface SHALL poll /message/pending every 2 seconds
2. WHEN new messages are available, THE Chat_Interface SHALL fetch them using /message/list
3. WHEN new messages are fetched, THE Chat_Interface SHALL append them to the conversation view
4. THE Chat_Interface SHALL scroll to the latest message automatically
5. IF the polling request fails, THEN THE Chat_Interface SHALL retry after 5 seconds

### Requirement 10: File Caching and Serving

**User Story:** Como usuario final, quiero que las imágenes se muestren correctamente en el chat, para que pueda ver el contenido visual de las conversaciones.

#### Acceptance Criteria

1. WHEN the Backend_Server receives a message with FileWithBytes, THE Backend_Server SHALL cache the file with a unique ID
2. WHEN the Backend_Server returns messages to the frontend, THE Backend_Server SHALL replace FileWithBytes with FileWithUri containing the cache ID
3. WHEN the Frontend_Application requests an image, THE Frontend_Application SHALL use the /message/file/{file_id} endpoint
4. WHEN the Backend_Server serves a cached file, THE Backend_Server SHALL return it with the correct MIME type
5. IF an image MIME type is detected, THEN THE Backend_Server SHALL decode base64 before serving

### Requirement 11: Error Handling and Logging

**User Story:** Como desarrollador, quiero ver logs claros de todos los procesos, para que pueda diagnosticar problemas fácilmente.

#### Acceptance Criteria

1. THE Process_Orchestrator SHALL prefix each log line with the process name
2. WHEN the Backend_Server processes a message, THE Backend_Server SHALL log the message ID and parts count
3. WHEN an Agent_Process fails to register, THE Backend_Server SHALL log the error with the agent URL
4. WHEN the Frontend_Application encounters an API error, THE Frontend_Application SHALL display a user-friendly error message
5. IF a file upload fails, THEN THE Chat_Interface SHALL display an error notification

### Requirement 12: TypeScript Type Safety

**User Story:** Como desarrollador del frontend, quiero que el código TypeScript tenga tipos correctos, para que pueda detectar errores en tiempo de compilación.

#### Acceptance Criteria

1. THE API_Client SHALL define TypeScript interfaces for all API request and response types
2. THE Chat_Interface SHALL use typed props for all React components
3. THE Admin_Dashboard SHALL use typed state management
4. WHEN the Frontend_Application builds, THE TypeScript compiler SHALL report zero type errors
5. THE Frontend_Application SHALL use strict TypeScript configuration
