# Agent2Agent (A2A) Samples

Welcome to the A2A Samples repository! Here you will find code samples and demos using the Agent2Agent protocol.

<a href="https://studio.firebase.google.com/new?template=https%3A%2F%2Fgithub.com%2Fa2aproject%2Fa2a-samples%2Ftree%2Fmain%2F.firebase-studio">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="https://cdn.firebasestudio.dev/btn/try_light_20.svg">
    <source
      media="(prefers-color-scheme: light)"
      srcset="https://cdn.firebasestudio.dev/btn/try_dark_20.svg">
    <img
      height="20"
      alt="Try in Firebase Studio"
      src="https://cdn.firebasestudio.dev/btn/try_blue_20.svg">
  </picture>
</a>

<div style="text-align: right;">
  <details>
    <summary>🌐 Language</summary>
    <div style="text-align: center;">
      <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=en">English</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=zh-CN">简体中文</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=zh-TW">繁體中文</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=ja">日本語</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=ko">한국어</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=hi">हिन्दी</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=th">ไทย</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=fr">Français</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=de">Deutsch</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=es">Español</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=it">Italiano</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=ru">Русский</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=pt">Português</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=nl">Nederlands</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=pl">Polski</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=ar">العربية</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=fa">فارسی</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=tr">Türkçe</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=vi">Tiếng Việt</a>
      | <a href="https://openaitx.github.io/view.html?user=a2aproject&project=a2a-samples&lang=id">Bahasa Indonesia</a>
    </div>
  </details>
</div>

This repository contains code samples and demos which use the [Agent2Agent (A2A) Protocol](https://goo.gle/a2a).

## 🚀 Quick Start

This repository has been configured to use **Groq (Llama 4)** for ultra-fast agent responses.

### Prerequisites

1. **Groq API Key** - Get one at [console.groq.com](https://console.groq.com)
2. **Google API Key** (optional) - Only needed for image generation
3. **Python 3.12+** with `uv` package manager
4. **Node.js 18+** for the frontend

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dracero/a2a-test-alone.git
   cd a2a-test-alone
   ```

2. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. Start all services:
   ```bash
   npm run dev
   ```
   *Note: This runs the `start_ordered.py` script natively, starting the agents, waiting for their ports to be ready, and then launching the backend orchestrator and Next.js frontend. Works on Windows, macOS, and Linux natively without WSL.*

4. Open your browser at [http://localhost:3000](http://localhost:3000)

### Documentation

- **[INICIO-RAPIDO.md](INICIO-RAPIDO.md)** - Quick start guide (Spanish)
- **[RESUMEN-COMPLETO.md](RESUMEN-COMPLETO.md)** - Complete summary of changes
- **[CAMBIO-A-GROQ.md](CAMBIO-A-GROQ.md)** - Groq migration details

## Architecture & Ecosistema

Este sistema se compone de los siguientes elementos organizados de forma concurrente y orquestada:

```mermaid
graph TD
    User([Cliente / Usuario]) <--> Frontend[Frontend Next.js: Puerto 3000]
    Frontend <--> Orchestrator[Backend Orquestador: Puerto 12000]
    
    subgraph Agents [Agentes Especializados A2A]
        AgentMed[Asistente Médico: Puerto 10002]
        AgentImg[Generador de Imágenes: Puerto 10001]
        AgentPhys[Tutor Socrático de Física: Puerto 10003]
    end
    
    Orchestrator <--> AgentMed
    Orchestrator <--> AgentImg
    Orchestrator <--> AgentPhys
    
    subgraph MemorySystem [Memoria en Grafo Neo4j y Auto-Aprendizaje]
        Neo4jClient[MemoryClient]
        AuraDB[(Neo4j Aura Cloud DB)]
        LocalEmbedder[SentenceTransformers Local: BAAI/bge-small-en-v1.5]
        LearningLoop[Extractor en Segundo Plano: Groq LLM]
    end
    
    Orchestrator -->|1. Almacenar y Recuperar Contexto| Neo4jClient
    Neo4jClient -->|Generar Vectores en CPU| LocalEmbedder
    Neo4jClient <-->|Consultas Cypher| AuraDB
    Orchestrator -->|2. Disparar extracción asíncrona| LearningLoop
    LearningLoop -->|3. Persistir preferencia| Neo4jClient
```

### Componentes Clave

1. **3 Agentes Especializados**:
   - **Generador de Imágenes** (puerto 10001): Usa CrewAI + Stable Diffusion XL (vía Hugging Face Inference API) para generar y editar imágenes basándose en prompts de texto.
   - **Asistente Médico** (puerto 10002): Analiza imágenes médicas (radiografías, resonancias) y realiza búsquedas complementarias con Tavily.
   - **Tutor Socrático de Física** (puerto 10003): Utiliza procesamiento de PDFs (base vectorial Qdrant) y enseña a los estudiantes mediante el método socrático formulando preguntas guía en base a la bibliografía oficial de Física I.

2. **Backend Orquestador** (puerto 12000):
   - Construido con FastAPI y **BeeAI Workflow**.
   - Analiza el contexto de la conversación y enruta dinámicamente las consultas del usuario al agente más calificado.
   - Integrado con **Neo4j Agent Memory** y un bucle de **Auto-Aprendizaje (Self-Learning)**:
     - **Memoria a Corto Plazo**: Registra el historial de interacciones en el grafo.
     - **Memoria a Largo Plazo**: Recupera las preferencias del usuario y las inyecta en los prompts del orquestador.
     - **Auto-Aprendizaje**: Analiza asíncronamente en segundo plano si el usuario expresó nombres, preferencias académicas o correcciones, guardándolas permanentemente en Neo4j Aura Cloud DB sin ralentizar las respuestas.
     - **Embeddings Locales**: Genera vectores localmente usando `SentenceTransformers` (modelo `BAAI/bge-small-en-v1.5`), eliminando costes y dependencias de claves de OpenAI.

3. **Frontend** (puerto 3000):
   - Aplicación Next.js React moderna y responsiva.
   - Dashboard de administración para monitorizar los agentes y el inspector del protocolo A2A.
   - Chat interactivo en tiempo real con soporte multimedia (texto, PDF e imágenes).

## 🧠 NAMS: Neo4j Agent Memory System

NAMS es un sistema avanzado de gestión de memoria para agentes inteligentes basado en **Neo4j Aura Cloud DB** y **SentenceTransformers** (ejecutado localmente en CPU). 

Su arquitectura divide la cognición en tres subsistemas clave:

### 1. Memoria a Corto Plazo (Conversacional)
* **Función**: Almacena cada intercambio de mensajes (User/Assistant) en una estructura de grafo dirigida por `session_id`.
* **Beneficio**: Permite al orquestador reconstruir el hilo completo del diálogo en formato conversacional nativo y recuperar los últimos mensajes para mantener la coherencia semántica inmediata.

### 2. Memoria a Largo Plazo (Perfil & Preferencias)
* **Función**: Almacena hechos aprendidos sobre el usuario (nombre, dificultades específicas de aprendizaje, velocidad preferida, etc.) como nodos de entidad vinculados a su usuario único.
* **Flujo**: Antes de despachar cada nueva consulta a un agente, el orquestador recupera las preferencias almacenadas y las inyecta dinámicamente como directrices contextuales (ej. `"El estudiante prefiere explicaciones matemáticas detalladas, se llama Diego y tiene dificultades en cinemática"`).

### 3. Bucle de Auto-Aprendizaje Asíncrono (Self-Learning)
* **Proceso**: Al finalizar cada respuesta al usuario, el orquestador dispara una rutina asíncrona en segundo plano:
  1. Envía el último intercambio al extractor de preferencias (alimentado por Groq LLM).
  2. Si se detecta un hecho valioso, corrección o dato de perfil, se genera una preferencia.
  3. La preferencia se vectoriza localmente usando el modelo `BAAI/bge-small-en-v1.5` de `SentenceTransformers` (vector de 384 dimensiones).
  4. Se persiste el nuevo nodo y vector en la base de datos en grafo Neo4j.
* **Beneficio**: Este análisis ocurre de manera completamente asíncrona en segundo plano sin añadir un solo milisegundo de latencia a las interacciones de chat en tiempo real del usuario.

---

## 📊 Evaluación con LangSmith (Métricas de Generación y Recuperación)

Para evaluar y monitorizar el rendimiento de los agentes, se pueden definir evaluadores en LangSmith enfocados en dos áreas clave:

### 1. Métricas de Generación (Generation Metrics)
Evalúan la calidad y veracidad del texto producido por los LLMs de los agentes.
*   **Evaluador de Correctitud / QA (Correctness / Accuracy):**
    *   *Propósito:* Mide si la respuesta generada por el agente es fácticamente correcta comparada con una respuesta de referencia (Ground Truth) de un dataset.
    *   *Implementación:* Se realiza mediante *LLM-as-a-judge* (usando razonamiento *Chain-of-Thought*). LangSmith provee evaluadores listos como `cot_qa` para contrastar `prediction` y `reference`.
*   **Evaluador de Fidelidad / Sin Alucinaciones (Faithfulness / Groundedness):**
    *   *Propósito:* Mide si la respuesta del modelo se basa exclusivamente en los fragmentos de contexto recuperados (evitando invenciones externas).
    *   *Implementación:* Un evaluador LLM personalizado que califica si cada afirmación de la respuesta generada es soportada directamente por el contexto provisto.

### 2. Métricas de Recuperación (Retrieval Metrics)
Evalúan el desempeño del buscador (retriever en Qdrant) al extraer los documentos relevantes.
*   **Evaluador de Relevancia del Contexto (Context Relevance):**
    *   *Propósito:* Determina si los fragmentos de texto o imágenes recuperados son verdaderamente relevantes para responder la consulta original.
    *   *Implementación:* Evaluador LLM asistido que lee la consulta del usuario y el contenido de los documentos recuperados para calcular un score de relevancia y descartar el "ruido".
*   **Métricas de Recuperación Clásicas (MAP / MRR / Recall@K):**
    *   *Propósito:* Evalúa matemáticamente la posición y cantidad de documentos esperados recuperados en el buscador.
        *   **Recall@K (Exhaustividad):** Proporción de documentos relevantes encontrados dentro de los primeros $K$ resultados.
        *   **MRR (Mean Reciprocal Rank):** Posición del primer documento relevante en la lista.
    *   *Implementación:* Mediante un evaluador personalizado en Python (`Custom RunEvaluator`) que extrae los metadatos de los runs de tipo `retriever` en LangSmith y los compara contra los IDs del dataset de prueba.

---

## Related Repositories

-   [A2A](https://github.com/a2aproject/A2A) - A2A Specification and documentation.
-   [a2a-python](https://github.com/a2aproject/a2a-python) - A2A Python SDK.
-   [a2a-inspector](https://github.com/a2aproject/a2a-inspector) - UI tool for inspecting A2A enabled agents.

## Contributing

Contributions welcome! See the [Contributing Guide](CONTRIBUTING.md).

## Getting help

Please use the [issues page](https://github.com/a2aproject/a2a-samples/issues) to provide suggestions, feedback or submit a bug report.

## Disclaimer

This repository itself is not an officially supported Google product. The code in this repository is for demonstrative purposes only.

Important: The sample code provided is for demonstration purposes and illustrates the mechanics of the Agent-to-Agent (A2A) protocol. When building production applications, it is critical to treat any agent operating outside of your direct control as a potentially untrusted entity.

All data received from an external agent—including but not limited to its AgentCard, messages, artifacts, and task statuses—should be handled as untrusted input. For example, a malicious agent could provide an AgentCard containing crafted data in its fields (e.g., description, name, skills.description). If this data is used without sanitization to construct prompts for a Large Language Model (LLM), it could expose your application to prompt injection attacks. Failure to properly validate and sanitize this data before use can introduce security vulnerabilities into your application.

Developers are responsible for implementing appropriate security measures, such as input validation and secure handling of credentials to protect their systems and users.

Recordar los .env en el root de cada directorio
