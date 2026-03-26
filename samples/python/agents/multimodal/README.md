# Tutor Socrático de Física Multimodal

Tutor socrático de Física I que usa el método socrático para guiar el aprendizaje. Recibe texto e imágenes, hace 3 preguntas guía y luego da la respuesta completa.

## Features

- **Método socrático**: 3 preguntas guía antes de la respuesta completa
- Análisis multimodal (texto + imágenes de experimentos/diagramas)
- Búsqueda vectorial en documentos académicos (Qdrant)
- Procesamiento e indexación de PDFs
- Memoria conversacional
- Streaming responses
- Alineado al temario de Física I - UBA

## Requirements

- Python >= 3.13
- Groq API Key (Llama 4)
- Qdrant instance (cloud or local)
- PyTorch with CUDA support

## Configuration

Set the following environment variables in the root `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_KEY=your_qdrant_api_key
PDF_DIR=/path/to/pdf/documents  # Optional, defaults to /content
```

## Usage

```bash
cd samples/python/agents/multimodal
uv run python -m app
```

Or from the project root:

```bash
npm run dev:agent:multimodal
```

The agent will start on `http://localhost:10003`

## How it works

1. **El estudiante envía una consulta** (texto y/o imagen)
2. **El orquestador** detecta que es una consulta de física y la delega al tutor
3. **El tutor hace 3 preguntas socráticas** para guiar el pensamiento
4. **El estudiante responde** cada pregunta
5. **El tutor genera la respuesta completa** incorporando las respuestas del estudiante

## PDF Processing

On first run, the agent will:
1. Check if Qdrant collections exist
2. If not, process PDFs from `PDF_DIR`
3. Extract text and images from PDFs
4. Create vector embeddings
5. Store in Qdrant for fast retrieval

Subsequent runs will skip processing if collections already exist.
