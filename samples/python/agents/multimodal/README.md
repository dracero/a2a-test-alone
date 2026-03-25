# Physics Multimodal Agent

Virtual physics professor specialized in multimodal analysis (text + images) with vector search capabilities.

## Features

- Physics concept explanations
- Scientific image analysis
- Vector search in academic documents (Qdrant)
- PDF processing and indexing
- Conversational memory
- Streaming responses
- UBA Physics I curriculum alignment

## Requirements

- Python >= 3.13
- Google API Key (Gemini)
- Qdrant instance (cloud or local)
- PyTorch with CUDA support

## Configuration

Set the following environment variables in the root `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key
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

## Capabilities

- Explain physics concepts (mechanics, thermodynamics, etc.)
- Analyze experimental diagrams and scientific images
- Search through academic PDF documents
- Provide educational examples and explanations
- Classify topics according to UBA curriculum
- Process and index PDF documents automatically

## PDF Processing

On first run, the agent will:
1. Check if Qdrant collections exist
2. If not, process PDFs from `PDF_DIR`
3. Extract text and images from PDFs
4. Create vector embeddings
5. Store in Qdrant for fast retrieval

Subsequent runs will skip processing if collections already exist.
