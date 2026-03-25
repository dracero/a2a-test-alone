# Medical Images Agent

Medical assistant specialized in analyzing medical images with search capabilities.

## Features

- Medical image analysis using Google Gemini
- Medical information search using Tavily
- Conversational memory
- Streaming responses
- Push notifications support

## Requirements

- Python >= 3.13
- Google API Key (Gemini)
- Tavily API Key (for medical search)

## Configuration

Set the following environment variables in the root `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

```bash
cd samples/python/agents/medical_Images
uv run python -m app
```

Or from the project root:

```bash
npm run dev:agent:medical
```

The agent will start on `http://localhost:10002`

## Capabilities

- Analyze X-rays, MRIs, CT scans, and other medical images
- Search for medical information and research
- Provide clinical evaluations based on visual findings
- Maintain conversation context for follow-up questions
