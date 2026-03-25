# Image Generation Agent

Agent for generating and editing images using Google Gemini and CrewAI.

## Features

- Generate high-quality images from text descriptions
- Edit existing images
- LangSmith monitoring support
- A2A protocol integration

## Requirements

- Python >= 3.13
- Google API Key (Gemini)

## Configuration

Set the following environment variables in the root `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_key  # Optional
```

## Usage

```bash
cd samples/python/agents/images
uv run python -m app
```

Or from the project root:

```bash
npm run dev:agent:images
```

The agent will start on `http://localhost:10001`
