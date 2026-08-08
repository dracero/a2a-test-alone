"""Crew AI based sample for A2A protocol with LangSmith monitoring.

Handles the agents and presents the tools required.
"""

import base64
import logging
import os
import re
import urllib.parse
import urllib.request
from collections.abc import AsyncIterable
from io import BytesIO
from typing import Any
from uuid import uuid4


from app.in_memory_cache import InMemoryCache
# Import LangSmith configuration
from app.langsmith_config import (LANGSMITH_ENABLED, get_langsmith_status,
                                  langsmith_client, traceable)
from crewai import LLM, Agent, Crew, Task
from crewai.process import Process
from crewai.tools import tool
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from pydantic import BaseModel

# Monkey patch CrewAI cache_breakpoint issue for Groq/LiteLLM
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except Exception:
    pass

# Load .env from project root (5 levels up: agent.py -> app -> images -> agents -> python -> samples -> root)
root_dir = Path(__file__).resolve().parents[5]
env_path = root_dir / '.env'
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)


class Imagedata(BaseModel):
    """Represents image data.

    Attributes:
      id: Unique identifier for the image.
      name: Name of the image.
      mime_type: MIME type of the image.
      bytes: Base64 encoded image data.
      error: Error message if there was an issue with the image.
    """

    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    bytes: str | None = None
    error: str | None = None


@tool('image_generation_tool')
def generate_image_tool(
    prompt: str, session_id: str, artifact_file_id: str = ""
) -> str:
    """Generate or modify images based on a text prompt.
    
    Args:
        prompt: Description of the image to generate
        session_id: Session identifier for caching
        artifact_file_id: Optional ID of existing image to modify (leave empty for new images)
    
    Returns:
        Image ID string or error message starting with 'ERROR:'
    """
    # ✅ Convertir string vacío a None
    artifact_id = artifact_file_id if artifact_file_id and artifact_file_id.strip() else None
    return _generate_image_internal(prompt, session_id, artifact_id)


# List of models to try in order (fallback strategy)
# Using inference providers for better free-tier availability
_MODELS_TO_TRY = [
    # Provider-based models (more reliable, use HF's $0.10/mo free credits)
    {"model": "black-forest-labs/FLUX.1-schnell", "provider": "fal-ai", "label": "FLUX.1-schnell via fal"},
    {"model": "black-forest-labs/FLUX.1-schnell", "provider": "replicate", "label": "FLUX.1-schnell via replicate"},
    # Direct serverless (free tier, may have limits)
    {"model": "stabilityai/stable-diffusion-xl-base-1.0", "provider": None, "label": "SDXL direct"},
    {"model": "black-forest-labs/FLUX.1-schnell", "provider": None, "label": "FLUX.1-schnell direct"},
]


@traceable(name="image_generation_internal", run_type="tool", tags=["agent_type:image_generator", "image_generator", "a2a-agent"])
def _generate_image_internal(
    prompt: str, session_id: str, artifact_file_id: str = None
) -> str:
    """Internal image generation logic with LangSmith tracing using Hugging Face models with fallback."""
    if not prompt:
        raise ValueError('Prompt cannot be empty')

    cache = InMemoryCache()

    logger.info(f'Session id {session_id}')
    print(f'🎨 Generating image for session: {session_id}')

    # Log generation start to LangSmith
    if LANGSMITH_ENABLED:
        try:
            langsmith_client.create_feedback(
                run_id=None,
                key="image_generation_start",
                value={
                    "prompt": prompt,
                    "session_id": session_id,
                    "artifact_file_id": artifact_file_id
                }
            )
        except Exception as e:
            logger.debug(f"LangSmith feedback error: {e}")

    # Reference image logic
    if artifact_file_id:
        print(f'⚠️ Reference image {artifact_file_id} - editing not yet implemented')

    print('🆕 Generating new image with Hugging Face (multi-model fallback)')

    hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HF_TOKEN')
    
    if not hf_token:
        error_msg = 'HUGGINGFACEHUB_API_TOKEN or HF_TOKEN not set'
        logger.error(error_msg)
        print(f'❌ {error_msg}')
        return f"ERROR: {error_msg}"
    
    # Enhanced prompt for better image quality
    enhanced_prompt = (
        f"{prompt}, "
        f"high quality, detailed, professional, 4k, masterpiece"
    )
    
    print(f'📝 Enhanced prompt: {enhanced_prompt}')
    
    from huggingface_hub import InferenceClient
    
    last_error = None
    
    for model_config in _MODELS_TO_TRY:
        model_id = model_config["model"]
        provider = model_config["provider"]
        label = model_config["label"]
        
        try:
            print(f'🚀 Trying: {label}...')
            
            # Create client with or without provider
            if provider:
                client = InferenceClient(provider=provider, api_key=hf_token)
            else:
                client = InferenceClient(token=hf_token)
            
            # Build kwargs based on model type
            kwargs = {
                "prompt": enhanced_prompt,
                "model": model_id,
            }
            
            # FLUX models don't support negative_prompt or some params
            if "FLUX" not in model_id:
                kwargs["num_inference_steps"] = 50
                kwargs["guidance_scale"] = 7.5
                kwargs["negative_prompt"] = "blurry, bad quality, watermark, text, signature, low resolution"
            
            # Make request to Hugging Face
            image = client.text_to_image(**kwargs)
            
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data_bytes = img_byte_arr.getvalue()
            
            if not image_data_bytes or len(image_data_bytes) < 100:
                print(f'⚠️ {label}: Empty or too small response, trying next...')
                last_error = f'{label}: No valid image data'
                continue
            
            print(f'✅ Image data received from {label}: {len(image_data_bytes)} bytes')
            
            # Store image in cache
            data = Imagedata(
                bytes=base64.b64encode(image_data_bytes).decode('utf-8'),
                mime_type='image/png',
                name='generated_image.png',
                id=uuid4().hex,
            )
            
            session_data = cache.get(session_id)
            if session_data is None:
                cache.set(session_id, {data.id: data})
            else:
                session_data[data.id] = data

            print(f'✅ Image generated with ID: {data.id} (model: {label})')
            
            # Log success to LangSmith
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="image_generation_success",
                        value={
                            "image_id": data.id,
                            "prompt": prompt,
                            "session_id": session_id,
                            "mime_type": data.mime_type,
                            "model": label
                        }
                    )
                except:
                    pass

            return data.id
            
        except Exception as e:
            error_str = str(e)
            # Detect specific errors that mean "try next model"
            is_credits_error = any(kw in error_str.lower() for kw in [
                'credit', 'depleted', 'quota', 'rate limit', '429', '402',
                'billing', 'payment', 'subscription', 'pro ', 'upgrade'
            ])
            is_model_error = any(kw in error_str.lower() for kw in [
                'model is not available', 'not found', '404', 'loading',
                'service unavailable', '503', 'overloaded'
            ])
            
            if is_credits_error or is_model_error:
                print(f'⚠️ {label}: {error_str[:120]}... trying next model')
            else:
                print(f'⚠️ {label} failed: {error_str[:120]}... trying next model')
            
            last_error = f'{label}: {error_str}'
            continue
    
    # All models failed
    error_msg = f'All image generation models failed. Last error: {last_error}'
    logger.error(error_msg)
    print(f'❌ {error_msg}')
    
    # Log error to LangSmith
    if LANGSMITH_ENABLED:
        try:
            langsmith_client.create_feedback(
                run_id=None,
                key="image_generation_error",
                value={"error": error_msg, "prompt": prompt}
            )
        except:
            pass
    
    return f"ERROR: {error_msg}"


class ImageGenerationAgent:
    """Agent that generates images based on user prompts with LangSmith monitoring."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/png']

    def __init__(self):
        """Initialize the image generation agent with LangSmith tracing."""
        # Log agent initialization
        if LANGSMITH_ENABLED:
            logger.info(f"📊 LangSmith monitoring enabled - Project: {os.getenv('LANGCHAIN_PROJECT')}")
        
        # Usar Groq para el razonamiento del agente
        # CrewAI usa LiteLLM internamente, el formato correcto es: groq/<model>
        from crewai import LLM as CrewAILLM
        self.model = CrewAILLM(
            model='groq/llama-3.1-8b-instant',  # Modelo rápido y con límites altos en Groq
            api_key=os.getenv('GROQ_API_KEY'),
        )

        self.image_creator_agent = Agent(
            role='Image Generation Specialist',
            goal=(
                "Generate images using the image_generation_tool based on user requests. "
                "Always call the tool with the correct parameters: prompt, session_id, and artifact_file_id."
            ),
            backstory=(
                "You are an AI assistant specialized in image generation. "
                "You have access to a powerful image generation tool that creates images from text descriptions. "
                "Your job is to understand the user's request and call the image_generation_tool with the right parameters. "
                "Always return the exact image ID that the tool provides."
            ),
            verbose=True,  # ✅ Cambiar a True para ver qué está pasando
            allow_delegation=False,
            tools=[generate_image_tool],
            llm=self.model,
        )

        self.image_creation_task = Task(
            description=(
                "Receive a user prompt: '{user_prompt}'.\n"
                "Your job is to analyze the prompt and create or modify an image.\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. Use the 'image_generation_tool' to generate images\n"
                "2. Pass these exact parameters:\n"
                "   - prompt: The user's request (required)\n"
                "   - session_id: Use '{session_id}' (required)\n"
                "   - artifact_file_id: Use '{artifact_file_id}' if provided, otherwise use empty string '' (optional)\n\n"
                "3. If the user mentions 'this image', 'that image', or similar references, "
                "include context in your prompt to the tool\n\n"
                "4. The tool will return an image ID on success or an error message starting with 'ERROR:'\n\n"
                "Example tool call:\n"
                "Action: image_generation_tool\n"
                "Action Input: {{\n"
                '  "prompt": "a cute cow in a field",\n'
                '  "session_id": "{session_id}",\n'
                '  "artifact_file_id": "{artifact_file_id}"\n'
                "}}"
            ),
            expected_output='The ID of the generated image (a 32-character hexadecimal string) or an error message',
            agent=self.image_creator_agent,
        )

        self.image_crew = Crew(
            agents=[self.image_creator_agent],
            tasks=[self.image_creation_task],
            process=Process.sequential,
            verbose=True,  # ✅ Cambiar a True para debugging
        )

    def extract_artifact_file_id(self, query):
        """Extract artifact file ID from query string."""
        try:
            pattern = r'(?:id|artifact-file-id)\s+([0-9a-f]{32})'
            match = re.search(pattern, query)

            if match:
                return match.group(1)
            return ""  # ✅ Devolver string vacío en lugar de None
        except Exception:
            return ""

    @traceable(name="crew_execution", run_type="chain", tags=["agent_type:image_generator", "image_generator", "a2a-agent"])
    async def _execute_crew_with_tracing(self, inputs: dict) -> str:
        """Execute crew with LangSmith tracing."""
        # Log crew start
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="crew_start",
                    value={
                        "inputs": inputs,
                        "agents": [self.image_creator_agent.role],
                        "tasks_count": len(self.image_crew.tasks)
                    }
                )
            except:
                pass
        
        # Execute crew asynchronously
        result = await self.image_crew.kickoff_async(inputs)
        
        # Log crew completion
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="crew_completion",
                    value={
                        "result": str(result),
                        "success": True
                    }
                )
            except:
                pass
        
        return result

    @traceable(name="generate_image_workflow", run_type="chain", tags=["agent_type:image_generator", "image_generator", "a2a-agent"])
    async def invoke(self, query, session_id) -> str:
        """Kickoff CrewAI and return the response with LangSmith monitoring."""
        artifact_file_id = self.extract_artifact_file_id(query)

        inputs = {
            'user_prompt': query,
            'session_id': session_id,
            'artifact_file_id': artifact_file_id if artifact_file_id else "",  # ✅ Asegurar string vacío
        }
        
        logger.info(f'Inputs {inputs}')
        print(f'🚀 Starting generation with CrewAI...')
        print(f'📝 Prompt: {query}')
        print(f'🔑 Session ID: {session_id}')
        print(f'🎯 Artifact ID: {artifact_file_id if artifact_file_id else "none"}')
        
        if LANGSMITH_ENABLED:
            print(f'📊 LangSmith monitoring active')
        
        try:
            response = await self._execute_crew_with_tracing(inputs)
            print(f'✅ Crew completed')
            print(f'📤 Raw response: {response}')
            print(f'📤 Response type: {type(response)}')
            
            # ✅ Validar si la respuesta es None o vacía
            if response is None:
                error_msg = "Crew returned None - no response generated"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
            
            # ✅ Extraer el resultado correcto
            # CrewAI puede devolver diferentes tipos de objetos
            if hasattr(response, 'raw'):
                response_str = str(response.raw).strip()
            elif hasattr(response, 'result'):
                response_str = str(response.result).strip()
            else:
                response_str = str(response).strip()
            
            print(f'📤 Processed response: {response_str}')
            
            if not response_str:
                error_msg = "Crew returned empty response"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
            
            # ✅ Validar que sea un ID válido (32 caracteres hexadecimales) o un error
            if response_str.startswith("ERROR:"):
                return response_str
            elif re.match(r'^[0-9a-f]{32}$', response_str):
                print(f'✅ Valid image ID returned: {response_str}')
                
                # Log workflow success
                if LANGSMITH_ENABLED:
                    try:
                        langsmith_client.create_feedback(
                            run_id=None,
                            key="workflow_success",
                            value={
                                "query": query,
                                "session_id": session_id,
                                "result": response_str
                            }
                        )
                    except:
                        pass
                
                return response_str
            else:
                # Si no es un ID válido ni un error, buscar el ID en la respuesta
                id_match = re.search(r'([0-9a-f]{32})', response_str)
                if id_match:
                    image_id = id_match.group(1)
                    print(f'✅ Extracted image ID from response: {image_id}')
                    return image_id
                
                # Detectar si el LLM devolvió formato de llamada a función como texto (ej: function=image_generation_tool>{...}</function>)
                if 'image_generation_tool' in response_str or 'prompt' in response_str:
                    logger.info("Detectada llamada a herramienta emitida como texto por el LLM. Procesando e invocando generador...")
                    import json
                    json_match = re.search(r'\{[^{}]*"prompt"[^{}]*\}', response_str, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    
                    if json_match:
                        try:
                            tool_args = json.loads(json_match.group(0))
                            tool_prompt = tool_args.get("prompt", query)
                            tool_session_id = tool_args.get("session_id", session_id) or session_id
                            tool_artifact_id = tool_args.get("artifact_file_id") or artifact_file_id or None
                            if tool_artifact_id and not str(tool_artifact_id).strip():
                                tool_artifact_id = None
                            
                            print(f'🔧 Fallback por texto de herramienta: ejecutando _generate_image_internal(prompt="{tool_prompt}")')
                            gen_res = _generate_image_internal(tool_prompt, tool_session_id, tool_artifact_id)
                            if gen_res:
                                return gen_res
                        except Exception as parse_err:
                            logger.warning(f"Error parseando JSON de llamada a herramienta: {parse_err}")

                # Intentar fallback de generación directa con la consulta recibida
                print(f'🔧 Fallback final: ejecutando generación directa con query="{query}"')
                fallback_res = _generate_image_internal(query, session_id, artifact_file_id if artifact_file_id else None)
                if fallback_res:
                    return fallback_res

                # Check if the LLM returned an error message about credits/limits
                credits_keywords = ['credit', 'depleted', 'quota', 'limit', 'billing', 'subscription', 'upgrade', 'payment']
                is_credits_issue = any(kw in response_str.lower() for kw in credits_keywords)
                
                if is_credits_issue:
                    error_msg = "Image generation service temporarily unavailable (credits/quota issue). Please try again later."
                else:
                    truncated = response_str[:200] + '...' if len(response_str) > 200 else response_str
                    error_msg = f"Image generation failed: {truncated}"
                
                logger.error(f"Invalid response format: {response_str[:300]}")
                print(f'❌ {error_msg}')
                return f"ERROR: {error_msg}"
            
        except Exception as e:
            logger.error(f'Error in crew execution: {e}')
            print(f'❌ Crew error: {e}')
            
            # Log workflow error
            if LANGSMITH_ENABLED:
                try:
                    langsmith_client.create_feedback(
                        run_id=None,
                        key="workflow_error",
                        value={
                            "query": query,
                            "session_id": session_id,
                            "error": str(e)
                        }
                    )
                except:
                    pass
            
            return f"ERROR: {str(e)}"

    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not supported by CrewAI."""
        raise NotImplementedError('Streaming is not supported by CrewAI.')

    def get_image_data(self, session_id: str, image_key: str) -> Imagedata:
        """Return Imagedata given a key. This is a helper method from the agent."""
        cache = InMemoryCache()
        session_data = cache.get(session_id)
        
        print(f'🔍 Looking for image: {image_key} in session: {session_id}')
        print(f'📦 Session data exists: {session_data is not None}')
        
        if session_data:
            print(f'📦 Available image IDs: {list(session_data.keys())}')
        
        try:
            if session_data is None:
                logger.error(f'No session data found for session_id: {session_id}')
                return Imagedata(error='Session not found, please try again.')
            
            if image_key not in session_data:
                logger.error(f'Image key {image_key} not found in session {session_id}')
                return Imagedata(error=f'Image {image_key} not found in session.')
            
            image_data = session_data[image_key]
            print(f'✅ Image data found: {image_data.id}')
            return image_data
            
        except (KeyError, TypeError) as e:
            logger.error(f'Error getting image data: {e}')
            return Imagedata(error='Error retrieving image, please try again.')
    
    def get_langsmith_status(self) -> dict:
        """Get current LangSmith configuration status."""
        return get_langsmith_status()
