"""Crew AI based sample for A2A protocol with LangSmith monitoring.

Handles the agents and presents the tools required.
"""

import base64
import logging
import os
import re
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
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

load_dotenv()

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


@traceable(name="image_generation_internal", run_type="tool")
def _generate_image_internal(
    prompt: str, session_id: str, artifact_file_id: str = None
) -> str:
    """Internal image generation logic with LangSmith tracing."""
    if not prompt:
        raise ValueError('Prompt cannot be empty')

    client = genai.Client()
    cache = InMemoryCache()

    text_input = (
        prompt,
        'Ignore any input images if they do not match the request.',
    )

    ref_image = None
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

    # Get reference image from cache if exists
    try:
        ref_image_data = None
        session_image_data = cache.get(session_id)
        
        if session_image_data and artifact_file_id:
            try:
                ref_image_data = session_image_data[artifact_file_id]
                logger.info('Found reference image in prompt input')
                print(f'🔄 Using reference image: {artifact_file_id}')
            except Exception:
                ref_image_data = None
                
        if not ref_image_data and session_image_data:
            latest_image_key = list(session_image_data.keys())[-1]
            ref_image_data = session_image_data[latest_image_key]
            print(f'🔄 Using latest image as reference: {latest_image_key}')

        if ref_image_data:
            ref_bytes = base64.b64decode(ref_image_data.bytes)
            ref_image = Image.open(BytesIO(ref_bytes))
            
    except Exception as e:
        ref_image = None
        logger.warning(f"Could not load reference image: {e}")

    if ref_image:
        contents = [text_input, ref_image]
        print('🖼️ Generating with reference image')
    else:
        contents = text_input
        print('🆕 Generating new image')

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-preview-image-generation',
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            ),
        )
        print('✅ Response received from Gemini')
        
    except Exception as e:
        logger.error(f'Error generating image: {e}')
        print(f'❌ Generation error: {e}')
        
        # Log error to LangSmith
        if LANGSMITH_ENABLED:
            try:
                langsmith_client.create_feedback(
                    run_id=None,
                    key="image_generation_error",
                    value={"error": str(e), "prompt": prompt}
                )
            except:
                pass
        
        # ✅ Devolver mensaje de error claro en lugar de número
        return f"ERROR: {str(e)}"

    # Process response
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            try:
                print('Creating image data')
                data = Imagedata(
                    bytes=base64.b64encode(part.inline_data.data).decode('utf-8'),
                    mime_type=part.inline_data.mime_type,
                    name='generated_image.png',
                    id=uuid4().hex,
                )
                
                session_data = cache.get(session_id)
                if session_data is None:
                    cache.set(session_id, {data.id: data})
                else:
                    session_data[data.id] = data

                print(f'✅ Image generated with ID: {data.id}')
                
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
                                "mime_type": data.mime_type
                            }
                        )
                    except:
                        pass

                return data.id
                
            except Exception as e:
                logger.error(f'Error processing image: {e}')
                print(f'❌ Error processing image: {e}')
                return f"ERROR: {str(e)}"
    
    # ✅ Si no hay inline_data, devolver mensaje claro
    return "ERROR: No image data received from Gemini"


class ImageGenerationAgent:
    """Agent that generates images based on user prompts with LangSmith monitoring."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/png']

    def __init__(self):
        """Initialize the image generation agent with LangSmith tracing."""
        # Log agent initialization
        if LANGSMITH_ENABLED:
            logger.info(f"📊 LangSmith monitoring enabled - Project: {os.getenv('LANGCHAIN_PROJECT')}")
        
        if os.getenv('GOOGLE_GENAI_USE_VERTEXAI'):
            self.model = LLM(model='vertex_ai/gemini-2.5-flash')
        elif os.getenv('GOOGLE_API_KEY'):
            self.model = LLM(
                model='gemini/gemini-2.5-flash',
                api_key=os.getenv('GOOGLE_API_KEY'),
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

    @traceable(name="crew_execution", run_type="chain")
    def _execute_crew_with_tracing(self, inputs: dict) -> str:
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
        
        # Execute crew
        result = self.image_crew.kickoff(inputs)
        
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

    @traceable(name="generate_image_workflow", run_type="chain")
    def invoke(self, query, session_id) -> str:
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
            response = self._execute_crew_with_tracing(inputs)
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
                else:
                    error_msg = f"Invalid response format: {response_str}"
                    logger.error(error_msg)
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
