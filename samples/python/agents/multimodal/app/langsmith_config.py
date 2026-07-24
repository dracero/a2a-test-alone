"""LangSmith configuration module for A2A Physics Multimodal Tutor Agent."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_NAME = "a2a-multimodal-tutor"
DEFAULT_TAGS = ["agent_type:multimodal_tutor", "multimodal_tutor", "a2a-agent"]

LANGSMITH_ENABLED = False
traceable = None
langsmith_client = None


def _dummy_traceable(*args, **kwargs):
    """Dummy decorator when LangSmith is not available."""
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


def setup_langsmith_environment(project_name=DEFAULT_PROJECT_NAME):
    """Configure environment variables for LangSmith tracing for Physics Multimodal Tutor.
    
    Returns:
        tuple: (enabled: bool, traceable: callable, client: Client|None)
    """
    global LANGSMITH_ENABLED, traceable, langsmith_client

    # Auto load .env if API key is not present in environment
    if not os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
        try:
            from dotenv import load_dotenv
            curr = Path(__file__).resolve()
            for p in [curr.parents[i] for i in range(1, len(curr.parents))]:
                env_file = p / '.env'
                if env_file.exists():
                    load_dotenv(dotenv_path=env_file, override=False)
                    break
        except Exception:
            pass

    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY", "")
    target_project = os.getenv("LANGCHAIN_PROJECT") or project_name
    
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": api_key,
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_PROJECT": target_project
    }
    
    # Only enable if API key is present
    if not langsmith_config["LANGCHAIN_API_KEY"]:
        logger.info("⚠️ LANGCHAIN_API_KEY not found - LangSmith disabled for Physics Tutor")
        LANGSMITH_ENABLED = False
        traceable = _dummy_traceable
        langsmith_client = None
        return False, _dummy_traceable, None
    
    # Set environment variables
    for key, value in langsmith_config.items():
        os.environ[key] = value
        logger.info(f"✅ {key} configured -> {value}")
    
    try:
        from langsmith import Client, traceable as _ls_traceable
        client = Client()
        logger.info(f"🔗 Connected to LangSmith - Project: {os.environ['LANGCHAIN_PROJECT']}")
        
        LANGSMITH_ENABLED = True
        traceable = _ls_traceable
        langsmith_client = client
        return True, _ls_traceable, client
        
    except Exception as e:
        logger.warning(f"⚠️ Error setting up LangSmith for Physics Tutor: {e}")
        logger.info("💡 System will run without LangSmith monitoring")
        LANGSMITH_ENABLED = False
        traceable = _dummy_traceable
        langsmith_client = None
        return False, _dummy_traceable, None


def get_langsmith_status():
    """Get current LangSmith configuration status."""
    if os.getenv("LANGCHAIN_TRACING_V2") != "true" or not os.getenv("LANGCHAIN_API_KEY"):
        return {
            "enabled": False,
            "message": "LangSmith not configured"
        }
    
    return {
        "enabled": True,
        "project": os.getenv("LANGCHAIN_PROJECT"),
        "endpoint": os.getenv("LANGCHAIN_ENDPOINT"),
        "tracing": os.getenv("LANGCHAIN_TRACING_V2")
    }


# Initialize LangSmith on module import
LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith_environment()
