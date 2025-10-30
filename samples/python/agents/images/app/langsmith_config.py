"""LangSmith configuration module for A2A CrewAI agent."""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def setup_langsmith_environment():
    """Configure environment variables for LangSmith tracing.
    
    Returns:
        tuple: (enabled: bool, traceable: callable, client: Client|None)
    """
    # LangSmith configuration
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY", ""),
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "image_generation_a2a")
    }
    
    # Only enable if API key is present
    if not langsmith_config["LANGCHAIN_API_KEY"]:
        logger.info("⚠️ LANGCHAIN_API_KEY not found - LangSmith disabled")
        return False, _dummy_traceable, None
    
    # Set environment variables
    for key, value in langsmith_config.items():
        os.environ[key] = value
        logger.info(f"✅ {key} configured")
    
    try:
        # Install langsmith if needed
        logger.info("📦 Checking LangSmith dependencies...")
        #subprocess.check_call(
        #    [sys.executable, "-m", "pip", "install", "langsmith", "--quiet"]
        #)

        # Import after environment setup
        from langsmith import Client, traceable

        # Verify connection
        client = Client()
        logger.info(f"🔗 Connected to LangSmith - Project: {os.environ['LANGCHAIN_PROJECT']}")
        
        return True, traceable, client
        
    except Exception as e:
        logger.warning(f"⚠️ Error setting up LangSmith: {e}")
        logger.info("💡 System will run without LangSmith monitoring")
        return False, _dummy_traceable, None


def _dummy_traceable(*args, **kwargs):
    """Dummy decorator when LangSmith is not available."""
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


def get_langsmith_status():
    """Get current LangSmith configuration status."""
    if os.getenv("LANGCHAIN_TRACING_V2") != "true":
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
