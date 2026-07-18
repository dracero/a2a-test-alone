"""
Re-exporta el rotador de API Keys desde el módulo raíz del proyecto.

Este wrapper permite usar imports relativos dentro del paquete server:
    from .api_key_rotator import google_key_rotator, invoke_with_retry
"""

import sys
import os

# Agregar el directorio raíz del proyecto al path para poder importar api_key_rotator
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from api_key_rotator import (  # noqa: E402, F401
    GoogleApiKeyRotator,
    google_key_rotator,
    create_google_llm,
    invoke_with_retry,
    ainvoke_with_retry,
    _is_quota_error,
)

__all__ = [
    "GoogleApiKeyRotator",
    "google_key_rotator",
    "create_google_llm",
    "invoke_with_retry",
    "ainvoke_with_retry",
    "_is_quota_error",
]
