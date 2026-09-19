"""
Rotador de API Keys de Google — prioriza la key paga y conmuta con cooldown ante 403/429.

Módulo standalone (sin dependencias de proyecto) que puede ser importado
desde cualquier agente o servicio.

Uso:
    from api_key_rotator import google_key_rotator, invoke_with_retry, create_google_llm

    key = google_key_rotator.get_key()
    google_key_rotator.report_failure(key)

    llm = create_google_llm()
    response = invoke_with_retry(llm, messages)
"""

import os
import time
import threading
import logging
from typing import Optional, Any

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def _extract_str_key(key_obj: Any) -> str:
    """Extrae una string pura incluso si key_obj es un SecretStr o None."""
    if not key_obj:
        return ""
    if hasattr(key_obj, "get_secret_value"):
        return key_obj.get_secret_value()
    return str(key_obj)


# ── Strategy Pattern (GoF): Estrategias de Selección de Keys ──────

class KeySelectionStrategy(ABC):
    """Interfaz abstracta para estrategias de selección de API keys (Patrón Strategy GoF)."""

    @abstractmethod
    def select_key(
        self,
        keys: list[str],
        cooldowns: dict[str, float],
        paid_cooldown: float,
        free_cooldown: float,
    ) -> tuple[Optional[str], Optional[int]]:
        """Selecciona una key disponible y retorna (key, index)."""
        pass


class PriorityPaidKeyStrategy(KeySelectionStrategy):
    """Estrategia prioritaria: favorece siempre la primera key (paga).
    
    Solo si la principal entra en cooldown conmuta a las secundarias (gratuitas).
    Si todas están en cooldown, fuerza la primera key (paga) por tener cooldown menor.
    """

    def select_key(
        self,
        keys: list[str],
        cooldowns: dict[str, float],
        paid_cooldown: float,
        free_cooldown: float,
    ) -> tuple[Optional[str], Optional[int]]:
        if not keys:
            return None, None

        now = time.time()
        for idx, key in enumerate(keys):
            cooldown_ts = cooldowns.get(key, 0)
            cooldown_limit = paid_cooldown if idx == 0 else free_cooldown
            if now - cooldown_ts >= cooldown_limit:
                cooldowns.pop(key, None)
                return key, idx

        # Si todas están en cooldown, forzar la primera (paga)
        first_key = keys[0]
        cooldowns.pop(first_key, None)
        print(f"⚠️ Todas las keys en cooldown. Forzando uso de la key paga ...{first_key[-4:]}")
        return first_key, 0


class RoundRobinKeyStrategy(KeySelectionStrategy):
    """Estrategia Round-Robin: distribuye las solicitudes equitativamente entre keys activas."""

    def __init__(self):
        self._current_index = 0
        self._lock = threading.Lock()

    def select_key(
        self,
        keys: list[str],
        cooldowns: dict[str, float],
        paid_cooldown: float,
        free_cooldown: float,
    ) -> tuple[Optional[str], Optional[int]]:
        if not keys:
            return None, None

        with self._lock:
            total = len(keys)
            now = time.time()
            for offset in range(total):
                idx = (self._current_index + offset) % total
                key = keys[idx]
                cooldown_ts = cooldowns.get(key, 0)
                cooldown_limit = paid_cooldown if idx == 0 else free_cooldown
                if now - cooldown_ts >= cooldown_limit:
                    cooldowns.pop(key, None)
                    self._current_index = (idx + 1) % total
                    return key, idx

            # Si todas están en cooldown, forzar la primera key
            first_key = keys[0]
            cooldowns.pop(first_key, None)
            self._current_index = (0 + 1) % total
            return first_key, 0


class GoogleApiKeyRotator:
    """Rotator para múltiples Google API Keys con soporte para el patrón Strategy (GoF).

    Por defecto usa PriorityPaidKeyStrategy (prioriza la primera key paga y conmuta
    a secundarias con cooldown ante 403/429). Permite configurar estrategias alternativas
    como RoundRobinKeyStrategy.
    """

    COOLDOWN_SECONDS = 60
    PAID_KEY_COOLDOWN_SECONDS = 15

    def __init__(
        self,
        env_var: str = "GOOGLE_API_KEYS",
        fallback_var: str = "GOOGLE_API_KEY",
        strategy: Optional[KeySelectionStrategy] = None,
    ):
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._cooldowns: dict[str, float] = {}
        self._env_var = env_var
        self._fallback_var = fallback_var
        self._loaded = False
        self._strategy: KeySelectionStrategy = strategy or PriorityPaidKeyStrategy()
        
        # Intentar carga inicial
        self.load_keys()

    def set_strategy(self, strategy: KeySelectionStrategy):
        """Inyecta una estrategia de selección de keys en caliente (Patrón Strategy)."""
        with self._lock:
            self._strategy = strategy
            print(f"🔄 Estrategia de selección cambiada a: {strategy.__class__.__name__}")

    def load_keys(self):
        """Carga las API keys desde las variables de entorno."""
        raw = os.getenv(self._env_var, "")
        if raw:
            self._keys = [k.strip() for k in raw.split(",") if k.strip()]

        if not self._keys:
            fallback = os.getenv(self._fallback_var, "")
            if fallback:
                self._keys = [fallback.strip()]

        if self._keys:
            self._loaded = True
            print(
                f"🔑 GoogleApiKeyRotator: {len(self._keys)} key(s) cargadas "
                f"[{', '.join(f'...{k[-4:]}' for k in self._keys)}]"
            )
        else:
            if self._loaded:
                print("⚠️ GoogleApiKeyRotator: Las keys cargadas anteriormente se han perdido o vaciado.")

    @property
    def total_keys(self) -> int:
        if not self._keys:
            self.load_keys()
        return len(self._keys)

    def get_key(self) -> str:
        """Devuelve una key disponible delegando en la estrategia configurada (Patrón Strategy)."""
        if not self._keys:
            self.load_keys()
        if not self._keys:
            return ""

        with self._lock:
            key, idx = self._strategy.select_key(
                self._keys,
                self._cooldowns,
                self.PAID_KEY_COOLDOWN_SECONDS,
                self.COOLDOWN_SECONDS,
            )
            if key:
                print(f"🔑 Usando Google API Key ...{key[-4:]}")
                return key
            return ""

    def report_failure(self, key: Any):
        """Marca una key como fallida (cooldown diferenciado: paga vs. free)."""
        key_str = _extract_str_key(key)
        if not key_str or key_str not in self._keys:
            return
        with self._lock:
            self._cooldowns[key_str] = time.time()
            idx = self._keys.index(key_str)
            effective_cd = self.PAID_KEY_COOLDOWN_SECONDS if idx == 0 else self.COOLDOWN_SECONDS
            print(f"🚫 Key ...{key_str[-4:]} en cooldown por {effective_cd}s")

    def clear_cooldowns(self):
        with self._lock:
            self._cooldowns.clear()


# ── Singleton ──────────────────────────────────────────────────────
google_key_rotator = GoogleApiKeyRotator()


def create_google_llm(
    model: str = "gemini-3.5-flash",
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
    rotator: Optional[GoogleApiKeyRotator] = None,
):
    """Crea un ChatGoogleGenerativeAI con la siguiente key disponible."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if rotator is None:
        rotator = google_key_rotator
    key = rotator.get_key()
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_tokens=max_output_tokens,
        google_api_key=key,
        max_retries=1,
    )


# ── Detección de errores de cuota o claves inválidas ───────────────
def _is_quota_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    return any(
        ind in err_str
        for ind in [
            "400", "403", "429", "resource_exhausted", "resourceexhausted",
            "rate limit", "rate_limit", "quota", "too many requests",
            "api_key_invalid", "api key not valid", "invalid api key",
            "prepayment credits are depleted", "invalid argument provided to gemini",
            "503", "unavailable", "high demand", "overloaded", "500", "502", "504",
            "server error", "servererror", "deadline_exceeded",
        ]
    )


def _rebuild_llm(llm: Any, new_key: str):
    """Re-crea un ChatGoogleGenerativeAI con una nueva key reutilizando create_google_llm."""
    model = getattr(llm, "model_name", None) or getattr(llm, "model", "gemini-3.5-flash")
    temperature = getattr(llm, "temperature", 0.3)
    max_output = getattr(llm, "max_output_tokens", 8192)
    # Crear un rotator ad-hoc de una sola key para reusar create_google_llm
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output,
        max_tokens=max_output,
        google_api_key=new_key,
        max_retries=1,
    )


def normalize_llm_response(response: Any) -> Any:
    """Normaliza response.content a str si es una lista (formato multimodal o LangChain GenAI 2.0+)."""
    if response is None or not hasattr(response, "content"):
        return response

    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(str(part["text"]))
                elif "text" in part:
                    text_parts.append(str(part["text"]))
                elif "content" in part:
                    text_parts.append(str(part["content"]))
                elif part.get("type") == "thought":
                    continue
                else:
                    text_parts.append(str(part))
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
            else:
                text_parts.append(str(part))
        response.content = "".join(text_parts).strip()
    elif not isinstance(content, str):
        response.content = str(content)
    return response


def invoke_with_retry(llm: Any, messages: Any, *, rotator: Optional[GoogleApiKeyRotator] = None, max_retries: int = 0, base_wait: float = 2.0):
    """Invoca LLM con retry automático ante 403/429, rotando la API key."""
    if rotator is None:
        rotator = google_key_rotator
    if max_retries == 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = _extract_str_key(getattr(llm, "google_api_key", ""))

    for attempt in range(max_retries + 1):
        try:
            res = llm.invoke(messages)
            return normalize_llm_response(res)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            print(f"⚠️ [Retry {attempt+1}/{max_retries}] Cuota excedida con key ...{current_key[-4:] if current_key else '????'}: {str(exc)[:200]}")
            if attempt >= max_retries:
                break
            if current_key:
                rotator.report_failure(current_key)
            new_key = rotator.get_key()
            llm = _rebuild_llm(llm, new_key)
            current_key = new_key
            wait = base_wait * (2 ** attempt)
            print(f"⏳ Esperando {wait:.1f}s antes de reintentar...")
            time.sleep(wait)

    raise last_exc


async def ainvoke_with_retry(llm: Any, messages: Any, *, rotator: Optional[GoogleApiKeyRotator] = None, max_retries: int = 0, base_wait: float = 2.0):
    """Versión async de invoke_with_retry."""
    import asyncio
    if rotator is None:
        rotator = google_key_rotator
    if max_retries == 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = _extract_str_key(getattr(llm, "google_api_key", ""))

    for attempt in range(max_retries + 1):
        try:
            res = await llm.ainvoke(messages)
            return normalize_llm_response(res)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            print(f"⚠️ [Async Retry {attempt+1}/{max_retries}] Cuota excedida con key ...{current_key[-4:] if current_key else '????'}: {str(exc)[:200]}")
            if attempt >= max_retries:
                break
            if current_key:
                rotator.report_failure(current_key)
            new_key = rotator.get_key()
            llm = _rebuild_llm(llm, new_key)
            current_key = new_key
            wait = base_wait * (2 ** attempt)
            print(f"⏳ Esperando {wait:.1f}s antes de reintentar...")
            await asyncio.sleep(wait)

    raise last_exc


# ── Utilidades compartidas ─────────────────────────────────────────

def sync_env_key(rotator: Optional[GoogleApiKeyRotator] = None) -> str:
    """Obtiene la key activa del rotador y sincroniza os.environ para LiteLLM/SDKs externos.

    Centraliza el patrón repetido:
        active_key = google_key_rotator.get_key()
        os.environ["GEMINI_API_KEY"] = active_key
        os.environ["GOOGLE_API_KEY"] = active_key

    Returns:
        La key activa (puede ser string vacío si no hay keys).
    """
    if rotator is None:
        rotator = google_key_rotator
    active_key = rotator.get_key()
    if active_key:
        os.environ["GEMINI_API_KEY"] = active_key
        os.environ["GOOGLE_API_KEY"] = active_key
    return active_key


# ── Chain of Responsibility Pattern (GoF): Sanitización de Contexto ──

class ContextFilterHandler(ABC):
    """Manejador abstracto de la Cadena de Responsabilidad (GoF Chain of Responsibility)."""

    def __init__(self, next_handler: Optional["ContextFilterHandler"] = None):
        self._next_handler: Optional["ContextFilterHandler"] = next_handler

    def set_next(self, handler: "ContextFilterHandler") -> "ContextFilterHandler":
        """Configura el siguiente manejador en la cadena y lo retorna para encadenamiento fluido."""
        self._next_handler = handler
        return handler

    @abstractmethod
    def process(self, context: str, **kwargs) -> str:
        """Procesa y filtra el contexto actual."""
        pass

    def handle(self, context: str, **kwargs) -> str:
        """Ejecuta el paso actual y delega al siguiente eslabón si existe."""
        filtered = self.process(context, **kwargs)
        if self._next_handler:
            return self._next_handler.handle(filtered, **kwargs)
        return filtered


class SectionBoundaryFilterHandler(ContextFilterHandler):
    """Filtra secciones irrelevantes (ej. conversation history) preservando ontología y conocimiento."""

    IGNORE_HEADERS = (
        '## conversation history',
        '### relevant past messages',
        'conversation history',
        'relevant past messages',
    )
    PRESERVE_HEADERS = (
        '## relevant knowledge',
        '### user preferences',
    )

    def process(self, context: str, **kwargs) -> str:
        if not context or not context.strip():
            return ""

        filtered_lines: list[str] = []
        in_chat_history_section = False

        for line in context.split('\n'):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            if not line_lower:
                continue

            if any(h in line_lower for h in self.IGNORE_HEADERS):
                in_chat_history_section = True
                continue

            if any(h in line_lower for h in self.PRESERVE_HEADERS):
                in_chat_history_section = False
                continue

            if in_chat_history_section:
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)


class ChatPrefixFilterHandler(ContextFilterHandler):
    """Filtra líneas individuales que corresponden a turnos de chat ya presentes en el historial."""

    CHAT_PREFIXES = (
        'user:', 'assistant:', 'human:', 'ai:',
        'usuario:', 'asistente:', 'q:', 'a:',
        'pregunta:', 'respuesta:',
        '- [user]', '- [assistant]', '- [human]', '- [ai]',
        '- [usuario]', '- [asistente]'
    )

    def process(self, context: str, **kwargs) -> str:
        if not context or not context.strip():
            return ""

        filtered_lines: list[str] = []
        for line in context.split('\n'):
            line_lower = line.strip().lower()
            if not line_lower:
                continue
            if any(line_lower.startswith(prefix) for prefix in self.CHAT_PREFIXES):
                continue
            filtered_lines.append(line)

        return '\n'.join(filtered_lines).strip()


class LengthLimitFilterHandler(ContextFilterHandler):
    """Trunca el texto resultante a una longitud máxima configurada."""

    def process(self, context: str, **kwargs) -> str:
        max_len = kwargs.get("max_len", 3000)
        if len(context) > max_len:
            return context[:max_len] + '\n[... truncado]'
        return context


class ContextSanitizationChain:
    """Ensambla y orquesta la cadena de sanitización de contexto NAMS."""

    def __init__(self):
        self._head: ContextFilterHandler = SectionBoundaryFilterHandler()
        (
            self._head
            .set_next(ChatPrefixFilterHandler())
            .set_next(LengthLimitFilterHandler())
        )

    def sanitize(self, nams_context: str, max_len: int = 3000) -> str:
        if not nams_context or not nams_context.strip():
            return ""
        return self._head.handle(nams_context, max_len=max_len)


_default_sanitization_chain = ContextSanitizationChain()


def sanitize_nams_context(nams_context: str, max_len: int = 3000) -> str:
    """Filtra y trunca el contexto NAMS para evitar que domine el prompt.

    Facade sobre ContextSanitizationChain (Patrón GoF Chain of Responsibility).
    Elimina líneas que parecen historial de chat y trunca a max_len caracteres.

    Este filtro se usa de forma transversal por:
    - beeai_host_manager (sesión activa)
    - beeai_orchestrator_workflow (send_to_agent)
    - multimodal agent (_sanitize_nams_context)
    """
    return _default_sanitization_chain.sanitize(nams_context, max_len=max_len)
