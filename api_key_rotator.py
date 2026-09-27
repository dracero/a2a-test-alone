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
        exclude_keys: Optional[set[str]] = None,
    ) -> tuple[Optional[str], Optional[int]]:
        """Selecciona una key disponible y retorna (key, index)."""
        pass


class PriorityPaidKeyStrategy(KeySelectionStrategy):
    """Estrategia prioritaria: favorece siempre la primera key (paga).
    
    Solo si la principal entra en cooldown conmuta a las secundarias (gratuitas).
    Si todas están en cooldown, busca la key con menor tiempo restante de penalización.
    """

    def select_key(
        self,
        keys: list[str],
        cooldowns: dict[str, float],
        paid_cooldown: float,
        free_cooldown: float,
        exclude_keys: Optional[set[str]] = None,
    ) -> tuple[Optional[str], Optional[int]]:
        if not keys:
            return None, None

        now = time.time()
        exclude = exclude_keys or set()

        # Pase 1: Buscar key activa que no esté excluida en este ciclo de reintento
        for idx, key in enumerate(keys):
            if key in exclude:
                continue
            cooldown_ts = cooldowns.get(key, 0)
            cooldown_limit = paid_cooldown if idx == 0 else free_cooldown
            if now - cooldown_ts >= cooldown_limit:
                cooldowns.pop(key, None)
                return key, idx

        # Pase 2: Si no hubo no-excluidas, buscar cualquier key fuera de cooldown
        for idx, key in enumerate(keys):
            cooldown_ts = cooldowns.get(key, 0)
            cooldown_limit = paid_cooldown if idx == 0 else free_cooldown
            if now - cooldown_ts >= cooldown_limit:
                cooldowns.pop(key, None)
                return key, idx

        # Pase 3: Si todas están en cooldown, seleccionar la de menor tiempo restante
        best_idx = 0
        min_remaining = float("inf")
        for idx, key in enumerate(keys):
            cooldown_ts = cooldowns.get(key, 0)
            limit = paid_cooldown if idx == 0 else free_cooldown
            remaining = limit - (now - cooldown_ts)
            if remaining < min_remaining:
                min_remaining = remaining
                best_idx = idx

        chosen_key = keys[best_idx]
        cooldowns.pop(chosen_key, None)
        print(f"⚠️ Todas las keys en cooldown. Conmutando a ...{chosen_key[-4:]} ({min_remaining:.1f}s)")
        return chosen_key, best_idx


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
        exclude_keys: Optional[set[str]] = None,
    ) -> tuple[Optional[str], Optional[int]]:
        if not keys:
            return None, None

        exclude = exclude_keys or set()
        with self._lock:
            total = len(keys)
            now = time.time()
            for offset in range(total):
                idx = (self._current_index + offset) % total
                key = keys[idx]
                if key in exclude:
                    continue
                cooldown_ts = cooldowns.get(key, 0)
                cooldown_limit = paid_cooldown if idx == 0 else free_cooldown
                if now - cooldown_ts >= cooldown_limit:
                    cooldowns.pop(key, None)
                    self._current_index = (idx + 1) % total
                    return key, idx

            # Fallback a cualquier key libre
            for offset in range(total):
                idx = (self._current_index + offset) % total
                key = keys[idx]
                cooldown_ts = cooldowns.get(key, 0)
                cooldown_limit = paid_cooldown if idx == 0 else free_cooldown
                if now - cooldown_ts >= cooldown_limit:
                    cooldowns.pop(key, None)
                    self._current_index = (idx + 1) % total
                    return key, idx

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

        fallback = os.getenv(self._fallback_var, "").strip()
        if fallback and fallback not in self._keys:
            self._keys.append(fallback)

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

    COOLDOWN_SECONDS = 60
    PAID_KEY_COOLDOWN_SECONDS = 15
    EXHAUSTED_COOLDOWN_SECONDS = 300  # 5 min para cuotas agotadas (429)
    OVERLOAD_COOLDOWN_SECONDS = 25    # 25s para sobrecarga temporal (503)

    def get_key(self, exclude_keys: Optional[set[str]] = None) -> str:
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
                exclude_keys=exclude_keys,
            )
            if key:
                print(f"🔑 Usando Google API Key ...{key[-4:]}")
                return key
            return ""

    def report_failure(self, key: Any, is_exhausted: bool = False, is_overloaded: bool = False):
        """Marca una key como fallida con cooldown diferenciado."""
        key_str = _extract_str_key(key)
        if not key_str or key_str not in self._keys:
            return
        with self._lock:
            self._cooldowns[key_str] = time.time()
            idx = self._keys.index(key_str)
            if is_exhausted:
                effective_cd = self.EXHAUSTED_COOLDOWN_SECONDS
                reason = "cuota agotada (429)"
            elif is_overloaded:
                effective_cd = self.OVERLOAD_COOLDOWN_SECONDS
                reason = "sobrecarga (503)"
            elif idx == 0:
                effective_cd = self.PAID_KEY_COOLDOWN_SECONDS
                reason = "key prioritaria"
            else:
                effective_cd = self.COOLDOWN_SECONDS
                reason = "falla temporal"
            print(f"🚫 Key ...{key_str[-4:]} en cooldown por {effective_cd}s ({reason})")

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

    if "2.5" in model:
        model = "gemini-3.5-flash"
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


def _rebuild_llm(llm: Any, new_key: str, model_override: Optional[str] = None):
    """Re-crea un ChatGoogleGenerativeAI con una nueva key reutilizando create_google_llm."""
    model = model_override or getattr(llm, "model_name", None) or getattr(llm, "model", "gemini-3.5-flash")
    if "2.5" in model:
        model = "gemini-3.5-flash"
    temperature = getattr(llm, "temperature", 0.3)
    max_output = getattr(llm, "max_output_tokens", 8192)
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


def invoke_with_retry(llm: Any, messages: Any, *, rotator: Optional[GoogleApiKeyRotator] = None, max_retries: int = 0, base_wait: float = 0.2):
    """Invoca LLM con retry automático y conmutación ágil de keys."""
    if rotator is None:
        rotator = google_key_rotator
    if max_retries == 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = _extract_str_key(getattr(llm, "google_api_key", ""))
    tried_keys: set[str] = {current_key} if current_key else set()

    for attempt in range(max_retries + 1):
        try:
            res = llm.invoke(messages)
            return normalize_llm_response(res)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            err_str = str(exc).lower()
            is_overloaded = any(k in err_str for k in ["503", "unavailable", "high demand", "overloaded"])
            is_exhausted = any(k in err_str for k in ["quota", "resource_exhausted", "resourceexhausted", "credits are depleted", "429"])
            print(f"⚠️ [Retry {attempt+1}/{max_retries}] Error ({'503 High Demand' if is_overloaded else ('Cuota 429' if is_exhausted else 'Error')}) con key ...{current_key[-4:] if current_key else '????'}: {str(exc)[:160]}")
            if attempt >= max_retries:
                break
            
            if current_key:
                rotator.report_failure(current_key, is_exhausted=is_exhausted, is_overloaded=is_overloaded)

            new_key = rotator.get_key(exclude_keys=tried_keys)
            is_different_key = bool(new_key and new_key != current_key)
            if new_key:
                tried_keys.add(new_key)
                os.environ["GEMINI_API_KEY"] = new_key
                os.environ["GOOGLE_API_KEY"] = new_key
                try:
                    import litellm
                    litellm.api_key = new_key
                except Exception:
                    pass
            llm = _rebuild_llm(llm, new_key, model_override="gemini-3.5-flash")
            current_key = new_key

            wait = 0.2 if is_different_key else min(1.5 * (1.5 ** attempt), 8.0)
            print(f"⏳ Esperando {wait:.1f}s antes de reintentar...")
            time.sleep(wait)

    raise last_exc


async def ainvoke_with_retry(llm: Any, messages: Any, *, rotator: Optional[GoogleApiKeyRotator] = None, max_retries: int = 0, base_wait: float = 0.2):
    """Versión async de invoke_with_retry con conmutación ágil sobre gemini-3.5-flash."""
    import asyncio
    if rotator is None:
        rotator = google_key_rotator
    if max_retries == 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = _extract_str_key(getattr(llm, "google_api_key", ""))
    tried_keys: set[str] = {current_key} if current_key else set()

    for attempt in range(max_retries + 1):
        try:
            res = await llm.ainvoke(messages)
            return normalize_llm_response(res)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            err_str = str(exc).lower()
            is_overloaded = any(k in err_str for k in ["503", "unavailable", "high demand", "overloaded"])
            is_exhausted = any(k in err_str for k in ["quota", "resource_exhausted", "resourceexhausted", "credits are depleted", "429"])
            print(f"⚠️ [Async Retry {attempt+1}/{max_retries}] Error ({'503 High Demand' if is_overloaded else ('Cuota 429' if is_exhausted else 'Error')}) con key ...{current_key[-4:] if current_key else '????'}: {str(exc)[:160]}")
            if attempt >= max_retries:
                break

            if current_key:
                rotator.report_failure(current_key, is_exhausted=is_exhausted, is_overloaded=is_overloaded)

            new_key = rotator.get_key(exclude_keys=tried_keys)
            is_different_key = bool(new_key and new_key != current_key)
            if new_key:
                tried_keys.add(new_key)
                os.environ["GEMINI_API_KEY"] = new_key
                os.environ["GOOGLE_API_KEY"] = new_key
                try:
                    import litellm
                    litellm.api_key = new_key
                except Exception:
                    pass
            llm = _rebuild_llm(llm, new_key, model_override="gemini-3.5-flash")
            current_key = new_key

            wait = 0.2 if is_different_key else min(1.5 * (1.5 ** attempt), 8.0)
            print(f"⏳ Esperando {wait:.1f}s antes de reintentar...")
            await asyncio.sleep(wait)

    raise last_exc


# ── Utilidades compartidas y rotación NAMS ──────────────────────────

def sync_env_key(rotator: Optional[GoogleApiKeyRotator] = None) -> str:
    """Obtiene la key activa del rotador y sincroniza os.environ y litellm para LiteLLM/SDKs externos.

    Centraliza el patrón repetido:
        active_key = google_key_rotator.get_key()
        os.environ["GEMINI_API_KEY"] = active_key
        os.environ["GOOGLE_API_KEY"] = active_key
        litellm.api_key = active_key

    Returns:
        La key activa (puede ser string vacío si no hay keys).
    """
    if rotator is None:
        rotator = google_key_rotator
    active_key = rotator.get_key()
    if active_key:
        os.environ["GEMINI_API_KEY"] = active_key
        os.environ["GOOGLE_API_KEY"] = active_key
        try:
            import litellm
            litellm.api_key = active_key
        except Exception:
            pass
    return active_key


def rotate_and_sync_env_key(
    failed_key: Optional[str] = None,
    rotator: Optional[GoogleApiKeyRotator] = None,
    is_exhausted: bool = True,
    is_overloaded: bool = False,
    exclude_keys: Optional[set[str]] = None,
) -> str:
    """Reporta falla de una clave (si se especifica), obtiene la siguiente y sincroniza os.environ y litellm.

    Usado por la memoria NAMS y subsistemas que interactúan con LiteLLM / Neo4j Agent Memory.
    """
    if rotator is None:
        rotator = google_key_rotator
    if failed_key:
        rotator.report_failure(failed_key, is_exhausted=is_exhausted, is_overloaded=is_overloaded)
    new_key = rotator.get_key(exclude_keys=exclude_keys)
    if new_key:
        os.environ["GEMINI_API_KEY"] = new_key
        os.environ["GOOGLE_API_KEY"] = new_key
        try:
            import litellm
            litellm.api_key = new_key
        except Exception:
            pass
    return new_key


async def aexecute_with_nams_key_rotation(
    coro_factory: Any,
    rotator: Optional[GoogleApiKeyRotator] = None,
    max_retries: int = 0,
    fallback_coro_factory: Optional[Any] = None,
    action_name: str = "operación NAMS",
) -> Any:
    """Ejecuta una operación asíncrona de NAMS (Neo4j Agent Memory / LiteLLM) rotando API keys automáticamente ante 429/403/quota.

    Si una clave agota su cuota o es rechazada durante el registro en NAMS:
    1. Reporta la clave fallida al rotador para enviarla a cooldown.
    2. Obtiene la siguiente clave disponible y actualiza os.environ y litellm.api_key.
    3. Reintenta la operación con la nueva clave.
    4. Si se agotan todas las claves disponibles y existe un fallback_coro_factory, lo ejecuta
       (ej: guardar mensaje en Neo4j omitiendo extracción de entidades para no perder la conversación).
    """
    import asyncio
    if rotator is None:
        rotator = google_key_rotator
    if max_retries <= 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = sync_env_key(rotator)
    tried_keys: set[str] = {current_key} if current_key else set()

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                # Si no es error de cuota/rate-limit/auth, propagar directamente
                raise

            err_str = str(exc).lower()
            is_overloaded = any(k in err_str for k in ["503", "unavailable", "high demand", "overloaded"])
            is_exhausted = any(k in err_str for k in ["quota", "resource_exhausted", "resourceexhausted", "credits are depleted", "429"])
            key_suffix = current_key[-4:] if current_key else "????"
            logger.warning(
                f"🔄 [NAMS Key Rotation {attempt+1}/{max_retries}] Error ({'503 High Demand' if is_overloaded else ('Cuota 429' if is_exhausted else 'API Error')}) "
                f"en {action_name} con key ...{key_suffix}: {str(exc)[:160]}"
            )

            if attempt >= max_retries:
                break

            if current_key:
                rotator.report_failure(current_key, is_exhausted=is_exhausted, is_overloaded=is_overloaded)

            new_key = rotator.get_key(exclude_keys=tried_keys)
            is_different = bool(new_key and new_key != current_key)
            if new_key:
                tried_keys.add(new_key)
                os.environ["GEMINI_API_KEY"] = new_key
                os.environ["GOOGLE_API_KEY"] = new_key
                try:
                    import litellm
                    litellm.api_key = new_key
                except Exception:
                    pass
            current_key = new_key

            wait = 0.2 if is_different else min(1.5 * (1.5 ** attempt), 8.0)
            logger.info(f"⏳ [NAMS Key Rotation] Conmutando a key ...{current_key[-4:] if current_key else '????'}. Esperando {wait:.1f}s...")
            await asyncio.sleep(wait)

    if fallback_coro_factory is not None:
        logger.warning(f"⚠️ Todas las claves agotadas para {action_name}. Ejecutando fallback de degradación...")
        return await fallback_coro_factory()

    raise last_exc


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
