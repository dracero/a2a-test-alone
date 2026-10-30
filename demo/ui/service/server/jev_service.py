"""
JEV (TypeSafe AI) System One Integration Service.
Provides typed, calibrated decision primitives (Choice, Noul, Score)
to replace unstructured LLM routing and classification prompts.

Uses JEV_API_KEY (or TYPESAFE_API_KEY) from environment.
Live docs reference: https://docs.typesafe.ai
"""

import os
import logging
from typing import Optional, Any
from typesafe_sdk import (
    AsyncTypeSafeClient,
    TypeSafeClient,
    Choice,
    Noul,
    Score,
    TypeSafeError,
)

logger = logging.getLogger(__name__)


def get_jev_api_key() -> Optional[str]:
    """Retrieve JEV / TypeSafe API key and ensure environment consistency."""
    key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if key and not os.getenv("TYPESAFE_API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = key
    return key


def get_async_jev_client() -> AsyncTypeSafeClient:
    """Instantiate and return AsyncTypeSafeClient with configured API key."""
    api_key = get_jev_api_key()
    return AsyncTypeSafeClient(api_key=api_key)


def get_sync_jev_client() -> TypeSafeClient:
    """Instantiate and return synchronous TypeSafeClient with configured API key."""
    api_key = get_jev_api_key()
    return TypeSafeClient(api_key=api_key)


# =====================================================================
# 1. Orchestrator Routing (Agent Selection)
# =====================================================================

async def classify_agent_routing(
    user_message: str,
    available_agents: list[dict[str, Any]],
    history_text: str = "",
    neo4j_context_text: str = "",
    has_images: bool = False,
    image_count: int = 0,
) -> tuple[str, float, dict[str, float]]:
    """
    Use JEV System One model to route the user message to the best specialized agent
    or DIRECT (for greetings, small talk, general capability inquiries).
    
    Returns:
        (chosen_agent, confidence, probabilities)
    """
    api_key = get_jev_api_key()
    if not api_key:
        logger.warning("⚠️ JEV_API_KEY not found. Fallback to first available agent.")
        fallback = available_agents[0]["name"] if available_agents else "DIRECT"
        return fallback, 0.0, {}

    # Build criteria dynamically from available agents
    criteria: dict[str, Optional[str]] = {
        "DIRECT": (
            "The user request is a simple greeting (hello, hi, hola, hey, buenos días, buenas tardes), "
            "small talk, expression of thanks/goodbye, or a general question asking what the assistant can do "
            "or what services/agents are available, with NO images."
        )
    }

    for agent in available_agents:
        name = agent.get("name", "")
        desc = agent.get("description", "")
        name_lower = name.lower()

        if "médico" in name_lower or "medico" in name_lower or "medical" in name_lower:
            criteria[name] = (
                "Any request, question, or inquiry related to medicine, pathology, histology, biology, "
                "anatomy, tissues, organs, cells, diseases, clinical cases, or requests to search, display, "
                "or retrieve microscopic medical figures or histology micrographs."
            )
        elif "física" in name_lower or "fisica" in name_lower or "physics" in name_lower or "socrático" in name_lower:
            criteria[name] = (
                "Any request, problem, or inquiry related to physics, kinematics, projectile motion, dynamics, "
                "Newton's laws, forces, energy, momentum, gravitation, thermodynamics, mechanics, equations, "
                "or physical science exercises."
            )
        elif "image" in name_lower or "generator" in name_lower or "generador" in name_lower:
            criteria[name] = (
                "Requests explicitly asking to generate, create, draw, paint, or render a generic, artistic, "
                "or creative non-medical synthetic image (e.g., 'draw a cat', 'create an image of a mountain', "
                "'generate a sunset'). Never route medical questions or histology image searches here."
            )
        else:
            criteria[name] = desc or f"Specialized tasks handled by agent '{name}'."

    state: dict[str, Any] = {
        "user_message": user_message,
        "has_attached_images": has_images,
        "image_count": image_count,
    }
    if history_text:
        state["recent_conversation_history"] = history_text[:1200]
    if neo4j_context_text:
        state["user_profile_and_preferences"] = neo4j_context_text[:1000]

    instructions = (
        "Analyze the user's message, context, and attached image information. "
        "Select the specialized agent best suited to handle the request, or 'DIRECT' if the request "
        "is a greeting, small talk, gratitude, or general question about available features."
    )

    try:
        async with get_async_jev_client() as client:
            resp = await client.system_one(
                state=state,
                questions={
                    "agent_routing": Choice(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
            )
            choice_ans = resp.choices["agent_routing"]
            chosen = choice_ans.choice
            conf = choice_ans.confidence
            probs = choice_ans.probabilities

            logger.info(
                f"🎯 [JEV Routing] Choice: '{chosen}' | Conf: {conf:.2f} | Probs: {probs}"
            )
            return chosen, conf, probs
    except TypeSafeError as e:
        logger.error(f"❌ JEV API error during routing: {e}")
        fallback = available_agents[0]["name"] if available_agents else "DIRECT"
        return fallback, 0.0, {}
    except Exception as e:
        logger.error(f"❌ Unexpected error in JEV routing: {e}")
        fallback = available_agents[0]["name"] if available_agents else "DIRECT"
        return fallback, 0.0, {}


# =====================================================================
# 2. Active Session Intent (Continue vs Switch Agent)
# =====================================================================

def decide_session_continuation_sync(active_agent: str, user_message: str) -> tuple[bool, float]:
    """
    Synchronously determine whether the student's message continues the tutoring session
    with active_agent or intends to switch tasks/agents.
    
    Returns:
        (should_continue: bool, confidence: float)
    """
    api_key = get_jev_api_key()
    if not api_key:
        return True, 0.0

    state = {
        "active_agent": active_agent,
        "student_message": user_message,
    }
    instructions = (
        f"Determine whether the student's message is a continuation of the active tutoring session with '{active_agent}' "
        f"or if they want to switch to a completely different task, agent, or service."
    )
    criteria = {
        "CONTINUAR": (
            "The student is answering a question, asking for clarification, expressing doubt, discussing the current subject, "
            "or asking for direct explanations / answers within the current session."
        ),
        "CAMBIAR": (
            "The student explicitly wants to do something completely different, such as generating an image, "
            "switching to another agent, changing the topic away from the active subject, or leaving the session."
        ),
    }

    try:
        with get_sync_jev_client() as client:
            resp = client.system_one(
                state=state,
                questions={
                    "session_action": Choice(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
            )
            ans = resp.choices["session_action"]
            should_continue = (ans.choice == "CONTINUAR")
            logger.info(
                f"🧠 [JEV Session Intent] '{user_message[:50]}' -> {ans.choice} (conf: {ans.confidence:.2f})"
            )
            return should_continue, ans.confidence
    except Exception as e:
        logger.warning(f"⚠️ Error in JEV sync session decision: {e}. Defaulting to True.")
        return True, 0.0


async def decide_session_continuation_async(active_agent: str, user_message: str) -> tuple[bool, float]:
    """
    Asynchronously determine whether the student's message continues the tutoring session.
    """
    api_key = get_jev_api_key()
    if not api_key:
        return True, 0.0

    state = {
        "active_agent": active_agent,
        "student_message": user_message,
    }
    instructions = (
        f"Determine whether the student's message is a continuation of the active tutoring session with '{active_agent}' "
        f"or if they want to switch to a completely different task, agent, or service."
    )
    criteria = {
        "CONTINUAR": (
            "The student is answering a question, asking for clarification, expressing doubt, discussing the current subject, "
            "or asking for direct explanations within the session."
        ),
        "CAMBIAR": (
            "The student explicitly wants to do something completely different, switch agents, generate an image, or stop the session."
        ),
    }

    try:
        async with get_async_jev_client() as client:
            resp = await client.system_one(
                state=state,
                questions={
                    "session_action": Choice(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
            )
            ans = resp.choices["session_action"]
            return (ans.choice == "CONTINUAR"), ans.confidence
    except Exception as e:
        logger.warning(f"⚠️ Error in JEV async session decision: {e}")
        return True, 0.0


# =====================================================================
# 3. Knowledge Graph Validation (Hybrid System One: Noul + Choice + Score)
# =====================================================================

UNIVERSAL_COGNITIVE_TAXONOMY: dict[str, str] = {
    "NINGUNO": (
        "The claim is scientifically, medically, or factually correct and consistent with canonical knowledge."
    ),
    "CONFUSION_CONCEPTUAL_O_ESTRUCTURAL": (
        "Confuses, conflates, or swaps two distinct concepts, entities, anatomical structures, physical quantities, or terms."
    ),
    "INVERSION_CAUSA_EFECTO_O_DIRECCIONALIDAD": (
        "Inverts cause and effect, chronological sequence, directionality of an interaction, or physiological/physical process flow."
    ),
    "ATRIBUCION_ERRONEA_DE_PROPIEDADES": (
        "Incorrectly assigns properties, functions, formulas, characteristics, or behaviors of one entity/concept to another."
    ),
    "APLICACION_FUERA_DE_DOMINIO_O_CONDICION": (
        "Applies a valid rule, formula, law, or tissue characteristic outside its domain of validity, boundary conditions, or reference frame."
    ),
    "CONTRADICCION_DE_PRINCIPIO_RECTOR": (
        "Directly contradicts a fundamental governing law, universal conservation principle, physiological homeostasis axiom, or core scientific baseline."
    ),
    "DISCREPANCIA_NOMENCLATURA_O_ESCALA": (
        "Minor discrepancy in terminology, units, scale, or taxonomic classification without fundamental conceptual collapse."
    ),
    "OTRO_ERROR_CONCEPTUAL": (
        "A domain-specific conceptual mistake that does not fit into the standard categories above."
    ),
}

UNIVERSAL_SEVERITY_RUBRIC: list[str] = [
    "1. Correcto o discrepancia trivial / irrelevante: afirmación científicamente válida o con imprecisión despreciable.",
    "2. Error menor: confusión de nomenclatura, signo, unidades o escala que no destruye el concepto de fondo.",
    "3. Error conceptual moderado: extrapolación indebida o aplicación de un concepto en un escenario no válido.",
    "4. Falencia conceptual grave: invierte o viola relaciones, propiedades estructurales o mecanismos esenciales.",
    "5. Falencia crítica/severa: contradice leyes universales de conservación, axiomas científicos primarios o verdades canónicas fundamentales.",
]


async def evaluate_claim_hybrid_system_one(
    student_claim: str,
    context_summary: str,
    domain_name: Optional[str] = None,
    custom_taxonomy: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Use JEV System One (TypeSafe AI) to execute a multi-primitive, typed pedagogical evaluation
    applicable to ANY domain (Physics, Medicine, Biology, Engineering, etc.):
    1. is_correct (Noul): Calibrated probability of scientific truth aligned with canonical KG.
    2. error_type (Choice): Epistemological / cognitive error category (Universal or Domain-specific).
    3. severity (Score): Calibrated ordinal severity rating (1.0 to 5.0).

    Returns:
        {
            "is_correct": bool,
            "probability": float,
            "error_type": str,
            "error_type_confidence": float,
            "severity": float,
            "severity_confidence": float,
        }
    """
    api_key = get_jev_api_key()
    if not api_key:
        return {
            "is_correct": True,
            "probability": 0.5,
            "error_type": "NINGUNO",
            "error_type_confidence": 0.0,
            "severity": 1.0,
            "severity_confidence": 0.0,
        }

    # Resolver criterios de taxonomía (universal + extensiones opcionales de dominio)
    active_taxonomy = dict(UNIVERSAL_COGNITIVE_TAXONOMY)
    if custom_taxonomy:
        active_taxonomy.update(custom_taxonomy)

    domain_label = f" ({domain_name})" if domain_name else ""

    state = {
        "domain": domain_name or "General Science / Academic Discipline",
        "canonical_knowledge": context_summary,
        "student_claim": student_claim,
    }

    questions = {
        "is_correct": Noul(
            instructions=(
                f"Evaluate whether the student's claim is scientifically, medically, or factually correct "
                f"and aligns with the canonical knowledge provided{domain_label}."
            ),
            criteria={
                "true": "The student claim is true, accurate, and scientifically consistent with canonical knowledge.",
                "false": "The student claim is scientifically incorrect, mistaken, or contradicts canonical knowledge.",
            },
        ),
        "error_type": Choice(
            instructions=(
                f"If the student's claim contains any conceptual misunderstanding, diagnostic error, or factual mistake{domain_label}, "
                f"classify the primary category of error. If the claim is correct or valid, select 'NINGUNO'."
            ),
            criteria=active_taxonomy,
        ),
        "severity": Score(
            instructions=(
                f"Rate the conceptual severity and pedagogical gravity of the error in the student's claim on a 1 to 5 scale "
                f"(1: completely correct or trivial mistake, 5: critical violation of fundamental scientific/medical laws)."
            ),
            criteria=UNIVERSAL_SEVERITY_RUBRIC,
        ),
    }

    try:
        async with get_async_jev_client() as client:
            resp = await client.system_one(state=state, questions=questions)
            noul_ans = resp.nouls["is_correct"]
            choice_ans = resp.choices["error_type"]
            score_ans = resp.scores["severity"]

            prob = float(noul_ans.noul)
            is_correct = prob >= 0.5

            error_type = str(choice_ans.choice)
            error_conf = float(choice_ans.confidence)

            severity = float(score_ans.score)
            severity_conf = float(score_ans.confidence)

            # Normalización y consistencia semántica
            if is_correct:
                error_type = "NINGUNO"
                severity = min(severity, 1.5)
            else:
                if error_type == "NINGUNO":
                    error_type = "OTRO_ERROR_CONCEPTUAL"
                severity = max(severity, 2.0)

            logger.info(
                f"🧠 [JEV System One Multi-Primitive] Claim: '{student_claim[:50]}' -> "
                f"is_correct={is_correct} (p={prob:.2f}), error_type={error_type} (conf={error_conf:.2f}), "
                f"severity={severity:.2f} (conf={severity_conf:.2f})"
            )

            return {
                "is_correct": is_correct,
                "probability": prob,
                "error_type": error_type,
                "error_type_confidence": error_conf,
                "severity": severity,
                "severity_confidence": severity_conf,
            }
    except Exception as e:
        logger.warning(f"⚠️ Error in JEV System One multi-primitive evaluation: {e}")
        return {
            "is_correct": True,
            "probability": 0.5,
            "error_type": "NINGUNO",
            "error_type_confidence": 0.0,
            "severity": 1.0,
            "severity_confidence": 0.0,
        }


async def evaluate_claim_against_kg(
    student_claim: str,
    context_summary: str,
) -> tuple[bool, float]:
    """
    Backward-compatible helper returning (is_correct, probability) using evaluate_claim_hybrid_system_one.
    """
    res = await evaluate_claim_hybrid_system_one(student_claim, context_summary)
    return res["is_correct"], res["probability"]


# =====================================================================
# 4. Socratic Mode Intent Classification (SALIR, ENTRAR, CONTINUAR)
# =====================================================================

async def classify_socratic_intent_async(query: str) -> tuple[str, float]:
    """
    Classify student intent regarding Socratic tutoring mode:
    - SALIR: explicitly asks to exit Socratic mode or wants direct answer.
    - ENTRAR: explicitly asks to enter Socratic tutoring mode.
    - CONTINUAR: normal tutoring response, physics question, or doubt.
    """
    api_key = get_jev_api_key()
    if not api_key:
        return "CONTINUAR", 0.0

    state = {"user_query": query}
    instructions = "Analyze the student's intent regarding the Socratic questioning tutoring mode."
    criteria = {
        "SALIR": "The student explicitly asks to stop receiving questions, demands the direct answer, or asks to exit Socratic mode.",
        "ENTRAR": "The student explicitly asks to enter Socratic mode, restart questions, or be guided step-by-step through questions.",
        "CONTINUAR": "Normal response to a physics problem, expressing doubt/confusion, or standard conversational interaction.",
    }

    try:
        async with get_async_jev_client() as client:
            resp = await client.system_one(
                state=state,
                questions={
                    "socratic_intent": Choice(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
            )
            ans = resp.choices["socratic_intent"]
            logger.info(
                f"🧠 [JEV Socratic Intent] '{query[:50]}' -> {ans.choice} (conf: {ans.confidence:.2f})"
            )
            return ans.choice, ans.confidence
    except Exception as e:
        logger.warning(f"⚠️ Error in JEV socratic intent: {e}")
        return "CONTINUAR", 0.0


def classify_socratic_intent_sync(query: str) -> tuple[str, float]:
    """Synchronous version of classify_socratic_intent."""
    api_key = get_jev_api_key()
    if not api_key:
        return "CONTINUAR", 0.0

    state = {"user_query": query}
    instructions = "Analyze the student's intent regarding the Socratic questioning tutoring mode."
    criteria = {
        "SALIR": "The student explicitly asks to stop receiving questions, demands the direct answer, or asks to exit Socratic mode.",
        "ENTRAR": "The student explicitly asks to enter Socratic mode, restart questions, or be guided step-by-step through questions.",
        "CONTINUAR": "Normal response to a physics problem, expressing doubt/confusion, or standard conversational interaction.",
    }

    try:
        with get_sync_jev_client() as client:
            resp = client.system_one(
                state=state,
                questions={
                    "socratic_intent": Choice(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
            )
            ans = resp.choices["socratic_intent"]
            return ans.choice, ans.confidence
    except Exception as e:
        logger.warning(f"⚠️ Error in JEV socratic intent sync: {e}")
        return "CONTINUAR", 0.0


# =====================================================================
# 5. Medical Image Requirement Classification (Noul Decision)
# =====================================================================

async def classify_medical_image_requirement(query: str, ontology_context: str = "") -> tuple[bool, float]:
    """
    Use JEV Noul primitive to determine if the medical query explicitly requests
    to see/display/retrieve an image, figure, or micrograph (vs theoretical questions).
    """
    api_key = get_jev_api_key()
    if not api_key:
        return False, 0.0

    state = {
        "user_query": query,
        "ontology_context": ontology_context[:500] if ontology_context else "",
    }
    instructions = (
        "Determine whether the user is explicitly requesting to view, display, search, or retrieve "
        "a histological image, photo, figure, or micrograph."
    )
    criteria = {
        "true": (
            "The user explicitly asks to see, view, show, display, or find an image, photo, micrograph, "
            "or figure of a tissue, organ, or histological structure (e.g., 'mostrame una foto', 'quiero ver una imagen de')."
        ),
        "false": (
            "The user asks a theoretical, descriptive, conceptual, or explanatory question without requesting "
            "to display an image, or uses words like 'ver' metaphorically (e.g., 'qué se puede ver en la muestra', 'qué es una imagen')."
        ),
    }

    try:
        async with get_async_jev_client() as client:
            resp = await client.system_one(
                state=state,
                questions={
                    "requires_image": Noul(
                        instructions=instructions,
                        criteria=criteria,
                    )
                },
            )
            prob = resp.nouls["requires_image"].noul
            requires = prob >= 0.5
            logger.info(
                f"🔬 [JEV Image Req] '{query[:50]}' -> prob={prob:.2f} (requires_image={requires})"
            )
            return requires, prob
    except Exception as e:
        logger.warning(f"⚠️ Error in JEV medical image requirement: {e}")
        return False, 0.0


# =====================================================================
# 6. Entity Extraction (JEV System One Replacing GliNER)
# =====================================================================

import json

try:
    from neo4j_agent_memory.extraction.base import (
        ExtractedEntity,
        ExtractionResult,
        is_valid_entity_name,
    )
except ImportError:
    from dataclasses import dataclass, field

    def is_valid_entity_name(name: str) -> bool:
        return bool(name and len(name.strip()) >= 2)

    @dataclass
    class ExtractedEntity:
        name: str
        type: str
        confidence: float = 1.0
        extractor: Optional[str] = None
        context: Optional[str] = None
        start_pos: Optional[int] = None
        end_pos: Optional[int] = None

    @dataclass
    class ExtractionResult:
        entities: list[Any] = field(default_factory=list)
        relations: list[Any] = field(default_factory=list)
        preferences: list[Any] = field(default_factory=list)
        source_text: Optional[str] = None


def load_harness_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """Dynamically loads NAMS Harness configuration from JSON file or environment variable."""
    resolved_path = (
        config_path
        or os.getenv("NAMS_HARNESS_CONFIG_PATH")
        or os.path.join(os.path.dirname(__file__), "../../../config/nams_harness_config.json")
    )
    if not os.path.exists(resolved_path):
        root_cand = os.path.join(os.getcwd(), "config/nams_harness_config.json")
        if os.path.exists(root_cand):
            resolved_path = root_cand

    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse harness config at {resolved_path}: {e}")

    return {
        "version": "1.0.0",
        "deficiency_similarity_threshold": 0.82,
        "deficiency_count_threshold": 5,
        "jev_entity_confidence_cutoff": 0.75,
        "jev_misconception_probability_cutoff": 0.50,
        "severity_threshold_for_immediate_candidate": 3.0,
        "canonical_retrieval_k_hops": 1,
        "max_candidate_spans_per_query": 8,
        "default_entity_types": [
            "CONCEPT",
            "PRINCIPLE",
            "PHYSICAL_QUANTITY",
            "FORMULA",
            "TISSUE",
            "CELL",
            "ORGAN",
            "PERSON",
        ],
    }


def extract_candidate_spans_dynamic(
    text: str,
    schema_entity_names: Optional[list[str]] = None,
    max_candidates: int = 8,
) -> list[str]:
    """Dynamically extracts candidate entity spans without hardcoding.
    Uses schema entity names if provided and linguistic token validity."""
    candidates: list[str] = []
    text_lower = text.lower()

    # 1. Match known schema concepts dynamically
    if schema_entity_names:
        for entity_name in schema_entity_names:
            if entity_name.lower() in text_lower and entity_name not in candidates:
                candidates.append(entity_name)

    # 2. Extract 2-gram and 3-gram noun phrases / token groups
    tokens = [t.strip(".,;:!?\"'()[]{}«»") for t in text.split() if t.strip()]
    tokens = [t for t in tokens if t]

    for n in (3, 2):
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i : i + n]
            if is_valid_entity_name(chunk[0]) and is_valid_entity_name(chunk[-1]):
                gram = " ".join(chunk)
                if len(gram) >= 4 and gram not in candidates:
                    candidates.append(gram)

    # 3. Extract 1-gram valid content tokens
    for t in tokens:
        if is_valid_entity_name(t) and len(t) >= 3:
            if not any(t.lower() in c.lower() for c in candidates):
                candidates.append(t)

    return candidates[:max_candidates]


class JEVEntityExtractor:
    """System One Entity Extractor using JEV (TypeSafe AI) to replace GliNER.
    
    Provides typed, calibrated entity extraction with zero GPU VRAM footprint.
    Fully implements the neo4j_agent_memory EntityExtractor Protocol.
    """

    def __init__(
        self,
        domain_name: str = "General",
        schema_config: Optional[Any] = None,
        config_path: Optional[str] = None,
        confidence_cutoff: Optional[float] = None,
    ):
        self.domain_name = domain_name
        self.schema_config = schema_config
        self.harness_config = load_harness_config(config_path)
        self.confidence_cutoff = (
            confidence_cutoff
            if confidence_cutoff is not None
            else self.harness_config.get("jev_entity_confidence_cutoff", 0.75)
        )
        self.max_candidates = self.harness_config.get("max_candidate_spans_per_query", 8)

    def _resolve_entity_types_criteria(
        self, requested_types: Optional[list[str]] = None
    ) -> dict[str, str]:
        """Dynamically build criteria dictionary from schema_config or harness config."""
        criteria: dict[str, str] = {
            "NONE": "The candidate span is not a valid domain concept, entity, or object."
        }

        types_list: list[str] = []
        if requested_types:
            types_list = requested_types
        elif self.schema_config and hasattr(self.schema_config, "entity_types"):
            types_list = [
                et.name if hasattr(et, "name") else str(et)
                for et in self.schema_config.entity_types
            ]
        elif "default_entity_types" in self.harness_config:
            types_list = self.harness_config["default_entity_types"]

        for t in types_list:
            t_upper = t.upper().strip()
            criteria[t_upper] = f"Candidate belongs to ontological class '{t_upper}' in domain '{self.domain_name}'."

        return criteria

    def _get_schema_concept_names(self) -> list[str]:
        """Extract concept names from schema or ontology dynamically if available."""
        names: list[str] = []
        if self.schema_config and hasattr(self.schema_config, "entity_types"):
            for et in self.schema_config.entity_types:
                if hasattr(et, "subtypes") and et.subtypes:
                    names.extend(et.subtypes)
                if hasattr(et, "name") and et.name:
                    names.append(et.name)
        return names

    async def extract(
        self,
        text: str,
        *,
        entity_types: Optional[list[str]] = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
    ) -> Any:
        """Extract entities using JEV System One model."""
        if not text or not text.strip():
            return ExtractionResult(source_text=text)

        schema_concepts = self._get_schema_concept_names()
        candidates = extract_candidate_spans_dynamic(
            text, schema_entity_names=schema_concepts, max_candidates=self.max_candidates
        )

        if not candidates:
            return ExtractionResult(source_text=text)

        api_key = get_jev_api_key()
        if not api_key:
            logger.warning("⚠️ JEV_API_KEY not configured. Falling back to dynamic heuristic extractor.")
            entities = [
                ExtractedEntity(
                    name=cand,
                    type="CONCEPT",
                    confidence=0.5,
                    extractor="DynamicHeuristicFallback",
                )
                for cand in candidates
                if is_valid_entity_name(cand)
            ]
            return ExtractionResult(entities=entities, source_text=text)

        criteria_types = self._resolve_entity_types_criteria(entity_types)

        state = {
            "source_text": text,
            "domain": self.domain_name,
            "candidate_spans": candidates,
        }

        questions: dict[str, Any] = {}
        for idx, cand in enumerate(candidates):
            questions[f"is_valid_{idx}"] = Noul(
                instructions=(
                    f"Determine whether candidate '{cand}' represents an actual domain concept, physical quantity, "
                    f"anatomical structure, or scientific entity in the source text."
                ),
                criteria={
                    "true": "The candidate is a genuine domain concept or entity in this query context.",
                    "false": "Colloquial phrase, common action, non-domain word, or irrelevant span.",
                },
            )
            questions[f"type_{idx}"] = Choice(
                instructions=f"Select the most appropriate entity category for candidate '{cand}'.",
                criteria=criteria_types,
            )

        extracted_entities = []
        try:
            async with get_async_jev_client() as client:
                resp = await client.system_one(state=state, questions=questions)
                for idx, cand in enumerate(candidates):
                    noul_ans = resp.nouls.get(f"is_valid_{idx}")
                    choice_ans = resp.choices.get(f"type_{idx}")

                    prob = float(noul_ans.noul) if noul_ans else 0.0
                    chosen_type = str(choice_ans.choice) if choice_ans else "NONE"

                    if prob >= self.confidence_cutoff and chosen_type != "NONE":
                        start_idx = text.lower().find(cand.lower())
                        end_idx = start_idx + len(cand) if start_idx != -1 else None

                        extracted_entities.append(
                            ExtractedEntity(
                                name=cand,
                                type=chosen_type,
                                confidence=prob,
                                extractor="JEV-SystemOne",
                                start_pos=start_idx if start_idx != -1 else None,
                                end_pos=end_idx,
                            )
                        )

            logger.info(
                f"🏷️ [JEV Entity Extractor ({self.domain_name})] Extracted {len(extracted_entities)} entities "
                f"from '{text[:50]}...' with cutoff >= {self.confidence_cutoff}"
            )
            return ExtractionResult(entities=extracted_entities, source_text=text)
        except Exception as e:
            logger.warning(f"⚠️ Error in JEV entity extraction: {e}. Returning heuristic fallback.")
            entities = [
                ExtractedEntity(
                    name=cand,
                    type="CONCEPT",
                    confidence=0.5,
                    extractor="JEV-ErrorFallback",
                )
                for cand in candidates
                if is_valid_entity_name(cand)
            ]
            return ExtractionResult(entities=entities, source_text=text)


async def extract_entities_jev(
    text: str,
    domain_name: str = "General",
    schema_config: Optional[Any] = None,
    confidence_cutoff: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Convenience functional wrapper returning dict representations of extracted entities."""
    extractor = JEVEntityExtractor(
        domain_name=domain_name,
        schema_config=schema_config,
        confidence_cutoff=confidence_cutoff,
    )
    res = await extractor.extract(text)
    return [
        {
            "name": getattr(e, "name", ""),
            "type": getattr(e, "type", ""),
            "confidence": getattr(e, "confidence", 1.0),
            "extractor": getattr(e, "extractor", "JEV-SystemOne"),
        }
        for e in getattr(res, "entities", [])
    ]

