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
