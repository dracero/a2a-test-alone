"""
OPA (Open Policy Agent) Governance Service for NAMS.
Provides deterministic, zero-prompt policy enforcement for:
1. Dual-Role authorization (Professor vs. Student KG mutation).
2. Misconception candidate recording permissions.
3. StudentDeficiency promotion governed by calibrated Harness² thresholds.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Ubicación por defecto del directorio de políticas Rego
DEFAULT_POLICIES_DIR = str(Path(__file__).resolve().parents[4] / "policies")
if not os.path.exists(DEFAULT_POLICIES_DIR):
    # Alternativa relativa a demo/ui o raíz
    alt_cand = str(Path(__file__).resolve().parents[2] / "policies")
    if os.path.exists(alt_cand):
        DEFAULT_POLICIES_DIR = alt_cand
    else:
        root_cand = str(Path(os.getcwd()) / "policies")
        if os.path.exists(root_cand):
            DEFAULT_POLICIES_DIR = root_cand


def _find_opa_binary() -> Optional[str]:
    """Locate opa binary in PATH or common local bin paths."""
    common_paths = [
        "/home/dracero/.local/bin/opa",
        os.path.expanduser("~/.local/bin/opa"),
        "/usr/local/bin/opa",
        "/usr/bin/opa",
    ]
    for p in common_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    # Fallback to which
    try:
        res = subprocess.run(["which", "opa"], capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def evaluate_opa_query(
    query: str,
    input_data: dict[str, Any],
    policies_dir: Optional[str] = None,
) -> Any:
    """
    Evaluates a Rego query using OPA CLI with deterministic sub-millisecond execution.
    Falls back to defensive in-memory rules if OPA binary is unavailable.
    """
    p_dir = policies_dir or DEFAULT_POLICIES_DIR
    opa_bin = _find_opa_binary()

    if opa_bin and os.path.exists(p_dir):
        try:
            input_json = json.dumps(input_data)
            cmd = [opa_bin, "eval", "-d", p_dir, "-I", query]
            proc = subprocess.run(
                cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if proc.returncode == 0:
                parsed = json.loads(proc.stdout)
                result = parsed.get("result", [])
                if result and "expressions" in result[0]:
                    val = result[0]["expressions"][0].get("value")
                    logger.debug(f"🛡️ [OPA] Query '{query}' evaluated to: {val}")
                    return val
            else:
                logger.warning(f"⚠️ OPA eval stderr: {proc.stderr}")
        except Exception as e:
            logger.warning(f"⚠️ OPA evaluation subprocess error: {e}")

    # Fallback en memoria equivalente a las reglas de Rego (garantiza cero downtime)
    logger.debug("🛡️ [OPA Fallback] Evaluando regla en memoria")
    if query.endswith("allow_kg_mutation"):
        return (
            input_data.get("mode") == "professor"
            and bool(input_data.get("user_id"))
            and not bool(input_data.get("is_student_session", False))
        )
    elif query.endswith("allow_misconception_candidate"):
        return (
            input_data.get("mode") == "student"
            and bool(input_data.get("student_id"))
            and bool(input_data.get("concept"))
            and float(input_data.get("severity", 1.0)) >= 1.0
        )
    elif query.endswith("promote_to_deficiency"):
        h_cfg = input_data.get("harness_config", {})
        count_thresh = h_cfg.get("deficiency_count_threshold", 5)
        sim_thresh = h_cfg.get("deficiency_similarity_threshold", 0.82)
        occurrences = input_data.get("occurrences", 0)
        similarity = input_data.get("similarity", 0.0)
        severity = input_data.get("severity", 1.0)

        # Regla 1: umbrales estándar
        if occurrences >= count_thresh and similarity >= sim_thresh and severity >= 2.0:
            return True
        # Regla 2: falencias críticas aceleradas
        if severity >= 4.5 and occurrences >= 3 and similarity >= 0.88:
            return True
        return False

    return False


def check_kg_mutation_allowed(
    mode: str,
    user_id: str,
    is_student_session: bool = False,
) -> bool:
    """Verifica si el actor actual tiene permiso para escribir o mutar conceptos canónicos."""
    input_data = {
        "mode": str(mode).lower(),
        "user_id": str(user_id),
        "is_student_session": bool(is_student_session),
    }
    return bool(evaluate_opa_query("data.nams.authz.allow_kg_mutation", input_data))


def check_misconception_candidate_allowed(
    mode: str,
    student_id: str,
    concept: str,
    severity: float = 1.0,
) -> bool:
    """Verifica si la acción califica para registrar un candidato a misconception."""
    input_data = {
        "mode": str(mode).lower(),
        "student_id": str(student_id),
        "concept": str(concept),
        "severity": float(severity),
    }
    return bool(evaluate_opa_query("data.nams.authz.allow_misconception_candidate", input_data))


def check_promote_to_deficiency(
    occurrences: int,
    similarity: float,
    severity: float,
    harness_config: dict[str, Any],
) -> bool:
    """Evalúa deterministamente con OPA si el cluster debe promoverse a StudentDeficiency."""
    input_data = {
        "occurrences": int(occurrences),
        "similarity": float(similarity),
        "severity": float(severity),
        "harness_config": harness_config,
    }
    return bool(evaluate_opa_query("data.nams.pedagogy.promote_to_deficiency", input_data))

