"""
DSPy GEPA Prompt Optimization for the Physics Multimodal Agent.

This standalone script uses DSPy's GEPA (Genetic-Evolutionary Prompt Architecture)
optimizer to evolve the agent's prompts so they better mimic a target professor's
teaching style.

GEPA = Reflective Prompt Evolution
  - Uses an LLM to REFLECT on program traces (what went well, what didn't)
  - Proposes improved instructions based on textual feedback (not just scalar scores)
  - Builds a tree of evolved prompt candidates, accumulating improvements
  - Uses Pareto-optimal selection across training examples
  - Much more sample-efficient than bayesian search (MIPROv2)

Ref: https://dspy.ai/tutorials/gepa_ai_program/
     https://arxiv.org/abs/2507.19457

Usage:
    python optimize_prompts.py                    # Run with auto='light'
    python optimize_prompts.py --budget medium    # More exploration
    python optimize_prompts.py --dry-run          # Preview without running
    python optimize_prompts.py --verbose          # Extra logging
"""

import json
import os
import re
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import dspy
from dotenv import load_dotenv

# ─────────────────────── Config ───────────────────────

# Load .env from project root
env_path = Path(__file__).resolve().parents[3] / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try cwd
    load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXAMPLES = SCRIPT_DIR / "training_examples.json"
OUTPUT_FILE = SCRIPT_DIR / "optimized_prompts.json"

VERBOSE = False  # Set via --verbose flag


def log(msg: str, always: bool = False):
    """Print if verbose or always."""
    if VERBOSE or always:
        print(msg)


# ─────────────────────── DSPy Signatures ───────────────────────


class PhysicsDirectResponse(dspy.Signature):
    """Generate a comprehensive physics response in the style of a UBA
    Physics I professor.  Use LaTeX ($..$ / $$..$$) for ALL formulas."""

    question: str = dspy.InputField(desc="Student question")
    context: str = dspy.InputField(desc="Relevant context from documents / temario")
    response: str = dspy.OutputField(
        desc="Professor-style response with theory, examples, and LaTeX formulas"
    )


class PhysicsSocraticQuestion(dspy.Signature):
    """Actúa como un profesor de física universitario hablando directamente con el estudiante.
    Haz UNA pregunta socrática para guiar al estudiante hacia la respuesta correcta.
    NO uses introducciones como 'Aquí tienes la pregunta' o 'Pregunta 1:'.
    Dirígete al estudiante directamente (en segunda persona)."""

    question: str = dspy.InputField(desc="Student's original question")
    question_number: int = dspy.InputField(desc="Which question (1-3)")
    previous_answers: str = dspy.InputField(
        desc="Previous student answers (may be empty)"
    )
    socratic_question: str = dspy.OutputField(
        desc="A guiding Socratic question for the student"
    )


class PhysicsPostSocratic(dspy.Signature):
    """Actúa como un profesor de física universitario hablando directamente con el estudiante.
    Después de 3 preguntas socráticas, da la explicación completa reconociendo
    el proceso de razonamiento del estudiante. Dirígete a él de forma directa y amigable."""

    question: str = dspy.InputField(desc="Original question")
    student_answers: str = dspy.InputField(
        desc="Student's answers to the 3 Socratic questions"
    )
    context: str = dspy.InputField(desc="Relevant context from documents")
    response: str = dspy.OutputField(
        desc="Full professor-style response with reflection on student answers, "
        "theory, LaTeX equations, and examples"
    )


# ─────────────────────── DSPy Module ───────────────────────


class PhysicsExplainer(dspy.Module):
    """Combined module wrapping the three prompting tasks."""

    def __init__(self):
        super().__init__()
        self.direct = dspy.ChainOfThought(PhysicsDirectResponse)
        self.socratic = dspy.ChainOfThought(PhysicsSocraticQuestion)
        self.post_socratic = dspy.ChainOfThought(PhysicsPostSocratic)

    def forward(self, question: str, context: str = "", mode: str = "direct"):
        if mode == "direct":
            return self.direct(question=question, context=context)
        elif mode == "socratic":
            return self.socratic(
                question=question,
                question_number=1,
                previous_answers="",
            )
        elif mode == "post_socratic":
            return self.post_socratic(
                question=question,
                student_answers="(simulated answers)",
                context=context,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ─────────────────────── GEPA Feedback Metric ───────────────────────

# Dimension weights
DIMENSION_WEIGHTS = {
    "structure": 1.0,
    "rigor": 1.2,       # Slightly prioritize math rigor
    "accessibility": 1.0,
    "depth": 1.0,
    "tone": 0.8,
}


def professor_style_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible feedback metric.

    Called with 5 arguments by GEPA:
      - gold: The gold Example with .response
      - pred: The Prediction from the module
      - trace: Optional full program trace
      - pred_name: Name of the predictor being optimized (e.g. 'direct')
      - pred_trace: Sub-trace for the specific predictor

    Returns either a float score, or a dict with {'score': float, 'feedback': str}
    so GEPA can use the textual feedback for reflective prompt evolution.
    """
    # Get the candidate response text
    candidate_response = getattr(pred, 'response', None) or getattr(pred, 'socratic_question', '')
    reference_response = getattr(gold, 'response', '')

    if not candidate_response:
        return dict(score=0.0, feedback="No response generated by the candidate.")

    judge_prompt = f"""You are an expert evaluator of physics teaching quality.

Compare the CANDIDATE response with the REFERENCE professor response and rate
the candidate on each dimension (0-10):

1. **Pedagogical Structure**: Clear progression, numbered points, logical flow
2. **Mathematical Rigor**: Correct LaTeX usage, proper notation (\\vec, \\frac, etc.)
3. **Accessibility**: Relatable analogies, intuition before formulas
4. **Depth**: Goes beyond surface-level, connects to broader physics
5. **Socratic Tone**: Engages the student, anticipates misconceptions

REFERENCE (professor):
{reference_response}

CANDIDATE:
{candidate_response}

Return ONLY a JSON object: {{"structure": X, "rigor": X, "accessibility": X, "depth": X, "tone": X}}
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            judge_lm = dspy.settings.lm
            result = judge_lm(judge_prompt)
            result_text = result[0] if isinstance(result, list) else str(result)

            # Extract JSON from possible markdown/text wrapping
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                log(f"  ⚠️ Attempt {attempt+1}: Could not parse judge JSON")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return 0.5

            # Weighted average → normalized to [0, 1]
            weighted_sum = sum(
                scores.get(dim, 5) * weight
                for dim, weight in DIMENSION_WEIGHTS.items()
            )
            total_weight = sum(DIMENSION_WEIGHTS.values())
            avg = weighted_sum / total_weight
            normalized_score = avg / 10.0

            log(f"  📊 Scores: {scores} → {normalized_score:.3f}")

            # Build textual feedback for GEPA reflection
            feedback_parts = []
            for dim, weight in DIMENSION_WEIGHTS.items():
                s = scores.get(dim, 5)
                if s < 4:
                    feedback_parts.append(f"WEAK in {dim} ({s}/10): Needs significant improvement.")
                elif s < 7:
                    feedback_parts.append(f"MODERATE in {dim} ({s}/10): Room for improvement.")
                else:
                    feedback_parts.append(f"STRONG in {dim} ({s}/10): Good performance.")

            feedback_text = (
                f"Overall score: {normalized_score:.2f}. "
                f"Dimension breakdown: {'; '.join(feedback_parts)}. "
            )

            # Add predictor-specific feedback if available
            if pred_name:
                feedback_text += f" (Evaluating predictor: '{pred_name}')"

            if pred_name is not None or pred_trace is not None:
                return dict(score=normalized_score, feedback=feedback_text)
            return normalized_score

        except json.JSONDecodeError as e:
            log(f"  ⚠️ Attempt {attempt+1}: JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return dict(score=0.5, feedback=f"Could not parse evaluation JSON: {e}") if (pred_name is not None or pred_trace is not None) else 0.5

        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str:
                wait = 2 ** (attempt + 1)
                log(f"  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
                if attempt < max_retries - 1:
                    continue
            log(f"  ⚠️ Metric evaluation error: {e}")
            return dict(score=0.5, feedback=f"Evaluation error: {e}") if (pred_name is not None or pred_trace is not None) else 0.5

    return dict(score=0.5, feedback="Metric evaluation failed after retries.") if (pred_name is not None or pred_trace is not None) else 0.5


# ─────────────────────── Helpers ───────────────────────


def load_training_examples(path: Path) -> list[dspy.Example]:
    """Load training examples from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        ex = dspy.Example(
            question=item["question"],
            context=item.get("context", ""),
            response=item["professor_response"],
        ).with_inputs("question", "context")
        examples.append(ex)

    print(f"📚 Loaded {len(examples)} training examples from {path.name}")
    return examples


def extract_optimized_prompts(optimized_module: PhysicsExplainer) -> dict:
    """Extract the optimized prompt strings from the compiled module."""
    prompts = {}

    for name, submod in [
        ("direct_response", optimized_module.direct),
        ("socratic_question", optimized_module.socratic),
        ("post_socratic_response", optimized_module.post_socratic),
    ]:
        info = {"instruction": "", "demos": []}

        # DSPy stores the optimized instruction in the predict's signature
        if hasattr(submod, "predict"):
            predict = submod.predict
        elif hasattr(submod, "module"):
            predict = submod.module
        else:
            predict = submod

        # Try to get the extended signature with optimized instructions
        if hasattr(predict, "extended_signature"):
            sig = predict.extended_signature
            if hasattr(sig, "instructions"):
                info["instruction"] = sig.instructions
        elif hasattr(predict, "signature"):
            sig = predict.signature
            if hasattr(sig, "instructions"):
                info["instruction"] = sig.instructions

        # Get the few-shot demos if any
        if hasattr(predict, "demos"):
            for demo in predict.demos:
                demo_dict = {}
                for key in demo.keys():
                    demo_dict[key] = str(getattr(demo, key, ""))
                info["demos"].append(demo_dict)

        prompts[name] = info

    return prompts


def print_detailed_scores(example, pred, label: str = ""):
    """Print a detailed score breakdown for a single example."""
    result = professor_style_metric(example, pred)

    if isinstance(result, dict):
        score = result.get("score", 0)
        feedback = result.get("feedback", "")
        print(f"\n  {'📊 ' + label + ' — ' if label else '📊 '}Score: {score:.3f}")
        if feedback:
            print(f"     Feedback: {feedback[:200]}")
        return score
    else:
        print(f"\n  {'📊 ' + label + ' — ' if label else '📊 '}Score: {result:.3f}")
        return result


# ─────────────────────── Main ───────────────────────


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="Optimize multimodal agent prompts with DSPy GEPA"
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=DEFAULT_EXAMPLES,
        help="Path to training examples JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output path for optimized prompts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model identifier (e.g. groq/llama-4-scout-17b-16e-instruct)",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=None,
        help="LLM model for GEPA reflection (defaults to same as --model). "
             "A strong model is recommended for reflection.",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["light", "medium", "heavy"],
        default="light",
        help="GEPA budget: light (~few evals), medium, heavy (most exploration)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Override budget with exact max metric calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just test the module without optimizing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=1.0,
        help="Reflection LM temperature (default: 1.0, higher = more creative proposals)",
    )
    parser.add_argument(
        "--track-stats",
        action="store_true",
        default=True,
        help="Track detailed GEPA statistics (default: True)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for GEPA optimization logs",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose

    print("=" * 60)
    print("🧬 DSPy GEPA — Reflective Prompt Evolution")
    print("=" * 60)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Configure LLM ──
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not set. Please set it in .env or environment.")
        sys.exit(1)

    model_name = args.model or "meta-llama/llama-4-scout-17b-16e-instruct"
    print(f"   Model: {model_name}")
    print(f"   Temperature: {args.temperature}")

    lm = dspy.LM(
        model=model_name,
        api_key=groq_key,
        temperature=args.temperature,
        max_tokens=4096,
    )
    dspy.configure(lm=lm)

    # Configure reflection LM (GEPA needs this to reflect on traces)
    reflection_model_name = args.reflection_model or model_name
    print(f"   Reflection Model: {reflection_model_name}")
    print(f"   Reflection Temperature: {args.reflection_temperature}")

    reflection_lm = dspy.LM(
        model=reflection_model_name,
        api_key=groq_key,
        temperature=args.reflection_temperature,
        max_tokens=8192,  # Reflection needs more tokens for analysis
    )

    # ── Load examples ──
    if not args.examples.exists():
        print(f"❌ Examples file not found: {args.examples}")
        sys.exit(1)

    trainset = load_training_examples(args.examples)

    # ── Build module ──
    module = PhysicsExplainer()

    # ── Dry run ──
    if args.dry_run:
        print("\n🧪 DRY RUN — Testing module with first example...")
        ex = trainset[0]
        print(f"\n📝 Question: {ex.question}")
        print(f"📚 Context:  {ex.context}")

        print("\n⏳ Generating response...")
        result = module(question=ex.question, context=ex.context, mode="direct")
        print(f"\n🎓 Response preview:\n{result.response[:600]}...")

        score = print_detailed_scores(ex, result, "Baseline")

        print("\n✅ Dry run complete. Remove --dry-run to optimize.")
        return

    # ── Optimize with GEPA ──
    budget_label = args.budget if not args.max_metric_calls else f"{args.max_metric_calls} calls"
    print(f"\n🚀 Starting GEPA optimization...")
    print(f"   Budget: {budget_label}")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Track stats: {args.track_stats}")
    if args.log_dir:
        print(f"   Log directory: {args.log_dir}")

    print("\n   GEPA works by:")
    print("   1. Running the program on training examples")
    print("   2. Using the reflection LM to analyze what went well/poorly")
    print("   3. Proposing improved instructions based on textual feedback")
    print("   4. Building a tree of evolved candidates (Pareto-optimal selection)")
    print("   5. Optionally merging the best ideas from different branches")

    start_time = time.time()

    # Build GEPA optimizer
    gepa_kwargs = {}
    if args.max_metric_calls:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["auto"] = args.budget

    optimizer = dspy.GEPA(
        metric=professor_style_metric,
        reflection_lm=reflection_lm,
        track_stats=args.track_stats,
        log_dir=args.log_dir,
        # Pareto selection finds prompts that work well across diverse examples
        candidate_selection_strategy="pareto",
        # Merge combines the best ideas from different prompt evolution branches
        use_merge=True,
        **gepa_kwargs,
    )

    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️ Optimization completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ── Show GEPA stats if available ──
    if args.track_stats and hasattr(optimized_module, 'detailed_results'):
        results = optimized_module.detailed_results
        print(f"\n📈 GEPA Optimization Statistics:")
        print(f"   Total candidates explored: {len(results.val_aggregate_scores)}")
        print(f"   Best candidate index: {results.best_idx}")
        print(f"   Best aggregate score: {results.val_aggregate_scores[results.best_idx]:.3f}")
        if results.total_metric_calls:
            print(f"   Total metric calls: {results.total_metric_calls}")
        if results.num_full_val_evals:
            print(f"   Full validation evaluations: {results.num_full_val_evals}")

        # Show score progression
        print(f"\n   Score progression (all candidates):")
        for i, score in enumerate(results.val_aggregate_scores):
            marker = " ← BEST" if i == results.best_idx else ""
            print(f"     Candidate {i}: {score:.3f}{marker}")

    # ── Extract and save ──
    optimized_prompts = extract_optimized_prompts(optimized_module)

    # Add metadata
    output = {
        "metadata": {
            "optimizer": "dspy.GEPA",
            "model": model_name,
            "reflection_model": reflection_model_name,
            "num_training_examples": len(trainset),
            "budget": args.budget if not args.max_metric_calls else None,
            "max_metric_calls": args.max_metric_calls,
            "temperature": args.temperature,
            "reflection_temperature": args.reflection_temperature,
            "optimization_time_seconds": round(elapsed, 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "prompts": optimized_prompts,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Optimized prompts saved to: {args.output}")
    print(f"   Keys: {list(optimized_prompts.keys())}")

    # Also save the compiled DSPy module
    module_path = args.output.with_name(args.output.stem + "_module.json")
    optimized_module.save(str(module_path))
    print(f"   DSPy module saved to: {module_path}")

    # ── Show a comparison ──
    print("\n" + "=" * 60)
    print("📊 Before vs After — First example:")
    print("=" * 60)
    ex = trainset[0]

    result_before = module(question=ex.question, context=ex.context, mode="direct")
    score_before = print_detailed_scores(ex, result_before, "BEFORE (baseline)")

    result_after = optimized_module(
        question=ex.question, context=ex.context, mode="direct"
    )
    score_after = print_detailed_scores(ex, result_after, "AFTER (GEPA-optimized)")

    # Handle the case where scores may be dicts or floats
    s_before = score_before if isinstance(score_before, (int, float)) else score_before.get('score', 0) if isinstance(score_before, dict) else 0
    s_after = score_after if isinstance(score_after, (int, float)) else score_after.get('score', 0) if isinstance(score_after, dict) else 0

    print(f"\n  🔄 Overall: {s_before:.3f} → {s_after:.3f} ({s_after - s_before:+.3f})")

    # Show the optimized instructions
    print("\n" + "=" * 60)
    print("📝 Optimized Instructions (what GEPA evolved):")
    print("=" * 60)
    for task_name, task_info in optimized_prompts.items():
        instruction = task_info.get("instruction", "")
        if instruction:
            print(f"\n  🔹 {task_name}:")
            # Show first 300 chars of instruction
            preview = instruction[:300] + "..." if len(instruction) > 300 else instruction
            print(f"     {preview}")

    print("\n" + "=" * 60)
    print(f"✅ DONE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
