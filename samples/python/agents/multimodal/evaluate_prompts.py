"""
DSPy Evaluate Prompts for the Physics Multimodal Agent.

This script evaluates and compares the baseline prompts against the
GEPA-optimized prompts loaded from `optimized_prompts.json`.
It provides a side-by-side score breakdown across all examples.

Usage:
    python evaluate_prompts.py --examples training_examples.json
    python evaluate_prompts.py --examples test_set.json
"""

import json
import os
import sys
import argparse
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate
from dotenv import load_dotenv

# Re-use components from optimize script
from optimize_prompts import (
    PhysicsExplainer,
    professor_style_metric,
    load_training_examples,
    DIMENSION_WEIGHTS
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXAMPLES = SCRIPT_DIR / "training_examples.json"
OPTIMIZED_PROMPTS_FILE = SCRIPT_DIR / "optimized_prompts.json"

VERBOSE = False

def log(msg: str):
    if VERBOSE:
        print(msg)


def extract_score_only(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Wrapper around the metric to only return the float score for Evaluate."""
    result = professor_style_metric(example, pred, trace, pred_name, pred_trace)
    if isinstance(result, dict):
        return result.get('score', 0.0)
    return result


def load_optimized_module(module: PhysicsExplainer, filepath: Path) -> PhysicsExplainer:
    """Load optimized settings from JSON into the module."""
    if not filepath.exists():
        print(f"❌ Cannot find {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = data.get("prompts", {})
    
    # helper: update signature and demos
    def update_submod(submod_name, submod_instance):
        info = prompts.get(submod_name)
        if not info:
            return
            
        predict = submod_instance.predict if hasattr(submod_instance, "predict") else submod_instance
        
        # Inject instruction
        if info.get("instruction"):
            if hasattr(predict, "signature"):
                predict.signature = predict.signature.with_instructions(info["instruction"])
        
        # Inject demos
        if info.get("demos"):
            demos = []
            for d in info["demos"]:
                demos.append(dspy.Example(**d).with_inputs(*[k for k in d.keys() if k != 'response' and k != 'socratic_question']))
            predict.demos = demos

    update_submod("direct_response", module.direct)
    update_submod("socratic_question", module.socratic)
    update_submod("post_socratic_response", module.post_socratic)

    return module


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Evaluate GEPA-optimized prompts vs baseline")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES, help="Path to examples JSON")
    parser.add_argument("--optimized", type=Path, default=OPTIMIZED_PROMPTS_FILE, help="Path to optimized_prompts.json")
    parser.add_argument("--model", type=str, default="groq/meta-llama/llama-4-scout-17b-16e-instruct", help="LLM model (must match optimize_prompts)")
    parser.add_argument("--num-threads", type=int, default=4, help="Threads for evaluation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = args.verbose

    print("=" * 60)
    print("⚖️  DSPy GEPA — Prompt Evaluation")
    print("=" * 60)

    # 1. Setup LM
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not set.")
        sys.exit(1)

    lm = dspy.LM(model=args.model, api_key=groq_key, max_tokens=4096)
    dspy.configure(lm=lm)

    # 2. Load dataset
    dataset = load_training_examples(args.examples)
    
    # 3. Setup modules
    print("\n⏳ Initializing Baseline Module...")
    baseline_module = PhysicsExplainer()
    
    print(f"⏳ Loading Optimized Module from {args.optimized.name}...")
    optimized_module = PhysicsExplainer()
    try:
        # First try to load the full DSPy compiled module if it exists
        dspy_ext = args.optimized.with_suffix('.dspy')
        if dspy_ext.exists():
            optimized_module.load(str(dspy_ext))
        else:
            # Fallback to manual JSON update
            optimized_module = load_optimized_module(optimized_module, args.optimized)
    except Exception as e:
        print(f"⚠️ Error loading compiled module, using baseline: {e}")

    # 4. Evaluate Baseline
    print("\n" + "-" * 40)
    print("🏃 RUNNING EVALUATION: BASELINE")
    print("-" * 40)
    evaluator = Evaluate(devset=dataset, metric=extract_score_only, num_threads=args.num_threads, display_progress=True, display_table=0)
    baseline_score = evaluator(baseline_module)

    # 5. Evaluate Optimized
    print("\n" + "-" * 40)
    print("🏃 RUNNING EVALUATION: GEPA OPTIMIZED")
    print("-" * 40)
    optimized_score = evaluator(optimized_module)

    # 6. Results
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS RECAP")
    print("=" * 60)
    print(f"  BASELINE (Default Prompts):   {baseline_score:.2f}%")
    print(f"  OPTIMIZED (GEPA Evolved):     {optimized_score:.2f}%")
    
    diff = optimized_score - baseline_score
    if diff > 0:
        print(f"\n  ✅ GEPA improved the score by +{diff:.2f} percentage points!")
    else:
        print(f"\n  ⚠️ GEPA did not improve the score ({diff:.2f})")

if __name__ == "__main__":
    main()
