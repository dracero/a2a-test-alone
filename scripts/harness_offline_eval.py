#!/usr/bin/env python3
"""
Harness² Offline Optimizer & Evaluator for NAMS Memory and Student Deficiency Detection.
Based on Google Research Harness² (https://github.com/google-research/harness2).

Implements the Recursive Agent Harnessing Propose-Probe-Compose cycle in OFFLINE mode:
1. Propose: Generates candidate adjustments for the agent harness (JEV confidence cutoffs,
   similarity thresholds, count accumulation criteria).
2. Probe: Evaluates each candidate harness variation against a curated golden benchmark
   of student queries and misconceptions without runtime latency overhead.
3. Compose: Consolidates the winning configuration and writes the calibrated parameters
   to config/nams_harness_config.json.

Usage:
    python scripts/harness_offline_eval.py [--apply] [--probe-candidates]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)
sys.path.insert(0, str(root_dir / "demo/ui"))
sys.path.insert(0, str(root_dir))

from service.server.jev_service import (
    load_harness_config,
    evaluate_claim_hybrid_system_one,
    extract_entities_jev,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harness_offline")


class HarnessOfflineOptimizer:
    """Offline evaluator and recursive optimizer based on google-research/harness2."""

    def __init__(
        self,
        benchmark_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.benchmark_path = benchmark_path or str(
            root_dir / "data/harness_benchmarks/student_queries_benchmark.json"
        )
        self.config_path = config_path or str(root_dir / "config/nams_harness_config.json")
        self.benchmark_data = self._load_benchmark()
        self.current_config = load_harness_config(self.config_path)

    def _load_benchmark(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.benchmark_path):
            logger.error(f"❌ Benchmark file not found at {self.benchmark_path}")
            return []
        try:
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading benchmark dataset: {e}")
            return []

    async def probe_single_query(
        self,
        case: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Probes a single student query through the offline evaluation harness."""
        query = case.get("query", "")
        domain = case.get("domain", "General")
        ground_truth_misconception = case.get("is_misconception", False)

        # 1. Probar extracción de entidades JEV
        cutoff = config.get("jev_entity_confidence_cutoff", 0.75)
        extracted_entities = await extract_entities_jev(
            text=query,
            domain_name=domain,
            confidence_cutoff=cutoff,
        )

        # 2. Probar evaluación de juicio epistémico JEV System One
        eval_res = await evaluate_claim_hybrid_system_one(
            student_claim=query,
            context_summary=f"Conocimiento canónico fundamental de {domain}.",
            domain_name=domain,
        )

        prob_correct = eval_res.get("probability", 1.0)
        misconception_cutoff = config.get("jev_misconception_probability_cutoff", 0.50)
        
        # Un estudiante presenta misconception si prob_correct < cutoff
        predicted_misconception = prob_correct < misconception_cutoff
        predicted_error_type = eval_res.get("error_type", "NINGUNO")
        predicted_severity = eval_res.get("severity", 1.0)

        # Matriz de confusión
        is_tp = predicted_misconception and ground_truth_misconception
        is_fp = predicted_misconception and not ground_truth_misconception
        is_tn = not predicted_misconception and not ground_truth_misconception
        is_fn = not predicted_misconception and ground_truth_misconception

        return {
            "id": case.get("id"),
            "query": query,
            "domain": domain,
            "ground_truth_misconception": ground_truth_misconception,
            "predicted_misconception": predicted_misconception,
            "predicted_error_type": predicted_error_type,
            "predicted_severity": predicted_severity,
            "extracted_entities_count": len(extracted_entities),
            "tp": is_tp,
            "fp": is_fp,
            "tn": is_tn,
            "fn": is_fn,
        }

    async def probe_harness_config(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Probe phase: Runs benchmark through the given harness config and computes metrics."""
        results = []
        for case in self.benchmark_data:
            res = await self.probe_single_query(case, config)
            results.append(res)

        tp = sum(1 for r in results if r["tp"])
        fp = sum(1 for r in results if r["fp"])
        tn = sum(1 for r in results if r["tn"])
        fn = sum(1 for r in results if r["fn"])

        total = len(results)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            "config": config,
            "metrics": {
                "total_cases": total,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(accuracy, 4),
            },
            "details": results,
        }

    def propose_candidates(self) -> list[dict[str, Any]]:
        """Propose phase: Generates candidate adjustments for the harness parameters."""
        base = dict(self.current_config)
        candidates = []

        # Variar umbrales de JEV y acumulación semántica
        candidate_cutoffs = [0.70, 0.75, 0.80]
        candidate_misconception_cutoffs = [0.45, 0.50, 0.55]
        candidate_sims = [0.80, 0.82, 0.85]

        for cutoff in candidate_cutoffs:
            for m_cutoff in candidate_misconception_cutoffs:
                for sim in candidate_sims:
                    cand = dict(base)
                    cand["jev_entity_confidence_cutoff"] = cutoff
                    cand["jev_misconception_probability_cutoff"] = m_cutoff
                    cand["deficiency_similarity_threshold"] = sim
                    candidates.append(cand)

        return candidates

    async def run_recursive_harnessing(
        self,
        explore_proposals: bool = False,
        apply_winner: bool = False,
    ) -> dict[str, Any]:
        """Executes the full Propose-Probe-Compose cycle."""
        logger.info("=" * 70)
        logger.info("🔬 GOOGLE RESEARCH HARNESS² - OFFLINE EVALUATION & OPTIMIZER")
        logger.info(f"   Benchmark dataset: {self.benchmark_path} ({len(self.benchmark_data)} casos)")
        logger.info("=" * 70)

        # 1. Probing configuración actual
        logger.info("▶️ [Probe] Evaluando configuración base...")
        baseline_eval = await self.probe_harness_config(self.current_config)
        logger.info(
            f"📊 [Baseline Metrics] F1: {baseline_eval['metrics']['f1_score']} | "
            f"Precision: {baseline_eval['metrics']['precision']} | "
            f"Recall: {baseline_eval['metrics']['recall']} | "
            f"Acc: {baseline_eval['metrics']['accuracy']}"
        )

        best_eval = baseline_eval

        # 2. Si se solicitó exploración recursiva de candidatos (Propose-Probe)
        if explore_proposals:
            candidates = self.propose_candidates()
            logger.info(f"▶️ [Propose] Generadas {len(candidates)} propuestas de harness para sondeo...")

            for i, cand in enumerate(candidates):
                cand_eval = await self.probe_harness_config(cand)
                cand_f1 = cand_eval["metrics"]["f1_score"]
                best_f1 = best_eval["metrics"]["f1_score"]

                if cand_f1 > best_f1:
                    logger.info(
                        f"✨ [Nuevo Mejor Candidato #{i+1}] F1: {cand_f1:.4f} > {best_f1:.4f} "
                        f"(Cutoff: {cand['jev_entity_confidence_cutoff']}, Sim: {cand['deficiency_similarity_threshold']})"
                    )
                    best_eval = cand_eval

        # 3. Compose: Consolidar la configuración ganadora
        winning_config = dict(best_eval["config"])
        winning_metrics = best_eval["metrics"]

        winning_config["offline_harness_meta"] = {
            "framework": "google-research/harness2",
            "mode": "offline_probe_compose",
            "last_calibrated_metrics": winning_metrics,
            "best_f1": winning_metrics["f1_score"],
        }

        logger.info("=" * 70)
        logger.info("🏆 [Compose] Configuración Óptima Consolidada:")
        logger.info(f"   • Similitud Coseno Falencias: {winning_config.get('deficiency_similarity_threshold')}")
        logger.info(f"   • Frecuencia Acumulación: {winning_config.get('deficiency_count_threshold')}")
        logger.info(f"   • Cutoff Entidad JEV: {winning_config.get('jev_entity_confidence_cutoff')}")
        logger.info(f"   • Cutoff Misconception JEV: {winning_config.get('jev_misconception_probability_cutoff')}")
        logger.info(f"   • F1-Score Resultante: {winning_metrics['f1_score']}")
        logger.info("=" * 70)

        if apply_winner:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(winning_config, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 [Persist] Configuración ganadora guardada en {self.config_path}")

        return {
            "baseline": baseline_eval["metrics"],
            "winning": winning_metrics,
            "config": winning_config,
        }


async def main():
    parser = argparse.ArgumentParser(description="Harness² Offline Optimizer para NAMS")
    parser.add_argument("--benchmark", type=str, default=None, help="Ruta al dataset golden")
    parser.add_argument("--config", type=str, default=None, help="Ruta al nams_harness_config.json")
    parser.add_argument("--probe-candidates", action="store_true", help="Explorar candidatos recursivamente")
    parser.add_argument("--apply", action="store_true", help="Guardar la configuración ganadora")
    args = parser.parse_args()

    optimizer = HarnessOfflineOptimizer(
        benchmark_path=args.benchmark,
        config_path=args.config,
    )
    await optimizer.run_recursive_harnessing(
        explore_proposals=args.probe_candidates,
        apply_winner=args.apply,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
