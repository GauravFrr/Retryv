"""
Evaluation Benchmark Comparison Script for Retryv (Phase 4 Unit 4.4).

Loads and compares evaluation reports across fixed_size, structure_aware,
and semantic chunking strategies.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/eval/reports")


def find_latest_report(strategy: str) -> Path:
    """Finds the most recent evaluation report JSON for a given strategy."""
    pattern = f"eval_{strategy}_*.json"
    files = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No evaluation report found for strategy '{strategy}' in {REPORTS_DIR}")
    return files[0]


def main():
    strategies = ["fixed_size", "structure_aware", "semantic"]
    reports = {}

    for strat in strategies:
        report_file = find_latest_report(strat)
        with open(report_file, "r", encoding="utf-8") as f:
            reports[strat] = json.load(f)

    metrics = ["retrieval_recall", "retrieval_precision", "citation_accuracy", "faithfulness", "correctness"]

    print("=" * 80)
    print(f"{'CHUNK STRATEGY COMPARISON BENCHMARK REPORT (52 QUERIES)':^80}")
    print("=" * 80)

    print("\n--- Aggregate Metrics Comparison (Overall Means) ---")
    header = f"{'Metric':<22} | {'Fixed-Size':<12} | {'Structure-Aware':<15} | {'Semantic':<12}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        v_fixed = reports["fixed_size"]["aggregate_metrics"].get(m, 0.0)
        v_struct = reports["structure_aware"]["aggregate_metrics"].get(m, 0.0)
        v_sem = reports["semantic"]["aggregate_metrics"].get(m, 0.0)
        print(f"{m:<22} | {v_fixed:<12.4f} | {v_struct:<15.4f} | {v_sem:<12.4f}")

    categories = ["lookup", "multi_hop", "ambiguous", "unanswerable"]
    for cat in categories:
        print(f"\n--- Category Breakdown: {cat.upper()} ---")
        print(header)
        print("-" * len(header))
        for m in metrics:
            v_fixed = reports["fixed_size"]["category_breakdown"][cat]["mean_scores"].get(m, 0.0)
            v_struct = reports["structure_aware"]["category_breakdown"][cat]["mean_scores"].get(m, 0.0)
            v_sem = reports["semantic"]["category_breakdown"][cat]["mean_scores"].get(m, 0.0)
            print(f"{m:<22} | {v_fixed:<12.4f} | {v_struct:<15.4f} | {v_sem:<12.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
