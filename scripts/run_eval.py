"""
CLI tool to execute evaluation benchmarks over Retryv's Golden Dataset.

Usage:
    # Run smoke test on a subset of 3 queries:
    .venv\\Scripts\\python scripts/run_eval.py --strategy fixed_size --limit 3

    # Run full 52-query benchmark for fixed_size strategy:
    .venv\\Scripts\\python scripts/run_eval.py --strategy fixed_size

    # Run benchmark without saving JSON report to disk:
    .venv\\Scripts\\python scripts/run_eval.py --strategy fixed_size --no-save
"""
import sys
import os
import argparse
import logging

# Ensure project root is in sys.path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.eval.runner import EvalRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks over Retryv Golden Dataset."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed_size",
        choices=["fixed_size", "structure_aware", "semantic"],
        help="Chunking strategy to evaluate (default: fixed_size).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (useful for smoke tests).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Sleep seconds between queries to protect rate limits (default: 2.0s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving evaluation report JSON to data/eval/reports/.",
    )
    args = parser.parse_args()

    runner = EvalRunner(inter_query_sleep=args.sleep)
    report = runner.run_evaluation(
        strategy=args.strategy,
        limit=args.limit,
        save_report=not args.no_save,
    )

    print("\n" + "=" * 70)
    print(f"EVALUATION BENCHMARK REPORT — {report.strategy.upper()}")
    print(f"Report ID    : {report.id}")
    print(f"Timestamp    : {report.timestamp}")
    print(f"Total Queries: {report.total_queries}")
    print("=" * 70)

    print("\n--- Aggregate Metrics (Overall Means) ---")
    for metric_name, score in report.aggregate_metrics.items():
        print(f"  {metric_name:<25}: {score:.4f}")

    print("\n--- Category Breakdown ---")
    header = f"  {'Category':<15} {'Count':>5}  " + "  ".join(
        f"{m[:8]:>8}" for m in report.aggregate_metrics.keys()
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cat_name, cat_data in report.category_breakdown.items():
        scores_str = "  ".join(
            f"{cat_data.mean_scores.get(m, 0.0):>8.4f}"
            for m in report.aggregate_metrics.keys()
        )
        print(f"  {cat_name:<15} {cat_data.count:>5}  {scores_str}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
