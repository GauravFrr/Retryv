"""
Validation script for the ConfidenceGuard threshold on the FastAPI docs corpus.

Reads the labeled query set from data/eval/golden_dataset.json — the single
source of truth shared with Phase 4 Unit 4.1 (full RAG evaluation).  Only
the 'retrieval_sufficient' field is used here; Phase 4 will additionally consume
'expected_answer', 'relevant_sources', and 'answer_quality_notes'.

Usage (from the project root):
    .venv\\Scripts\\python scripts/validate_confidence_threshold.py

    # Sweep a custom threshold:
    .venv\\Scripts\\python scripts/validate_confidence_threshold.py --threshold 0.030

Output: a per-query result table + aggregate precision/recall/F1/accuracy at the
current (or overridden) threshold, providing documented evidence for the chosen
threshold value.
"""
import sys
import os
import json
import argparse

# Ensure the project root is on the path when run from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.generation.confidence_guard import ConfidenceGuard
from app.retrieval.fusion import HybridRetriever

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval", "golden_dataset.json",
)


def load_eval_queries(path: str = DATASET_PATH) -> list[tuple[str, bool]]:
    """Load (query, retrieval_sufficient) pairs from the golden dataset.

    Only entries with a non-null 'retrieval_sufficient' field are included.
    Phase 4 fields (expected_answer, relevant_sources, etc.) are silently ignored here.

    Returns:
        List of (query_text, expected_sufficient_bool) tuples.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for entry in data["queries"]:
        if entry.get("retrieval_sufficient") is None:
            continue  # skip partially-labelled entries
        queries.append((entry["query"], bool(entry["retrieval_sufficient"])))

    return queries


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(threshold: float = None, dataset_path: str = DATASET_PATH) -> None:
    effective_threshold = threshold if threshold is not None \
        else settings.RETRIEVAL_CONFIDENCE_THRESHOLD

    eval_queries = load_eval_queries(dataset_path)

    print(f"\n{'=' * 74}")
    print(f"ConfidenceGuard Threshold Validation")
    print(f"Dataset  : {dataset_path}")
    print(f"Queries  : {len(eval_queries)}")
    print(f"Retrieval: HybridRetriever (Dense+Sparse RRF, rrf_k=60)")
    print(f"Threshold: {effective_threshold}")
    print(f"{'=' * 74}\n")

    guard = ConfidenceGuard(threshold=effective_threshold)
    retriever = HybridRetriever()

    results = []
    header = (
        f"{'#':>2}  {'Label':>5}  {'Pred':>5}  {'Score':>7}  "
        f"{'Query':<52}  {'OK'}"
    )
    print(header)
    print("-" * len(header))

    for i, (query, expected_sufficient) in enumerate(eval_queries, start=1):
        try:
            search_results = retriever.retrieve(query, strategy="fixed_size", top_k=5)
            top_score = search_results[0].score if search_results else 0.0
            predicted_sufficient = guard.is_retrieval_sufficient(search_results)
        except Exception as e:
            print(f"  ERROR on query [{query!r}]: {e}")
            continue

        correct = predicted_sufficient == expected_sufficient
        label_str = "ANSW" if expected_sufficient else "UNANSW"
        pred_str  = "PASS"  if predicted_sufficient  else "BLOCK"
        outcome   = "[OK]" if correct               else "[FAIL]"

        print(
            f"{i:>2}  {label_str:>6}  {pred_str:>5}  {top_score:>7.4f}  "
            f"{query[:52]:<52}  {outcome}"
        )
        results.append((expected_sufficient, predicted_sufficient))

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    if not results:
        print("No results — check dataset path and retriever state.")
        return

    tp = sum(1 for e, p in results if e and p)       # answerable, passed
    fp = sum(1 for e, p in results if not e and p)   # unanswerable, passed (bad)
    tn = sum(1 for e, p in results if not e and not p)# unanswerable, blocked (good)
    fn = sum(1 for e, p in results if e and not p)   # answerable, blocked (bad)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy  = (tp + tn) / len(results)

    print(f"\n{'-' * 50}")
    print(f"Threshold : {effective_threshold}")
    print(f"Total     : {len(results)}  (TP={tp}  FP={fp}  TN={tn}  FN={fn})")
    print(f"Precision : {precision:.3f}  (of PASS decisions, how many were correctly answerable)")
    print(f"Recall    : {recall:.3f}  (of answerable queries, how many were passed through)")
    print(f"F1        : {f1:.3f}")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"{'-' * 50}\n")

    if fn > 0:
        print(f"[!] {fn} FN(s): answerable queries incorrectly blocked — consider lowering threshold.")
    if fp > 0:
        print(f"[!] {fp} FP(s): unanswerable queries passed the guard — consider raising threshold.")
    if fn == 0 and fp == 0:
        print("[OK] Perfect precision and recall at this threshold on the current dataset.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate ConfidenceGuard threshold against data/eval/golden_dataset.json."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override threshold (default: settings.RETRIEVAL_CONFIDENCE_THRESHOLD=0.025). "
            "Useful for sweeping values: --threshold 0.020, --threshold 0.030, etc."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help="Path to an alternative golden dataset JSON file.",
    )
    args = parser.parse_args()
    run_evaluation(threshold=args.threshold, dataset_path=args.dataset)
