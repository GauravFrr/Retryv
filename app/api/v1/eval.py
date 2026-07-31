import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from pathlib import Path
import json

from app.eval.runner import EvalRunner
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple in-memory storage for background task states
eval_tasks: Dict[str, Dict[str, Any]] = {}


class EvalRunRequest(BaseModel):
    strategy: str = Field("fixed_size", description="Strategy to evaluate: 'fixed_size', 'structure_aware', 'semantic'")
    limit: Optional[int] = Field(None, description="Max queries to run")
    sleep: float = Field(1.0, description="Sleep seconds between queries")


def run_eval_task(task_id: str, strategy: str, limit: Optional[int], sleep: float):
    eval_tasks[task_id]["status"] = "running"
    try:
        runner = EvalRunner(inter_query_sleep=sleep)
        report = runner.run_evaluation(
            strategy=strategy,
            limit=limit,
            save_report=True
        )
        eval_tasks[task_id]["status"] = "completed"
        eval_tasks[task_id]["report_id"] = report.id
        eval_tasks[task_id]["results"] = {
            "total_queries": report.total_queries,
            "aggregate_metrics": report.aggregate_metrics
        }
    except Exception as e:
        logger.error(f"Background eval task {task_id} failed: {e}")
        eval_tasks[task_id]["status"] = "failed"
        eval_tasks[task_id]["error"] = str(e)


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
def run_eval(request: EvalRunRequest, background_tasks: BackgroundTasks):
    valid_strategies = ("fixed_size", "structure_aware", "semantic")
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid strategy '{request.strategy}'. Must be one of {valid_strategies}"
        )

    task_id = str(uuid.uuid4())
    eval_tasks[task_id] = {
        "task_id": task_id,
        "strategy": request.strategy,
        "status": "queued",
        "error": None
    }

    background_tasks.add_task(
        run_eval_task,
        task_id=task_id,
        strategy=request.strategy,
        limit=request.limit,
        sleep=request.sleep
    )

    return {"task_id": task_id, "status": "queued"}


@router.get("/status/{task_id}")
def get_eval_status(task_id: str):
    if task_id not in eval_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task ID not found."
        )
    return eval_tasks[task_id]


REPORTS_DIR = Path("data/eval/reports")


def find_latest_report(strategy: str) -> Optional[Path]:
    pattern = f"eval_{strategy}_*.json"
    files = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return files[0]


@router.get("/compare")
def compare_eval():
    strategies = ["fixed_size", "structure_aware", "semantic"]
    reports = {}

    for strat in strategies:
        try:
            report_file = find_latest_report(strat)
            if not report_file:
                reports[strat] = {"error": "Report not found"}
                continue
            with open(report_file, "r", encoding="utf-8") as f:
                reports[strat] = json.load(f)
        except Exception as e:
            reports[strat] = {"error": str(e)}

    # Build comparison summary
    comparison = {}
    metrics = ["retrieval_recall", "retrieval_precision", "citation_accuracy", "faithfulness", "correctness"]

    comparison["metrics"] = {}
    for m in metrics:
        comparison["metrics"][m] = {
            "fixed_size": reports.get("fixed_size", {}).get("aggregate_metrics", {}).get(m, None) if "error" not in reports.get("fixed_size", {}) else None,
            "structure_aware": reports.get("structure_aware", {}).get("aggregate_metrics", {}).get(m, None) if "error" not in reports.get("structure_aware", {}) else None,
            "semantic": reports.get("semantic", {}).get("aggregate_metrics", {}).get(m, None) if "error" not in reports.get("semantic", {}) else None
        }

    categories = ["lookup", "multi_hop", "ambiguous", "unanswerable"]
    comparison["categories"] = {}
    for cat in categories:
        comparison["categories"][cat] = {}
        for m in metrics:
            comparison["categories"][cat][m] = {
                "fixed_size": reports.get("fixed_size", {}).get("category_breakdown", {}).get(cat, {}).get("mean_scores", {}).get(m, None) if "error" not in reports.get("fixed_size", {}) else None,
                "structure_aware": reports.get("structure_aware", {}).get("category_breakdown", {}).get(cat, {}).get("mean_scores", {}).get(m, None) if "error" not in reports.get("structure_aware", {}) else None,
                "semantic": reports.get("semantic", {}).get("category_breakdown", {}).get(cat, {}).get("mean_scores", {}).get(m, None) if "error" not in reports.get("semantic", {}) else None
            }

    return {
        "comparison": comparison,
        "reports": {
            s: {
                "id": r.get("id"),
                "timestamp": r.get("timestamp"),
                "total_queries": r.get("total_queries"),
                "aggregate_metrics": r.get("aggregate_metrics")
            } for s, r in reports.items() if "error" not in r
        }
    }
