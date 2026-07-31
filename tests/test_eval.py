from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.eval.models import EvaluationReport

client = TestClient(app)


@patch("app.api.v1.eval.EvalRunner")
def test_run_eval_background_task(mock_runner_class):
    mock_runner = MagicMock()
    mock_runner.run_evaluation.return_value = EvaluationReport(
        id="test_report_id",
        strategy="fixed_size",
        timestamp="2026-07-31T09:00:00Z",
        total_queries=1,
        aggregate_metrics={"faithfulness": 1.0},
        category_breakdown={}
    )
    mock_runner_class.return_value = mock_runner

    response = client.post(
        "/api/v1/eval/run",
        json={
            "strategy": "fixed_size",
            "limit": 1,
            "sleep": 0.0
        }
    )
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"

    # Query status
    task_id = data["task_id"]
    status_response = client.get(f"/api/v1/eval/status/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] in ("queued", "running", "completed")


def test_run_eval_invalid_strategy():
    response = client.post(
        "/api/v1/eval/run",
        json={
            "strategy": "invalid_strategy"
        }
    )
    assert response.status_code == 400


@patch("app.api.v1.eval.find_latest_report")
def test_compare_eval(mock_find_latest):
    # Mock find_latest_report returning None to test fallback
    mock_find_latest.return_value = None
    response = client.get("/api/v1/eval/compare")
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert "reports" in data
