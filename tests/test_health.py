from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify that the /health endpoint returns 200 OK and expected status schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "Retryv"
    assert "version" in data
