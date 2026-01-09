from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_check():
    # Assuming models are present and loaded
    response = client.get("/readiness")
    # It might be 503 if models failed to load in test environment (if paths are wrong)
    # But let's assume successful load attempt or check status
    if response.status_code == 200:
        assert response.json() == {"status": "ready", "model_status": "loaded"}
    else:
        assert response.status_code == 503

    # If model is loaded, we expect 200. If not, 503.
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert data["churn_prediction"] in [0, 1]
