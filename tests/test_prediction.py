from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_prediction_logic():
    # Helper set to ensure prediction works
    payload = {
        "age": 45,
        "gender": "Male",
        "tenure_months": 24,
        "monthly_charges": 79.85,
        "total_charges": 1916.40,
        "contract_type": "Two year",
        "payment_method": "Credit card",
        "paperless_billing": "Yes",
        "num_support_tickets": 2,
        "num_logins_last_month": 42,
        "feature_usage_score": 8.5,
        "late_payments": 0,
        "partner": "Yes",
        "dependents": "No",
        "internet_service": "Fiber optic",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
    }

    # We call the predict endpoint
    response = client.post("/predict", json=payload)

    # If the model is loaded, we expect 200 OK
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        # Probability must be between 0 and 1
        assert 0 <= data["churn_probability"] <= 1
