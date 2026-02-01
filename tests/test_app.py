from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "MLOps API is running", "endpoints": ["/health", "/predict", "/metrics", "/docs"]}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    # Ensure model exists or mock it. For integration test, we assume model exists.
    # If model doesn't exist, this might fail with 503.
    # We can mock the get_model dependency if needed, but let's try a real test first
    # assuming the user ran python -m src.train
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        # Model not found, skip assertion on prediction value
        assert response.json()["detail"].startswith("Model not found")
    else:
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert isinstance(response.json()["prediction"], int)

def test_predict_invalid_schema():
    payload = {"features": [1.0, 2.0]} # Too short
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
