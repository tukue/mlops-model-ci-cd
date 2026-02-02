from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "MLOps API is running", "endpoints": ["/health", "/predict", "/metrics", "/model-info", "/docs"]}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    # Should have either model info or error message
    json_response = response.json()
    assert "active_version" in json_response or "error" in json_response

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

def test_predict_valid():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        # Model not found, skip assertion on prediction value
        assert "No active model found" in response.json()["detail"]
    else:
        assert response.status_code == 200
        json_response = response.json()
        assert "prediction" in json_response
        assert isinstance(json_response["prediction"], int)
        # Model version should be included if model exists
        assert "model_version" in json_response

def test_predict_invalid_schema():
    payload = {"features": [1.0, 2.0]} # Too short
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
