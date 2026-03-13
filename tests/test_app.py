from fastapi.testclient import TestClient
from app.main import app
import shutil
from pathlib import Path

# This function will run before each test in this file
def setup_function():
    # Clear cached model artifacts to prevent file locking issues on Windows
    model_path = Path(__file__).parent.parent / "artifacts" / "Qwen2.5-0.5B-Instruct"
    if model_path.exists():
        print(f"Clearing model cache at: {model_path}")
        shutil.rmtree(model_path)

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    expected_endpoints = ["/health", "/predict", "/metrics", "/docs"]
    response_json = response.json()
    assert response_json["message"] == "MLOps API is running"
    assert all(endpoint in response_json["endpoints"] for endpoint in expected_endpoints)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert "status" in json_response
    assert "model_ready" in json_response
    assert "resource_usage" in json_response
    assert "uptime_seconds" in json_response

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    metrics_text = response.text
    assert "ml_predictions_total" in metrics_text
    assert "process_memory_rss_bytes" in metrics_text
    assert "api_errors_total" in metrics_text

def test_predict_valid():
    payload = {"prompt": "Hello, world!", "max_new_tokens": 10}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    json_response = response.json()
    assert "generated_text" in json_response
    assert isinstance(json_response["generated_text"], str)
    assert len(json_response["generated_text"]) > 0
    assert "model_version" in json_response

def test_predict_invalid_schema():
    # Test with missing prompt
    payload = {"max_new_tokens": 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    # Test with invalid max_new_tokens
    payload = {"prompt": "test", "max_new_tokens": -5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
