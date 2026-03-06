from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    expected_endpoints = ["/health", "/predict", "/metrics", "/feedback", "/docs"]
    response_json = response.json()
    assert response_json["message"] == "MLOps API is running"
    assert all(endpoint in response_json["endpoints"] for endpoint in expected_endpoints)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert "status" in json_response
    assert "model_ready" in json_response
    assert "uptime_seconds" in json_response

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    metrics_text = response.text
    assert "ml_predictions_total" in metrics_text
    assert "process_memory_rss_bytes" in metrics_text
    assert "llm_prompt_tokens_total" in metrics_text
    assert "llm_generated_tokens_total" in metrics_text

def test_predict_valid():
    payload = {"prompt": "Hello, world!", "max_new_tokens": 10}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    json_response = response.json()
    assert "generated_text" in json_response
    assert isinstance(json_response["generated_text"], str)
    assert len(json_response["generated_text"]) > 0
    assert "model_version" in json_response
    assert "request_id" in json_response

def test_predict_invalid_schema():
    # Test with missing prompt
    payload = {"max_new_tokens": 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    # Test with invalid max_new_tokens
    payload = {"prompt": "test", "max_new_tokens": -5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_feedback():
    # First make a prediction to get a request_id (though feedback doesn't strictly validate it exists in this simple impl)
    payload = {"prompt": "Test prompt", "max_new_tokens": 5}
    pred_response = client.post("/predict", json=payload)
    request_id = pred_response.json().get("request_id", "dummy-id")

    feedback_payload = {
        "request_id": request_id,
        "score": 5,
        "comment": "Great response!"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_feedback_invalid_score():
    feedback_payload = {
        "request_id": "dummy-id",
        "score": 6, # Invalid score > 5
        "comment": "Too good!"
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 422
