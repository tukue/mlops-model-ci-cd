from fastapi.testclient import TestClient

from app import main
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
    expected_endpoints = ["/health", "/predict", "/drift-status", "/metrics", "/docs"]
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
    assert "ml_drift_detected" in metrics_text
    assert "ml_drifted_feature_count" in metrics_text

def test_drift_status_with_report(tmp_path, monkeypatch):
    report_path = tmp_path / "drift_report.json"
    report_path.write_text(
        """
        {
            "drift_detected": true,
            "drifted_features": {"sepal length (cm)": 0.001, "petal width (cm)": 0.02},
            "metrics": {"sepal length (cm)": {"p_value": 0.001}}
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "DRIFT_REPORT_PATH", report_path)

    response = client.get("/drift-status")

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert json_response["drift_detected"] is True
    assert json_response["drifted_feature_count"] == 2
    assert "sepal length (cm)" in json_response["drifted_features"]

def test_drift_status_without_report(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DRIFT_REPORT_PATH", tmp_path / "missing_report.json")

    response = client.get("/drift-status")

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "unavailable"
    assert json_response["drift_detected"] is False
    assert json_response["drifted_feature_count"] == 0

def test_drift_status_invalid_report(tmp_path, monkeypatch):
    report_path = tmp_path / "drift_report.json"
    report_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(main, "DRIFT_REPORT_PATH", report_path)

    response = client.get("/drift-status")

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "invalid_report"
    assert json_response["drift_detected"] is False
    assert json_response["drifted_feature_count"] == 0

def test_metrics_include_drift_report_values(tmp_path, monkeypatch):
    report_path = tmp_path / "drift_report.json"
    report_path.write_text(
        '{"drift_detected": true, "drifted_features": {"feature_a": 0.01}, "metrics": {}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "DRIFT_REPORT_PATH", report_path)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "ml_drift_detected 1.0" in response.text
    assert "ml_drifted_feature_count 1.0" in response.text

class DummyInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

class DummyInputIds:
    shape = (1, 2)

class DummyTokenizer:
    chat_template = None
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = "<eos>"

    def __call__(self, texts, return_tensors):
        return DummyInputs(input_ids=DummyInputIds())

    def decode(self, tokens, skip_special_tokens=True):
        return "hello from test"

class DummyModel:
    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]

def test_predict_valid(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: (DummyTokenizer(), DummyModel()))
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
