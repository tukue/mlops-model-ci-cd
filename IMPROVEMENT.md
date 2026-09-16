# MLOps Pipeline Improvements

## LLMOps Best Practices

### CI/CD for LLMs

*   [x] **Model Versioning:** Implement robust model versioning using a tool like DVC or MLflow. Track model weights, configurations, and training data.
*   [x] **Automated Testing:**  Include automated tests for model performance, bias, and security. Use a dedicated testing framework.
*   [ ] **Infrastructure as Code (IaC):** Define and manage infrastructure using tools like Terraform or CloudFormation.
*   [x] **Monitoring and Observability:** Implement comprehensive monitoring of model performance, resource utilization, and error rates.
    - Added Prometheus metrics for request latency, API errors, prediction errors, class distribution, model load status, in-flight requests, and uptime.
    - Added process resource gauges (CPU, memory RSS, thread count).
    - Added drift observability via `/drift-status` endpoint and drift gauges (`ml_drift_detected`, `ml_drifted_feature_count`).
    - Enhanced `/health` with readiness (`model_ready`), uptime, and resource snapshot.
    - Added structured request logging with request ID, latency, path, and status.
    - Hardened drift report reads against file-system errors and JSON decode failures.
    - Removed redundant model import null checks after lazy dependency loading.

### Specific Improvements

*   [ ]  Improve model evaluation metrics.
*   [x]  Implement shadow deployment for new models.
    - Added a shadow ("dark launch") deployment mechanism: a candidate model runs alongside the active model on live traffic while the client response always comes from the active model.
    - Shadow inference runs in a serialized background worker with a bounded queue, so it never adds latency to, or blocks, the active serving path.
    - Every shadow prediction is compared to the active output and appended to a JSONL log (`artifacts/shadow/shadow_log.jsonl`) with request ID, prompt, both outputs, token counts, latency, and agreement (`identical` / `differing` / `error`).
    - Added `/shadow-status` endpoint exposing config, model readiness, queue depth, and processed/failed/agreement counters.
    - Added Prometheus metrics: `ml_shadow_requests_total{outcome}`, `ml_shadow_duration_seconds`, `ml_shadow_agreement_total{agreement}`, `ml_shadow_errors_total{error_type}`, `ml_shadow_queue_length`, `ml_shadow_model_loaded`.
    - Shadow failures are isolated and counted; they never affect the client response.
    - Enabled via env vars: `SHADOW_ENABLED=1` plus `SHADOW_MODEL_NAME` (HF id) or `SHADOW_MODEL_PATH` (local artifact dir).
    - Security hardening:
        - `SHADOW_MODEL_PATH` is validated to stay inside the project root (CWE-22 path traversal).
        - Shadow generation caps `max_new_tokens` at 1000 to prevent resource exhaustion (CWE-770).
        - Shadow models load with `trust_remote_code=False` (no arbitrary remote code execution).
        - Shadow log records redact prompt/response text to a SHA-256 prefix; `/shadow-status` no longer exposes raw prompts or outputs (CWE-532).
        - `SHADOW_QUEUE_MAX` / `SHADOW_STATUS_ENTRIES` fall back to safe defaults when non-numeric or non-positive.
        - Shadow worker starts only after the active model has loaded successfully, and `stop()` handles a still-alive worker gracefully.
*   [ ]  Automate data validation.

## Recent Implementation Notes

*   Switched MLflow default tracking backend to SQLite (`sqlite:///mlflow.db`) to avoid deprecated filesystem tracking backend usage.
*   Updated inference input to preserve feature names and remove sklearn feature-name mismatch warnings.
*   Updated API tests to cover new observability endpoints/metrics.
*   Implemented shadow deployment (`app/shadow.py`): background candidate-model inference with JSONL comparison logging, `/shadow-status`, and Prometheus metrics; wired into `/predict` and lifecycle events.
*   Hardened shadow deployment: path-traversal validation, `trust_remote_code=False`, capped `max_new_tokens`, PII-redacted JSONL logging, redacted `/shadow-status`, validated env vars, ordered startup, and graceful worker stop.
*   Hardened `/drift-status` against unreadable report files and invalid JSON payloads.
*   Removed dead defensive code from model loading after import-time dependency checks were simplified.
