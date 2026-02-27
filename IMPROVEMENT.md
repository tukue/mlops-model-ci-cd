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

### Specific Improvements

*   [ ]  Improve model evaluation metrics.
*   [ ]  Implement shadow deployment for new models.
*   [ ]  Automate data validation.

## Recent Implementation Notes

*   Switched MLflow default tracking backend to SQLite (`sqlite:///mlflow.db`) to avoid deprecated filesystem tracking backend usage.
*   Updated inference input to preserve feature names and remove sklearn feature-name mismatch warnings.
*   Updated API tests to cover new observability endpoints/metrics.
