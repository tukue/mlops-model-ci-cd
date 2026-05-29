# API Reference

> All examples use `http://localhost:8000`. Replace with your deployment URL.
> Set `BASE_URL=http://your-domain:8000` and substitute in commands.

**Default Base URL**: `http://localhost:8000`

## Endpoints

### GET /

Returns API information and available endpoints.

**Response**
```json
{
  "message": "MLOps API is running",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "endpoints": ["/health", "/predict", "/drift-status", "/metrics", "/docs"]
}
```

---

### GET /health

Health check with readiness status and resource snapshot.

**Response**
```json
{
  "status": "ok",
  "model_ready": true,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "uptime_seconds": 312.5,
  "resource_usage": {
    "memory_rss_bytes": 245760000,
    "cpu_percent": 12.5,
    "thread_count": 8
  }
}
```

`status` is `"ok"` when model is loaded, `"degraded"` otherwise.

---

### POST /predict

Run model inference with configurable generation parameters.

**Request**
```json
{
  "prompt": "Write one sentence about cloud engineering.",
  "max_new_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | — | Input text prompt (min 1 char) |
| `max_new_tokens` | int | 150 | Max tokens to generate (1–150) |
| `temperature` | float | 0.7 | Sampling temperature (0.1–1.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold (0.1–1.0) |
| `top_k` | int | 50 | Top-k sampling (>= 1) |

**Response**
```json
{
  "generated_text": "Cloud engineering is the practice of designing, building, and maintaining infrastructure and applications on cloud platforms.",
  "model_version": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

**Example**
```bash
curl -X POST ${BASE_URL:-http://localhost:8000}/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_new_tokens": 30}'
```

---

### GET /drift-status

Returns the latest drift detection summary from the saved drift report.

**Response**
```json
{
  "status": "ok",
  "drift_detected": false,
  "drifted_feature_count": 0,
  "drifted_features": [],
  "metrics": {},
  "report_path": "/app/artifacts/drift_report.json"
}
```

When no report exists:
```json
{
  "status": "unavailable",
  "drift_detected": false,
  "drifted_feature_count": 0,
  "drifted_features": [],
  "message": "Drift report has not been generated yet."
}
```

---

### GET /metrics

Prometheus metrics endpoint for scraping. Returns `text/plain` in Prometheus exposition format.

**Example metrics**
```
# HELP ml_predictions_total Total predictions made
# TYPE ml_predictions_total counter
ml_predictions_total 42.0
# HELP ml_prediction_duration_seconds Prediction latency
# TYPE ml_prediction_duration_seconds histogram
ml_prediction_duration_seconds_bucket{le="0.005"} 0.0
ml_prediction_duration_seconds_bucket{le="0.01"} 5.0
...
```

---

### GET /docs

Interactive Swagger UI documentation.
