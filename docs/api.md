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
  "endpoints": ["/health", "/predict", "/drift-status", "/shadow-status", "/metrics", "/docs"]
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

> **Shadow mode**: when `SHADOW_ENABLED` is set, each request is also copied
> to the shadow candidate model in the background (see `/shadow-status`).
> The response above is always produced by the active model — shadow output
> never leaks into the client response.

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

### GET /shadow-status

Returns the state of the shadow (candidate) model deployment.

**Response**
```json
{
  "status": "ok",
  "enabled": true,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct-FineTuned",
  "model_ready": true,
  "queue_length": 0,
  "log_path": "artifacts/shadow/shadow_log.jsonl",
  "skipped": 0,
  "queued": 12,
  "dropped": 0,
  "processed": 12,
  "failed": 0,
  "agreement_identical": 8,
  "agreement_differing": 4,
  "recent_entries": ["..."]
}
```

When shadow deployment is not configured (`SHADOW_ENABLED` unset or no shadow
model given), `status` is `"disabled"` and all counters are zero.

| Field | Description |
|---|---|
| `status` | `ok` when enabled, `disabled` otherwise |
| `enabled` | Whether shadow traffic capture is active |
| `model_name` | Configured shadow model id or path |
| `model_ready` | Whether the shadow model is loaded in memory |
| `queue_length` | Pending shadow jobs awaiting the background worker |
| `processed` | Shadow predictions completed (success or failure) |
| `failed` | Shadow predictions that errored (client response unaffected) |
| `agreement_identical` / `agreement_differing` | Shadow output vs active output comparison buckets |
| `recent_entries` | Last N records from the shadow comparison log |

**Example**
```bash
curl ${BASE_URL:-http://localhost:8000}/shadow-status
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

---

## Shadow Deployment Configuration

> A candidate ("challenger") model can be safely evaluated against production
> traffic before promotion: live requests are copied to the shadow model in the
> background while the active model keeps answering. See `/shadow-status` and
> the shadow JSONL log for comparison results.

| Variable | Default | Description |
|---|---|---|
| `SHADOW_ENABLED` | `false` | Set to `1`/`true`/`yes` to enable shadow traffic |
| `SHADOW_MODEL_NAME` | — | Hugging Face model ID of the shadow candidate |
| `SHADOW_MODEL_PATH` | — | Local artifact path of the shadow candidate (preferred over `SHADOW_MODEL_NAME`) |
| `SHADOW_LOG_PATH` | `artifacts/shadow/shadow_log.jsonl` | JSONL comparison log location |
| `SHADOW_QUEUE_MAX` | `100` | Bounded worker queue size; excess jobs are dropped and counted |
| `SHADOW_STATUS_ENTRIES` | `50` | Number of recent log entries returned by `/shadow-status` |

**Example**
```bash
SHADOW_ENABLED=1 \
SHADOW_MODEL_PATH=/app/artifacts/Qwen2.5-0.5B-Instruct-FineTuned \
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
