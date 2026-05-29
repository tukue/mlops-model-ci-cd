# Monitoring & Observability

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| **Metrics** | Prometheus client | Custom app metrics (latency, errors, drift, resources) at `GET /metrics` |
| **Traces** | OpenTelemetry SDK | Manual GenAI semantic spans around `model.generate()` |
| **Auto-instrumentation** | OpenLIT | Automatic LLM call tracing for OpenAI, Hugging Face, etc. |
| **Collection** | OTEL Collector | Receives OTLP, batches, exports to Tempo + Prometheus |
| **Tracing backend** | Grafana Tempo | Stores and queries distributed traces |
| **Dashboards** | Grafana | Visualizes metrics + traces in unified LLM dashboard |

## Data Flow

```
FastAPI /predict  ──OTLP──►  OTEL Collector  ──►  Tempo (traces)
                         │
                         └──►  Prometheus (metrics)
                                  │
                                  ▼
                              Grafana Dashboards
```

## Prometheus Metrics

All Prometheus metrics are exported at `GET /metrics`:



### Prediction Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ml_predictions_total` | Counter | — | Total predictions served |
| `ml_prediction_duration_seconds` | Histogram | — | Prediction latency distribution |
| `ml_prediction_errors_total` | Counter | `reason` | Inference failures by reason |
| `ml_prediction_class_distribution_total` | Counter | `class_name` | Output length buckets (short/medium/long/empty) |

### API Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `api_requests_total` | Counter | `method`, `endpoint`, `status` | Request volume |
| `api_request_duration_seconds` | Histogram | `method`, `endpoint` | API latency by route |
| `api_errors_total` | Counter | `method`, `endpoint`, `exception_type` | Unhandled exceptions |
| `api_inflight_requests` | Gauge | — | Concurrent request count |

### Model Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ml_model_load_total` | Counter | `status` | Model load attempt count |
| `ml_model_loaded` | Gauge | — | 1 = loaded, 0 = not loaded |

### Drift Metrics

| Metric | Type | Description |
|---|---|---|
| `ml_drift_detected` | Gauge | 1 if drift detected, 0 otherwise |
| `ml_drifted_feature_count` | Gauge | Number of drifted features |

### Resource Metrics

| Metric | Type | Description |
|---|---|---|
| `process_memory_rss_bytes` | Gauge | Resident memory in bytes |
| `process_cpu_percent` | Gauge | CPU usage percent |
| `process_thread_count` | Gauge | Thread count |
| `service_uptime_seconds` | Gauge | Process uptime in seconds |

---

## /health Endpoint

Returns readiness, model status, uptime, and resource snapshot.

```
GET /health
```

**States**:
- `"ok"` — model loaded and ready
- `"degraded"` — model not loaded, API still serves health checks

---

## /drift-status Endpoint

Returns the latest drift detection summary from `drift_report.json`.

```
GET /drift-status
```

**Status values**:
- `"ok"` — report loaded successfully
- `"unavailable"` — no report file exists
- `"read_error"` — file cannot be read
- `"invalid_report"` — file is not valid JSON

---

## Request Tracking Middleware

Every request is logged with:
- Request ID (UUID)
- HTTP method and path
- Response status code
- Latency in milliseconds

Middleware also increments `api_errors_total` for 4xx/5xx responses and unhandled exceptions.
