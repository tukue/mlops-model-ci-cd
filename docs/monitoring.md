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

## OpenTelemetry + OpenLIT Telemetry

### What It Measures

Every `/predict` request produces a **distributed trace** with a span tree. The root span is the HTTP request (captured by `opentelemetry-instrumentation-fastapi`), with a child `chat` span wrapping `model.generate()`.

| Measurement | Source | What It Tracks |
|---|---|---|
| **Model identity** | GenAI span attributes | Provider (`huggingface`), model name, response model |
| **Input tokens** | `gen_ai.usage.input_tokens` | Prompt token count before generation |
| **Output tokens** | `gen_ai.usage.output_tokens` | Generated token count (cost driver) |
| **Generation parameters** | `gen_ai.request.*` attributes | `temperature`, `top_p`, `top_k`, `max_tokens` |
| **Completion reason** | `gen_ai.response.finish_reasons` | Why generation stopped (`stop`, `length`, etc.) |
| **LLM call duration** | Span start/end time | End-to-end latency of `model.generate()` |
| **Auto-instrumentation** | OpenLIT | Automatically captures LLM provider calls, token usage, and framework-internal spans (LangChain, etc.) |

### Span Attributes (GenAI Semantic Conventions)

```
Span: "chat"  (inside POST /predict)
├── gen_ai.operation.name        = "chat"
├── gen_ai.provider.name         = "huggingface"
├── gen_ai.request.model         = "Qwen/Qwen2.5-0.5B-Instruct"
├── gen_ai.request.max_tokens    = 150
├── gen_ai.request.temperature   = 0.7
├── gen_ai.request.top_p         = 0.9
├── gen_ai.request.top_k         = 50
├── gen_ai.usage.input_tokens    = 42
├── gen_ai.usage.output_tokens   = 128
├── gen_ai.response.model        = "Qwen/Qwen2.5-0.5B-Instruct"
└── gen_ai.response.finish_reasons = ["stop"]
```

### Trace Query Examples (Grafana Tempo — TraceQL)

```traceql
# All LLM chat spans in the last hour
{ gen_ai.operation.name = "chat" }

# Spans with high token usage
{ gen_ai.usage.output_tokens > 500 }

# Failed/fast generations
{ gen_ai.usage.output_tokens = 0 }
```

### How It Complements Prometheus Metrics

| Concern | Get via Prometheus | Get via OTel Traces |
|---|---|---|
| Request rate | `rate(api_requests_total[1m])` | — |
| Latency distribution | `ml_prediction_duration_seconds` histogram | Per-span duration with exact model params |
| Error count | `ml_prediction_errors_total` | Trace with error status + exception details |
| Token usage | — (alert when present) | Per-request `gen_ai.usage.*` attributes |
| Model identity | — | `gen_ai.request.model` + `gen_ai.response.model` |
| Sampling params | — | `gen_ai.request.temperature/top_p/top_k` |
| Individual request debug | — | Full trace tree with timing per operation |

### Code Reference

The GenAI span is created in `app/main.py:366`:

```python
with TRACER.start_as_current_span("chat") as span:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", "huggingface")
    span.set_attribute("gen_ai.request.model", MODEL_NAME)
    ...
    span.set_attribute("gen_ai.usage.input_tokens", input_token_count)
    # model.generate() runs here
    span.set_attribute("gen_ai.usage.output_tokens", output_token_count)
```

OpenLIT auto-instrumentation is initialized at startup (`app/main.py:97`):

```python
openlit.init(
    service_name=OTEL_SERVICE_NAME,
    otlp_endpoint=OTEL_OTLP_ENDPOINT,
    disable_content_capture=True,
)
```

---

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
