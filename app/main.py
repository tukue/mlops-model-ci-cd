import json
import logging
import os
import time
import uuid
from pathlib import Path

import joblib
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from app.schemas import PredictRequest, PredictResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from src.model_registry import ModelRegistry
from src.detect_drift import DRIFT_REPORT_PATH

app = FastAPI(title="MLOps ci-cd API")
logger = logging.getLogger("mlops_api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
PROCESS = psutil.Process(os.getpid())
START_TIME = time.time()

# Prometheus metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
API_REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency by method and endpoint",
    ["method", "endpoint"],
)
API_ERRORS = Counter(
    "api_errors_total",
    "Total unhandled API errors",
    ["method", "endpoint", "exception_type"],
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Prediction failures",
    ["reason"],
)
PREDICTION_CLASS_COUNT = Counter(
    "ml_prediction_class_total",
    "Predicted class distribution",
    ["prediction_class"],
)
MODEL_LOAD_COUNT = Counter("ml_model_load_total", "Model load attempts", ["status"])
MODEL_LOADED = Gauge("ml_model_loaded", "Model load status: 1=loaded, 0=not loaded")
PROCESS_MEMORY_RSS_BYTES = Gauge("process_memory_rss_bytes", "Process resident memory in bytes")
PROCESS_CPU_PERCENT = Gauge("process_cpu_percent", "Process CPU usage percent")
PROCESS_THREAD_COUNT = Gauge("process_thread_count", "Process thread count")
API_INFLIGHT_REQUESTS = Gauge("api_inflight_requests", "Requests currently being processed")
DRIFT_DETECTED = Gauge("ml_drift_detected", "Drift flag from latest drift report (1=true, 0=false)")
DRIFTED_FEATURE_COUNT = Gauge("ml_drifted_feature_count", "Count of features drifting in latest report")
SERVICE_UPTIME_SECONDS = Gauge("service_uptime_seconds", "API process uptime in seconds")

# Use absolute path relative to project root
# Check for an environment variable first, otherwise fall back to the default path.
# This makes the model path configurable, which is great for CI/CD and production.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent.parent / "artifacts" / "model.joblib"))

_model = None
_model_version = None


def update_resource_metrics() -> None:
    PROCESS_MEMORY_RSS_BYTES.set(PROCESS.memory_info().rss)
    PROCESS_CPU_PERCENT.set(PROCESS.cpu_percent(interval=None))
    PROCESS_THREAD_COUNT.set(PROCESS.num_threads())
    SERVICE_UPTIME_SECONDS.set(time.time() - START_TIME)


def update_drift_metrics() -> None:
    if not DRIFT_REPORT_PATH.exists():
        DRIFT_DETECTED.set(0)
        DRIFTED_FEATURE_COUNT.set(0)
        return

    try:
        with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.exception("failed_to_parse_drift_report path=%s", DRIFT_REPORT_PATH)
        DRIFT_DETECTED.set(0)
        DRIFTED_FEATURE_COUNT.set(0)
        return

    drift_detected = bool(report.get("drift_detected", False))
    drifted_features = report.get("drifted_features", {})
    DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFTED_FEATURE_COUNT.set(len(drifted_features))


def get_model():
    global _model, _model_version
    if _model is None:
        if not MODEL_PATH.exists():
            MODEL_LOAD_COUNT.labels(status="failure").inc()
            MODEL_LOADED.set(0)
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first or check MODEL_PATH env var."
            )
        try:
            _model = joblib.load(MODEL_PATH)
            _model_version = f"mtime_{int(MODEL_PATH.stat().st_mtime)}"
            MODEL_LOAD_COUNT.labels(status="success").inc()
            MODEL_LOADED.set(1)
        except Exception:
            MODEL_LOAD_COUNT.labels(status="failure").inc()
            MODEL_LOADED.set(0)
            raise
    return _model, _model_version

@app.middleware("http")
async def track_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    status_code = 500
    API_INFLIGHT_REQUESTS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        API_ERRORS.labels(
            method=request.method,
            endpoint=request.url.path,
            exception_type=exc.__class__.__name__,
        ).inc()
        logger.exception(
            "request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        duration_seconds = time.perf_counter() - start_time
        API_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code),
        ).inc()
        API_REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration_seconds)
        API_INFLIGHT_REQUESTS.dec()
        update_resource_metrics()
        logger.info(
            "request_completed request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration_seconds * 1000,
        )

@app.get("/")
def root():
    return {
        "message": "MLOps API is running",
        "endpoints": ["/health", "/predict", "/metrics", "/model-info", "/drift-status", "/docs"],
    }

@app.get("/health")
def health():
    update_resource_metrics()
    model_ready = MODEL_PATH.exists()
    return {
        "status": "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "uptime_seconds": round(time.time() - START_TIME, 3),
        "resource_usage": {
            "memory_rss_bytes": int(PROCESS.memory_info().rss),
            "cpu_percent": float(PROCESS.cpu_percent(interval=None)),
            "thread_count": int(PROCESS.num_threads()),
        },
    }

@app.get("/model-info")
def model_info():
    try:
        registry = ModelRegistry()
        version, _ = registry.get_active_model()
        metadata = registry.list_models()
        if version and version in metadata:
            return {
                "active_version": version,
                "created_at": metadata[version]["created_at"],
                "metrics": metadata[version]["metrics"]
            }
        return {"error": "No active model found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    update_resource_metrics()
    update_drift_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with PREDICTION_LATENCY.time():
        try:
            model, version = get_model()
            pred = int(model.predict([req.features])[0])
            PREDICTION_COUNT.inc()
            PREDICTION_CLASS_COUNT.labels(prediction_class=str(pred)).inc()
            return PredictResponse(prediction=pred, model_version=version)
        except FileNotFoundError as e:
            PREDICTION_ERRORS.labels(reason="model_unavailable").inc()
            raise HTTPException(status_code=503, detail=str(e))
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            PREDICTION_ERRORS.labels(reason="inference_failure").inc()
            logger.exception("prediction_inference_failed error=%s", exc.__class__.__name__)
            raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")


@app.get("/drift-status")
def drift_status():
    if not DRIFT_REPORT_PATH.exists():
        DRIFT_DETECTED.set(0)
        DRIFTED_FEATURE_COUNT.set(0)
        return {"status": "no_report", "report_path": str(DRIFT_REPORT_PATH)}

    try:
        with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.exception("drift_report_parse_failed path=%s", DRIFT_REPORT_PATH)
        raise HTTPException(status_code=500, detail="Drift report exists but is not readable.")

    drift_detected = bool(report.get("drift_detected", False))
    drifted_features = report.get("drifted_features", {})
    DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFTED_FEATURE_COUNT.set(len(drifted_features))

    return {
        "status": "drift_detected" if drift_detected else "no_drift",
        "drift_detected": drift_detected,
        "drifted_features": list(drifted_features.keys()),
        "report_path": str(DRIFT_REPORT_PATH),
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={"message": "No favicon"}, status_code=200)
