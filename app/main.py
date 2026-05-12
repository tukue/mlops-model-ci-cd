import json
import logging
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.schemas import PredictRequest, PredictResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None

app = FastAPI(title="MLOps CI/CD API")
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
    "Total API errors by method, endpoint, and error type",
    ["method", "endpoint", "exception_type"],
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Prediction failures",
    ["reason"],
)
PREDICTION_CLASS_DISTRIBUTION = Counter(
    "ml_prediction_class_distribution_total",
    "Distribution of prediction output classes",
    ["class_name"],
)
MODEL_LOAD_COUNT = Counter("ml_model_load_total", "Model load attempts", ["status"])
MODEL_LOADED = Gauge("ml_model_loaded", "Model load status: 1=loaded, 0=not loaded")
PROCESS_MEMORY_RSS_BYTES = Gauge("process_memory_rss_bytes", "Process resident memory in bytes")
PROCESS_CPU_PERCENT = Gauge("process_cpu_percent", "Process CPU usage percent")
PROCESS_THREAD_COUNT = Gauge("process_thread_count", "Process thread count")
API_INFLIGHT_REQUESTS = Gauge("api_inflight_requests", "Requests currently being processed")
SERVICE_UPTIME_SECONDS = Gauge("service_uptime_seconds", "API process uptime in seconds")
DRIFT_DETECTED = Gauge("ml_drift_detected", "Latest drift status: 1=drift detected, 0=no drift")
DRIFTED_FEATURE_COUNT = Gauge("ml_drifted_feature_count", "Number of drifted features in latest drift report")

# Update default model name
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
# Corrected default model path to match DVC output
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent.parent / "artifacts" / "Qwen2.5-0.5B-Instruct"))
DRIFT_REPORT_PATH = Path(os.environ.get("DRIFT_REPORT_PATH", Path(__file__).parent.parent / "artifacts" / "drift_report.json"))
SKIP_MODEL_LOAD_ON_STARTUP = os.getenv("SKIP_MODEL_LOAD_ON_STARTUP", "").lower() in {"1", "true", "yes"}

_tokenizer = None
_model = None
MODEL_LOADED.set(0)

def get_model():
    global _tokenizer, _model, torch, AutoModelForCausalLM, AutoTokenizer
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        try:
            import torch as torch_module
            from transformers import AutoModelForCausalLM as causal_lm_class
            from transformers import AutoTokenizer as tokenizer_class
        except ImportError:
            MODEL_LOAD_COUNT.labels(status="failure").inc()
            MODEL_LOADED.set(0)
            raise RuntimeError("PyTorch and Transformers are required for model inference.")

        torch = torch_module
        AutoModelForCausalLM = causal_lm_class
        AutoTokenizer = tokenizer_class

    if _model is None or _tokenizer is None:
        model_path_str = str(MODEL_PATH)
        if MODEL_PATH.exists():
            logger.info("loading_model_from_path path=%s", model_path_str)
            model_source = model_path_str
        else:
            logger.info("loading_model_from_hub model_name=%s", MODEL_NAME)
            model_source = MODEL_NAME

        try:
            # Removed trust_remote_code=True for security
            _tokenizer = AutoTokenizer.from_pretrained(model_source)
            _model = AutoModelForCausalLM.from_pretrained(model_source)

            # Ensure padding token is set
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

            MODEL_LOAD_COUNT.labels(status="success").inc()
            MODEL_LOADED.set(1)
            logger.info("model_loaded_successfully model_name=%s", model_source)
        except Exception:
            MODEL_LOAD_COUNT.labels(status="failure").inc()
            MODEL_LOADED.set(0)
            logger.exception("failed_to_load_model model_name=%s", model_source)
            raise
    return _tokenizer, _model

def update_resource_metrics() -> None:
    PROCESS_MEMORY_RSS_BYTES.set(PROCESS.memory_info().rss)
    PROCESS_CPU_PERCENT.set(PROCESS.cpu_percent(interval=None))
    PROCESS_THREAD_COUNT.set(PROCESS.num_threads())
    SERVICE_UPTIME_SECONDS.set(time.time() - START_TIME)

def classify_prediction_output(generated_text: str) -> str:
    """Bucket generative output so dashboards can show prediction distribution."""
    token_count = len(generated_text.split())
    if token_count == 0:
        return "empty"
    if token_count <= 20:
        return "short"
    if token_count <= 100:
        return "medium"
    return "long"

def load_drift_status() -> dict[str, Any]:
    if not DRIFT_REPORT_PATH.exists():
        DRIFT_DETECTED.set(0)
        DRIFTED_FEATURE_COUNT.set(0)
        return {
            "status": "unavailable",
            "drift_detected": False,
            "drifted_feature_count": 0,
            "drifted_features": [],
            "report_path": str(DRIFT_REPORT_PATH),
            "message": "Drift report has not been generated yet.",
        }

    try:
        with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except OSError:
        DRIFT_DETECTED.set(0)
        DRIFTED_FEATURE_COUNT.set(0)
        logger.exception("could_not_read_drift_report path=%s", DRIFT_REPORT_PATH)
        return {
            "status": "read_error",
            "drift_detected": False,
            "drifted_feature_count": 0,
            "drifted_features": [],
            "report_path": str(DRIFT_REPORT_PATH),
            "message": "Drift report could not be read.",
        }
    except json.JSONDecodeError:
        DRIFT_DETECTED.set(0)
        DRIFTED_FEATURE_COUNT.set(0)
        logger.exception("invalid_drift_report path=%s", DRIFT_REPORT_PATH)
        return {
            "status": "invalid_report",
            "drift_detected": False,
            "drifted_feature_count": 0,
            "drifted_features": [],
            "report_path": str(DRIFT_REPORT_PATH),
            "message": "Drift report is not valid JSON.",
        }

    drifted_features = report.get("drifted_features", {})
    if isinstance(drifted_features, dict):
        drifted_feature_names = list(drifted_features.keys())
    elif isinstance(drifted_features, list):
        drifted_feature_names = list(drifted_features)
    else:
        drifted_feature_names = []

    drift_detected = bool(report.get("drift_detected", False))
    drifted_feature_count = len(drifted_feature_names)
    DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFTED_FEATURE_COUNT.set(drifted_feature_count)

    return {
        "status": "ok",
        "drift_detected": drift_detected,
        "drifted_feature_count": drifted_feature_count,
        "drifted_features": drifted_feature_names,
        "metrics": report.get("metrics", {}),
        "report_path": str(DRIFT_REPORT_PATH),
    }

@app.middleware("http")
async def track_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    status_code = 500
    error_recorded = False
    API_INFLIGHT_REQUESTS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error_recorded = True
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
        if status_code >= 400 and not error_recorded:
            API_ERRORS.labels(
                method=request.method,
                endpoint=request.url.path,
                exception_type=f"http_{status_code}",
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

@app.on_event("startup")
def startup_event():
    if SKIP_MODEL_LOAD_ON_STARTUP:
        logger.info("skipping_model_load_on_startup")
        return

    try:
        get_model()
    except Exception:
        logger.critical("could_not_load_model_on_startup")

@app.get("/")
def root():
    return {
        "message": "MLOps API is running",
        "model_name": MODEL_NAME,
        "endpoints": ["/health", "/predict", "/drift-status", "/metrics", "/docs"],
    }

@app.get("/health")
def health():
    update_resource_metrics()
    model_ready = _model is not None and _tokenizer is not None
    return {
        "status": "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_name": MODEL_NAME,
        "uptime_seconds": round(time.time() - START_TIME, 3),
        "resource_usage": {
            "memory_rss_bytes": int(PROCESS.memory_info().rss),
            "cpu_percent": float(PROCESS.cpu_percent(interval=None)),
            "thread_count": int(PROCESS.num_threads()),
        },
    }

@app.get("/metrics")
def metrics():
    update_resource_metrics()
    load_drift_status()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/drift-status")
def drift_status():
    return load_drift_status()

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with PREDICTION_LATENCY.time():
        try:
            tokenizer, model = get_model()

            # Check for chat template support and apply it
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": req.prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without a chat template
                text = req.prompt

            inputs = tokenizer([text], return_tensors="pt")

            input_length = inputs.input_ids.shape[1]

            no_grad = torch.no_grad() if torch is not None else nullcontext()
            with no_grad:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    do_sample=True,
                    top_k=req.top_k, # Use top_k from the request
                    top_p=req.top_p,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Slice the output to remove the input prompt tokens
            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            PREDICTION_COUNT.inc()
            PREDICTION_CLASS_DISTRIBUTION.labels(
                class_name=classify_prediction_output(generated_text),
            ).inc()

            return PredictResponse(generated_text=generated_text, model_version=MODEL_NAME)

        except Exception as e:
            PREDICTION_ERRORS.labels(reason="inference_failure").inc()
            logger.exception("prediction_inference_failed error=%s", e.__class__.__name__)
            raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={"message": "No favicon"}, status_code=200)
