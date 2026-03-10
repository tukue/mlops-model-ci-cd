import json
import logging
import os
import time
import uuid
from pathlib import Path

import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer, AutoModelForCausalLM # Changed from AutoModelForSeq2SeqLM

from app.schemas import PredictRequest, PredictResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

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
    "Total unhandled API errors",
    ["method", "endpoint", "exception_type"],
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Prediction failures",
    ["reason"],
)
MODEL_LOAD_COUNT = Counter("ml_model_load_total", "Model load attempts", ["status"])
MODEL_LOADED = Gauge("ml_model_loaded", "Model load status: 1=loaded, 0=not loaded")
PROCESS_MEMORY_RSS_BYTES = Gauge("process_memory_rss_bytes", "Process resident memory in bytes")
PROCESS_CPU_PERCENT = Gauge("process_cpu_percent", "Process CPU usage percent")
PROCESS_THREAD_COUNT = Gauge("process_thread_count", "Process thread count")
API_INFLIGHT_REQUESTS = Gauge("api_inflight_requests", "Requests currently being processed")
SERVICE_UPTIME_SECONDS = Gauge("service_uptime_seconds", "API process uptime in seconds")

# Use an instruction-tuned model for better chat-like responses
DEFAULT_MODEL_NAME = "microsoft/DialoGPT-small" # Changed default model name
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent.parent / "artifacts" / "model"))

_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        model_path_str = str(MODEL_PATH)
        if MODEL_PATH.exists():
            logger.info("loading_model_from_path path=%s", model_path_str)
            model_source = model_path_str
        else:
            logger.info("loading_model_from_hub model_name=%s", MODEL_NAME)
            model_source = MODEL_NAME

        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_source)
            _model = AutoModelForCausalLM.from_pretrained(model_source) # Changed from AutoModelForSeq2SeqLM
            # DialoGPT models typically use a specific pad_token or eos_token for generation
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

@app.on_event("startup")
def startup_event():
    try:
        get_model()
    except Exception:
        logger.critical("could_not_load_model_on_startup")

@app.get("/")
def root():
    return {
        "message": "MLOps API is running",
        "model_name": MODEL_NAME,
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
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
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with PREDICTION_LATENCY.time():
        try:
            tokenizer, model = get_model()
            inputs = tokenizer(req.prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id # Added for causal LMs like DialoGPT
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            PREDICTION_COUNT.inc()
            
            return PredictResponse(generated_text=generated_text, model_version=MODEL_NAME)
        
        except Exception as e:
            PREDICTION_ERRORS.labels(reason="inference_failure").inc()
            logger.exception("prediction_inference_failed error=%s", e.__class__.__name__)
            raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={"message": "No favicon"}, status_code=200)
