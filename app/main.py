import os
from pathlib import Path
import joblib
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from app.schemas import PredictRequest, PredictResponse
from prometheus_client import Counter, Histogram, generate_latest
from src.model_registry import ModelRegistry

app = FastAPI(title="MLOps ci-cd API")

# Prometheus metrics
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])

# Use absolute path relative to project root
# Check for an environment variable first, otherwise fall back to the default path.
# This makes the model path configurable, which is great for CI/CD and production.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).parent.parent / "artifacts" / "model.joblib"))

_model = None
_model_version = None

def get_model():
    global _model, _model_version
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first or check MODEL_PATH env var."
            )
        _model = joblib.load(MODEL_PATH)
        # For simplicity, we'll just use the file's existence as the "version" in this context
        _model_version = "loaded_from_path"
    return _model, _model_version

@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    API_REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.get("/")
def root():
    return {"message": "MLOps API is running", "endpoints": ["/health", "/predict", "/metrics", "/model-info", "/docs"]}

@app.get("/health")
def health():
    return {"status": "ok"}

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
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with PREDICTION_LATENCY.time():
        try:
            model, version = get_model()
            pred = int(model.predict([req.features])[0])
            PREDICTION_COUNT.inc()
            return PredictResponse(prediction=pred, model_version=version)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={"message": "No favicon"}, status_code=200)
