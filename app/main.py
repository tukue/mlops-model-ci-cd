from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="MLOps ci-cd API")

# Use absolute path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first: python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/")
def root():
    return {"message": "MLOps API is running", "endpoints": ["/health", "/predict", "/docs"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = get_model()
        pred = int(model.predict([req.features])[0])
        return PredictResponse(prediction=pred)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(content={"message": "No favicon"}, status_code=200)
