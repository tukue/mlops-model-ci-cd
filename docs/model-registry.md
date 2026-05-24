# Model Registry

## Overview

The custom model registry (`src/model_registry.py`) manages model versions, deployment logic, and rollback capability.

## Features

- **Versioning**: Models are versioned by timestamp or tag
- **Metadata**: Stores accuracy, training date, feature schema
- **Load Logic**: Selects active version from `artifacts/`
- **Rollback**: Reverts to previous version by updating registry pointer
- **Graceful Degradation**: API returns degraded health if model is missing

## Registry Structure

```
artifacts/
├── v1/               # Version 1
│   ├── model.pkl
│   └── metadata.json
├── v2/               # Version 2 (current)
│   ├── model.pkl
│   └── metadata.json
└── registry.json     # Active version pointer
```

## API Integration

The registry is used by `app/main.py` to load the active model at startup:

```python
MODEL_PATH = Path("artifacts/Qwen2.5-0.5B-Instruct")
SKIP_MODEL_LOAD_ON_STARTUP = os.getenv("SKIP_MODEL_LOAD_ON_STARTUP", "")

@app.on_event("startup")
def startup_event():
    if SKIP_MODEL_LOAD_ON_STARTUP:
        return
    get_model()  # Loads from registry
```

## Environment Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID |
| `MODEL_PATH` | `./artifacts/Qwen2.5-0.5B-Instruct` | Local model path |
| `SKIP_MODEL_LOAD_ON_STARTUP` | `false` | Skip startup model load |
