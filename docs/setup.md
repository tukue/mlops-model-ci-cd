# Setup Guide

> All API examples use `http://localhost:8000`. For a remote deployment, set `BASE_URL` and substitute:
> ```bash
> export BASE_URL=http://your-domain:8000
> curl ${BASE_URL:-http://localhost:8000}/health
> ```

## Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- Git

## Local Development

### 1. Clone and Setup Environment

```bash
git clone <repo-url>
cd mlops-model-ci-cd
bash setup_env.sh
source .venv/Scripts/activate
```

### 2. Initialize DVC

```bash
dvc init
dvc repro  # Run training pipeline
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Start API

```bash
uvicorn app.main:app --reload
```

### 5. Verify

```bash
BASE_URL="${BASE_URL:-http://localhost:8000}"

# Health check
curl $BASE_URL/health

# Prediction
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_new_tokens": 30}'

# Metrics
curl $BASE_URL/metrics
```

---

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

### Docker Compose Configuration

```yaml
services:
  mlops-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./artifacts:/app/artifacts
    environment:
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID or local path |
| `MODEL_PATH` | `./artifacts/Qwen2.5-0.5B-Instruct` | Local model path |
| `SKIP_MODEL_LOAD_ON_STARTUP` | `false` | Skip loading model on startup |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DRIFT_REPORT_PATH` | `./artifacts/drift_report.json` | Drift report file path |

---

## Common Issues

**Model fails to load**: Ensure `MODEL_PATH` points to a valid model directory or `MODEL_NAME` is a valid Hugging Face model ID.

**Port conflict**: Change the port with `--port` flag or modify `docker-compose.yml`.

**DVC not found**: Install with `pip install dvc`.
