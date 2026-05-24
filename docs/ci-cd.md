# CI/CD Pipeline

## Overview

Two GitHub Actions workflows automate the ML lifecycle:

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci-cd.yaml` | Push/PR to `main` | Test, build, and validate |
| `train-llm.yml` | Manual dispatch | Fine-tune LLM with LoRA |

---

## ci-cd.yaml

### Pipeline Stages

```mermaid
flowchart LR
    CHECKOUT[Checkout] --> SETUP[Setup Python]
    SETUP --> INSTALL[Install Deps]
    INSTALL --> INIT_DVC[Init DVC]
    INIT_DVC --> TEST[pytest]
    TEST --> BUILD[Docker Build]
    BUILD --> RUN[Run Container]
    RUN --> SMOKE[Smoke Test]
    SMOKE --> STOP[Cleanup]
```

### Stage Details

**1. Checkout**
- `actions/checkout@v4` with full Git history

**2. Setup Python**
- `actions/setup-python@v5`, Python 3.9

**3. Install Dependencies**
- `pip install -r requirements.txt`
- Additional: `pytest`, `httpx`

**4. Initialize DVC**
- `dvc init --no-scm` for CI environment
- Creates `artifacts/` directory

**5. Run Tests**
```bash
pytest tests/ -v
```

**6. Build Docker Image**
```bash
docker build -t mlops-api .
```

**7. Smoke Test API (in Docker)**
- Start container, wait up to 60s for `/health`
- Validate `GET /` returns endpoint list
- Validate `GET /health` returns status
- Validate `GET /docs` returns Swagger UI
- Validate `GET /metrics` contains expected metrics
- Validate `POST /predict` returns `generated_text`

---

## train-llm.yml

Triggered manually via `workflow_dispatch` for LLM fine-tuning:

1. Checkout + Python setup
2. Install dependencies (includes `torch`, `transformers`, `peft`, `accelerate`)
3. Run training script
4. Save and version artifacts with DVC
5. Push to remote storage

---

## Local CI Simulation

```bash
# Run tests
pytest tests/ -v

# Build and test Docker
docker build -t mlops-api .
docker run -d --name mlops-test -p 8000:8000 mlops-api
sleep 10
curl http://localhost:8000/health
docker stop mlops-test && docker rm mlops-test
```
