# Architecture

## System Overview

```mermaid
flowchart TB
    classDef dev fill:#1a1a2e,color:#fff,stroke:#e94560,stroke-width:2px
    classDef ci fill:#16213e,color:#fff,stroke:#0f3460,stroke-width:2px
    classDef reg fill:#533483,color:#fff,stroke:#e94560,stroke-width:2px
    classDef serve fill:#2d4059,color:#fff,stroke:#00adb5,stroke-width:2px
    classDef obs fill:#1b1b2f,color:#fff,stroke:#f08a5d,stroke-width:2px
    classDef ext fill:#111,color:#fff,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5

    subgraph DEV["Development"]
        direction LR
        CODE["Python / FastAPI<br/>Source Code"] --> GIT["GitHub<br/>Repository"]
        DVC["DVC Pipeline<br/>dvc.yaml"] --> DATA["Datasets &<br/>Model Artifacts"]
        GIT --> DVC
    end
    class DEV dev

    subgraph CI_CD["CI/CD Pipeline — GitHub Actions"]
        direction TB
        GIT --> SETUP["⚡ Setup<br/>Python 3.9 + Dependencies"]
        SETUP --> TRAIN["🧠 Train<br/>src/train.py"]
        TRAIN --> TEST["✅ Test<br/>pytest tests/ -v"]
        TEST --> BUILD["🐳 Build<br/>Docker Image"]
        BUILD --> VALIDATE["🔍 Validate<br/>Smoke Test /health & /predict"]
    end
    class CI_CD ci

    subgraph REGISTRY["Model Registry"]
        direction LR
        TRAIN --> V1["Version 1"]
        TRAIN --> V2["Version 2 ✓ Active"]
        TRAIN --> V3["Version 3"]
        V1 --> META["Metadata: accuracy,<br/>date, schema"]
        V2 --> META
        V3 --> META
        META --> ROLLBACK["Rollback:<br/>revert pointer"]
    end
    class REGISTRY reg

    subgraph SERVE["Serving Layer"]
        direction TB
        BUILD --> DOCKER["Docker Container<br/>docker-compose up"]
        DOCKER --> API["FastAPI Server<br/>uvicorn app.main:app"]
        API --> PREDICT["POST /predict<br/>LLM Inference"]
        API --> HEALTH["GET /health<br/>Readiness + Resources"]
        API --> METRICS["GET /metrics<br/>Prometheus Export"]
    end
    class SERVE serve

    subgraph OBSERVE["Observability Layer"]
        direction TB
        METRICS --> PROM["Prometheus<br/>Scrapes /metrics"]
        PROM --> LATENCY["Response Time<br/>Histogram"]
        PROM --> TRAFFIC["Request Volume<br/>Counter"]
        PROM --> ERRORS["Error Rates<br/>Counter by type"]
        PROM --> DRIFT["Data Drift<br/>Gauge: 0/1"]
        PROM --> RESOURCE["System Resources<br/>Memory / CPU / Threads"]

        PREDICT --> OTEL["OpenTelemetry<br/>Custom GenAI Spans"]
        OTEL --> COLLECTOR["OTEL Collector<br/>Aggregates & Exports"]
        COLLECTOR --> TEMPO["Grafana Tempo<br/>Distributed Tracing"]
        COLLECTOR --> PROM2["Prometheus<br/>OTEL Metrics"]

        PREDICT --> OPENLIT["OpenLIT<br/>Auto-instrumentation"]
        OPENLIT --> COLLECTOR
    end
    class OBSERVE obs

    subgraph EXTERNAL["External Systems"]
        direction LR
        USER["User / Client"] --> API
        PROM --> GRAFANA["Grafana<br/>Dashboards"]
        TEMPO --> GRAFANA
        PROM2 --> GRAFANA
    end
    class EXTERNAL ext

    DEV --> CI_CD
    CI_CD --> REGISTRY
    CI_CD --> SERVE
    SERVE --> OBSERVE
    REGISTRY -.-> |"loads active version"| API
```

## Data Flow

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Developer  │────>│  GitHub Actions  │────>│  Model Registry  │
│   git push   │     │   ci-cd.yaml     │     │  versioned .pt   │
└──────────────┘     └──────────────────┘     └──────────────────┘
                             │                        │
                             │                        ▼
                             │               ┌──────────────────┐
                             │               │  FastAPI Server  │
                             │               │  /predict        │
                             └──────────────>│  /health         │
                                              │  /metrics        │
                                              │  OTel + OpenLIT  │
                                              └────────┬─────────┘
                                                       │
                                          ┌────────────┼────────────┐
                                          ▼            ▼            ▼
                                   ┌──────────┐ ┌──────────┐ ┌──────────┐
                                   │Prometheus│ │  OTEL    │ │  OpenLIT │
                                   │ /metrics │ │ Collector│ │ Auto-inst│
                                   └────┬─────┘ └────┬─────┘ └──────────┘
                                        │            │
                                        ▼            ▼
                                   ┌──────────┐ ┌──────────┐
                                   │ Grafana  │ │  Tempo   │
                                   │Dashboards│ │  Traces  │
                                   └──────────┘ └──────────┘
```

## Component Diagram

```mermaid
flowchart LR
    subgraph APP["app/"]
        MAIN["main.py<br/>• FastAPI routes<br/>• Middleware<br/>• Prometheus metrics"]
        SCHEMAS["schemas.py<br/>• Pydantic models<br/>• Request/Response<br/>• Validation"]
    end

    subgraph SRC["src/"]
        TRAIN["train.py<br/>• Model training<br/>• DVC integration"]
        REG["model_registry.py<br/>• Versioning<br/>• Load/rollback<br/>• Metadata"]
    end

    subgraph TESTS["tests/"]
        TAPP["test_app.py<br/>API tests"]
        TMODEL["test_model.py<br/>Model tests"]
        TDVC["test_dvc.py<br/>DVC tests"]
    end

    subgraph INFRA["Infrastructure"]
        DOCKERFILE["Dockerfile"]
        COMPOSE["docker-compose.yml"]
        WORKFLOW[".github/workflows/<br/>ci-cd.yaml"]
    end

    TRAIN --> REG
    REG --> MAIN
    MAIN --> SCHEMAS
    TESTS --> MAIN
    TESTS --> TRAIN
    INFRA --> MAIN
```

## Technology Choices

| Component | Choice | Why |
|---|---|---|
| **API Framework** | FastAPI | Async-native, automatic OpenAPI docs, Pydantic integration |
| **Model Serving** | Hugging Face Transformers | Industry standard for LLMs, broad model support |
| **CI/CD** | GitHub Actions | Native GitHub integration, free tier, large ecosystem |
| **Data Versioning** | DVC | Git-like semantics for data, pipeline reproducibility |
| **Monitoring** | Prometheus client + OpenTelemetry + OpenLIT | Standard metrics + GenAI semantic traces + auto-instrumentation |
| **Containerization** | Docker + Compose | Portable, reproducible, dev-prod parity |
| **Validation** | Pydantic v2 | Runtime type checking, JSON Schema generation |

## Key Design Properties

- **Stateless**: Each request is independent; scale horizontally with a load balancer
- **Observable**: Every operation is instrumented with metrics, logs, and request IDs
- **Resilient**: Graceful degradation, health checks, structured error handling
- **Reproducible**: DVC locks data and pipeline versions; Docker locks the runtime
- **Configurable**: Model, paths, and behavior controlled via environment variables
