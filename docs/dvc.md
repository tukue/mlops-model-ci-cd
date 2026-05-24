# DVC Data Versioning

## Overview

DVC (Data Version Control) tracks datasets, model artifacts, and pipeline stages that shouldn't be stored in Git.

## Key Files

| File | Purpose |
|---|---|
| `dvc.yaml` | Defines pipeline stages (training) |
| `dvc.lock` | Locks dependency/output versions for reproducibility |
| `.dvc/` | DVC internal cache and config |

## Pipeline Stages

Defined in `dvc.yaml`:

```yaml
stages:
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/
    outs:
      - artifacts/model.pkl
```

## Common Commands

```bash
# Reproduce the full pipeline
dvc repro

# Show pipeline DAG
dvc dag

# Track changes after repro
git add dvc.yaml dvc.lock
git commit -m "Update model pipeline"
```

## Best Practices

- Run `dvc repro` locally before pushing code changes
- Commit both `dvc.yaml` and `dvc.lock` to Git
- Use `.dvcignore` to exclude irrelevant files from DVC tracking
- Artifacts in `artifacts/` are Git-ignored but DVC-tracked
