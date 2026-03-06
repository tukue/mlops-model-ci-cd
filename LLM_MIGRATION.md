# LLM Migration Plan: From Classical ML to Transformer-Based Workflows

This document outlines the strategy and concrete steps for refactoring this MLOps repository from a classical machine learning setup (scikit-learn, joblib) to a modern, LLM-based workflow using Hugging Face Transformers.

The core philosophy is to enable rapid, reliable, and resource-efficient development suitable for both local laptops and CI/CD environments like GitHub Actions.

---

## 1. Keep GitHub Actions for CI, but Make it Inference-First

The CI pipeline should be fast, lightweight, and focused on validation, not on expensive training.

**Use PR CI to verify:**
- Dependencies install correctly (`pip install`).
- The tokenizer and a tiny model can be loaded from Hugging Face or a local path.
- The FastAPI application starts without errors.
- The `/health` endpoint returns a `200 OK` status.
- The `/predict` endpoint accepts a text prompt and returns a non-empty text response.
- The Docker image builds successfully.

For this, we will use a tiny Hugging Face model such as **`sshleifer/tiny-gpt2`** or **`distilgpt2`** both locally and in CI. This ensures that the pipeline logic is sound without requiring powerful hardware.

**The PR pipeline becomes:**
`push / PR` → `install dependencies` → `load tiny model` → `run tests` → `build Docker image` → `smoke test API`

This workflow is realistic on a standard laptop and on free GitHub-hosted runners.

## 2. Do Not Retrain in Every PR

The current `dvc repro` flow is designed for classical models where retraining is fast. With LLMs, even small fine-tuning jobs are computationally heavy and slow.

We will adopt two distinct workflows:

**Workflow A: `ci.yml`**
- **Trigger**: Runs on every push and pull request.
- **Purpose**: Verifies the integrity and functionality of the application.
- **Jobs**:
  - Linting and static analysis.
  - Unit and integration tests.
  - Building the application (e.g., Docker image).
  - Starting the API with a tiny, fixed model.
  - Smoke testing critical endpoints (`/health`, `/predict`).

**Workflow B: `train-llm.yml`**
- **Trigger**: Runs only on manual dispatch (`workflow_dispatch`), on a schedule, or on a protected branch merge.
- **Purpose**: To execute the actual model fine-tuning process.
- **Jobs**:
  - Pulling the full dataset from a remote source (like DVC).
  - Running the fine-tuning script (`train.py`).
  - Evaluating the model and logging metrics to MLflow.
  - Versioning and pushing the trained model artifact (e.g., LoRA adapters) to a registry or DVC remote.

This separation is the single most important change for achieving a practical LLM MLOps cycle.

## 3. Fine-Tune with LoRA, Not Full-Model Training

When training is required, we will use Parameter-Efficient Fine-Tuning (PEFT), specifically **LoRA (Low-Rank Adaptation)**. This approach significantly reduces the computational burden of fine-tuning.

**Benefits for this project:**
- **Frozen Base Model**: The large pre-trained model's weights remain unchanged.
- **Trainable Adapters**: Only a small number of new parameters (the LoRA adapter weights) are trained.
- **Small Artifacts**: The resulting artifacts are typically only a few megabytes, making them easy to store and manage.
- **Laptop-Friendly**: LoRA makes fine-tuning on a local machine with a consumer GPU a realistic goal.

This is the sweet spot for portfolio-level practice, demonstrating modern LLM training techniques without requiring an enterprise-level budget.

## 4. Train Locally, Validate in GitHub

The recommended development loop is:

1.  **Laptop**: Run the actual fine-tuning script on a small, representative dataset to produce a model artifact (e.g., LoRA adapters).
2.  **Git**: Push the code changes, updated DVC metadata, and any relevant configuration.
3.  **GitHub Actions**: The CI pipeline automatically runs, but it **does not re-train**. Instead, it validates that the new code works and that a model artifact (even a mock one) can be loaded and served correctly.

This gives us real CI/CD practice without depending on expensive, GPU-powered runners.

## 5. Keep DVC for Reproducibility, Not Heavy CI

DVC remains a valuable tool for tracking experiments and ensuring reproducibility. However, we will avoid running `dvc repro` in the main CI pipeline if it triggers a full training run.

**How we will use DVC:**
- Track the dataset samples used for local training.
- Track the tokenized or prepared datasets.
- Track the final model/adapter artifacts.
- Keep `params.yaml` for defining hyperparameters like the base model name, epochs, batch size, and max sequence length.

In the GitHub PR CI, we will either run a mock/lightweight DVC stage or pull a pre-produced small artifact to test the application.

## 6. Start with One Tiny LLM Use Case

Instead of complex tasks like RAG or building a chat assistant, we will begin with a simple, well-defined task like **prompt completion**.

- **Input**: `{"prompt": "Write one sentence about cloud engineering."}`
- **Output**: `{"generated_text": "..."}`

This allows us to refactor the current `/predict` endpoint from handling numeric features to handling text with minimal disruption.

## 7. Change the API Contract Early

The API contract defined in `app/schemas.py` will be updated to reflect the new LLM-centric approach.

**New `PredictRequest` schema:**
- `prompt` (string)
- `max_new_tokens` (integer)
- `temperature` (float, optional)

**New `PredictResponse` schema:**
- `generated_text` (string)
- `model_version` (string)

The CI smoke test will then only need to verify that the API returns a `200 OK` response with a non-empty `generated_text` field.

## 8. Recommended Stack for a Constrained Environment

Given the constraints of a laptop and free GitHub Actions runners, the recommended stack is:

- **Model**: Start with `distilgpt2`, then move to a small LoRA-tuned model.
- **Training Approach**: Inference-first, with LoRA-based fine-tuning performed manually or locally.
- **Serving**: FastAPI.
- **CI/CD**: GitHub Actions.
- **Artifacts**: DVC-tracked outputs stored in a cloud bucket (e.g., S3, GCS).
- **Experiment Tracking**: MLflow (already integrated).

This stack is powerful enough to demonstrate a complete MLOps cycle while remaining manageable and cost-effective.

---

## Implemented Changes Summary

The following recommendations have been successfully implemented:

*   **1. Keep GitHub Actions for CI, but Make it Inference-First**:
    *   The `ci-cd.yaml` workflow has been updated to remove model training.
    *   It now focuses on installing dependencies, running updated tests, building the Docker image, and smoke-testing the FastAPI application using a tiny LLM (`sshleifer/tiny-gpt2`).
*   **2. Do Not Retrain in Every PR**:
    *   The `ci-cd.yaml` no longer performs training.
    *   A new, separate workflow `train-llm.yml` has been created for manual fine-tuning, triggered via `workflow_dispatch`.
*   **3. Fine-Tune with LoRA, Not Full-Model Training**:
    *   The `train-llm.yml` workflow includes placeholders and dependency installations (`peft`, `accelerate`, `bitsandbytes`) to support LoRA-based fine-tuning.
*   **4. Train Locally, Validate in GitHub**:
    *   This strategy is now enabled by the separation of CI and training workflows.
*   **5. Keep DVC for Reproducibility, Not Heavy CI**:
    *   The `ci-cd.yaml` no longer runs `dvc repro` for training.
    *   `train-llm.yml` includes DVC commands to add and push trained model artifacts.
*   **6. Start with One Tiny LLM Use Case**:
    *   The `app/main.py` and `app/schemas.py` have been updated to support text prompt completion using an LLM.
*   **7. Change the API Contract Early**:
    *   `app/schemas.py` has been updated to define `PredictRequest` with `prompt`, `max_new_tokens`, and `temperature`, and `PredictResponse` with `generated_text` and `model_version`.
    *   The `test_app.py` has been updated to reflect this new API contract.
*   **8. Recommended Stack for a Constrained Environment**:
    *   The core components (FastAPI, Hugging Face Transformers, GitHub Actions, DVC) are now integrated as described.
    *   `requirements.txt` has been updated to include `transformers` and `torch`, and remove `scikit-learn` and `joblib`.
    *   Outdated test files (`test_model.py`, `test_dvc.py`, `test_drift.py`) have been removed.
