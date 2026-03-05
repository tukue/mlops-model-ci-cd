# LLM Migration Plan: Adapting for Open-Source Hugging Face LLMs

This document outlines the necessary changes to adapt the current MLOps repository for training and deploying open-source Hugging Face Large Language Models (LLMs).

## 1. Feasibility Analysis (Open-Source Tooling)

A primary consideration is whether this migration is feasible using only open-source tools.

**Conclusion: Highly Feasible, with caveats on hardware.**

The open-source ecosystem for MLOps with LLMs is mature and robust. We can achieve a full end-to-end pipeline without relying on proprietary software. The main challenge is not the availability of tools, but the management of computational resources (GPU, RAM).

### Open-Source Toolchain Breakdown:

| Stage | Task | Open-Source Tools | Feasibility Notes |
|---|---|---|---|
| **Data Management** | Versioning large text datasets | `DVC`, `Git LFS` | Excellent. DVC is already in use. Git LFS is a good alternative for very large files. |
| | Data Processing/Tokenization | `Hugging Face datasets`, `Pandas` | Excellent. `datasets` is highly optimized for large text corpora and memory-mapping. |
| **Experiment Tracking** | Logging metrics, params, artifacts | `MLflow` | Excellent. MLflow is already in use and integrates well with `transformers`. |
| **Model Training** | Fine-tuning LLMs | `Hugging Face transformers`, `PEFT`, `accelerate`, `PyTorch` | Excellent. This stack is the de-facto standard for open-source LLM training. |
| | Distributed Training | `DeepSpeed` (via `accelerate`), `FSDP` | Good. Requires more complex setup but is well-documented. |
| **CI/CD** | Pipeline Automation | `GitHub Actions` / `GitLab CI` | Excellent. Already in use. The main challenge is getting access to GPU runners if needed for CI-based training/testing. |
| **Model Serving** | API for inference | `FastAPI`, `vLLM`, `TGI` | Good. FastAPI is a solid baseline. For high-throughput/low-latency, `vLLM` or Hugging Face's `text-generation-inference` (TGI) are state-of-the-art. |
| **Model Monitoring** | Drift, performance | `Prometheus`, `Grafana`, `Evidently AI` | Good. Prometheus is already in use. `Evidently AI` provides excellent open-source support for data and model drift detection. |

### Technical Feasibility on a Laptop

Developing an LLM MLOps pipeline on a standard laptop is **feasible for code development and pipeline orchestration, but not for full-scale training or production-grade serving.**

**What IS Feasible on a Laptop:**
*   **Pipeline Development**: You can write and test the entire MLOps workflow code:
    *   Data processing scripts (using sampled data).
    *   `train.py` script logic.
    *   `app/main.py` for the serving API.
    *   `dvc.yaml` stages.
    *   `.github/workflows/ci-cd.yaml` for automation.
*   **Unit & Integration Testing**: Write `pytest` tests for all components. You can use mock objects or a "dummy" model in place of a real LLM to test the logic.
*   **Training on Tiny Models**: It is possible to fine-tune very small LLMs (e.g., `distilbert`, `TinyLlama`) on a CPU or a laptop GPU (if available) to verify the training script works.
*   **Inference with Small Models**: Running inference with these same tiny models in the FastAPI application is also feasible.

**What is NOT Feasible on a Laptop:**
*   **Training Large Models**: Fine-tuning models like Llama 7B, even with QLoRA, is practically impossible on a typical laptop. It requires a dedicated GPU with significant VRAM (e.g., 16-24GB+) and will be extremely slow without it.
*   **Serving Large Models**: Running inference for a 7B model requires a large amount of RAM and CPU, leading to very slow response times that are not suitable for a real application.
*   **Meaningful Evaluation**: Proper evaluation requires running the model on a substantial test set, which is computationally expensive.

**Strategy**: The recommended approach is to use the laptop for developing and debugging the pipeline with small, sampled data and mock/tiny models. The full pipeline can then be executed on a cloud-based GPU instance for the actual training, evaluation, and deployment.

---

## 2. Phase 1: Lightweight LLM Pilot

To validate the MLOps pipeline without heavy resource requirements, we will start with a "Lightweight LLM Pilot".

**Goal**: Establish a working end-to-end pipeline (Data -> Train -> Track -> Serve) using a small, open-source LLM that can run on a standard laptop or free-tier cloud instance (e.g., Google Colab).

**Selected Model Candidates:**
*   **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**: A 1.1B parameter model. Small enough for experimentation but capable of chat.
*   **`facebook/opt-125m`**: A very small (125M parameter) model. Excellent for testing pipeline mechanics (training loop, logging, saving) even if the output quality is low.
*   **`distilgpt2`**: A distilled version of GPT-2 (82M parameters). Very fast and lightweight.

**Pilot Workflow:**
1.  **Data**: Use a small subset (e.g., 1000 samples) of a public dataset like `imdb` (sentiment analysis) or `databricks/databricks-dolly-15k` (instruction tuning).
2.  **Training**:
    *   Fine-tune the selected model for a specific task (e.g., causal language modeling or sequence classification).
    *   Use `LoRA` (Low-Rank Adaptation) to further reduce memory usage.
    *   Run for a small number of epochs (e.g., 1-3) to verify the process completes.
3.  **Tracking**: Ensure loss curves, learning rate, and system metrics are logged to MLflow.
4.  **Serving**: Deploy the fine-tuned model using FastAPI. Verify it can accept a text prompt and return a text response.

**Success Criteria:**
*   `train.py` runs successfully without OOM (Out of Memory) errors.
*   Model artifacts (adapter weights, config) are saved to the `artifacts/` directory.
*   MLflow experiment shows a completed run with metrics.
*   The API endpoint `/predict` returns a valid JSON response with generated text.

---

## 3. Dependency Updates (`requirements.txt`)

The existing `requirements.txt` is tailored for traditional ML models. We need to add libraries essential for Hugging Face LLMs and deep learning.

**Proposed Changes:**
*   Add `transformers`: Core Hugging Face library for models and tokenizers.
*   Add `datasets`: For efficient handling and loading of large text datasets.
*   Add `accelerate`: For simplified distributed training and mixed-precision training.
*   Add `peft`: Parameter-Efficient Fine-tuning (e.g., LoRA, QLoRA) for efficient LLM fine-tuning.
*   Add `bitsandbytes`: For 8-bit quantization, enabling training of larger models on consumer GPUs.
*   Add `torch`: The underlying deep learning framework.
*   Add `vllm` (Optional, for high-performance serving).
*   Add `evidently` (For LLM monitoring).

## 4. Data Preparation Workflow

The current DVC-based data pipeline needs to be adapted for text data and LLM-specific preprocessing.

**Proposed Changes:**
*   **Data Ingestion**: Update scripts to ingest large text corpora.
*   **Text Preprocessing**:
    *   Tokenization using Hugging Face `AutoTokenizer`.
    *   Handling of special tokens (padding, EOS, BOS).
    *   Creation of input IDs, attention masks, and labels.
*   **Dataset Creation**: Utilize `datasets` library for efficient loading and batching.
*   **DVC Integration**: Ensure DVC tracks raw text data, preprocessed datasets, and tokenizer configurations.

## 5. Model Training Script (`train.py`)

The `train.py` script will require significant modifications to accommodate LLM fine-tuning.

**Proposed Changes:**
*   **Model Loading**: Load pre-trained LLMs using `AutoModelForCausalLM` or `AutoModelForSequenceClassification` (depending on task) from `transformers`.
*   **Tokenizer Loading**: Load corresponding tokenizer using `AutoTokenizer`.
*   **Fine-tuning Strategy**:
    *   Implement LoRA/QLoRA using `peft` for efficient fine-tuning.
    *   Configure `TrainingArguments` and `Trainer` from `transformers` for training loop management.
*   **Distributed Training**: Leverage `accelerate` for multi-GPU or multi-node training.
*   **Hyperparameter Management**: Adapt for LLM-specific hyperparameters (e.g., learning rate schedulers, gradient accumulation steps).
*   **MLflow Tracking**:
    *   Log LLM-specific metrics (e.g., perplexity, ROUGE, BLEU).
    *   Log model artifacts (model weights, tokenizer, `peft` adapters).

## 6. Model Evaluation

Evaluation metrics and procedures need to be updated for LLMs.

**Proposed Changes:**
*   **Quantitative Metrics**: Implement metrics like perplexity, ROUGE, BLEU, or task-specific metrics (e.g., F1 for classification, exact match for QA).
*   **Human Evaluation**: Consider integrating a workflow for human evaluation, especially for generative tasks.
*   **MLflow Logging**: Log all evaluation results and potentially example generations.

## 7. Model Serving (`app/main.py`, `schemas.py`, `Dockerfile`)

The serving component needs to handle LLM inference, which can be resource-intensive.

**Proposed Changes:**
*   **Model Loading**: Load the fine-tuned LLM and tokenizer. If using PEFT, load the base model and then attach the PEFT adapters.
*   **Inference Endpoint**:
    *   Create a FastAPI endpoint for text generation or other LLM tasks.
    *   Handle input prompts and output generations.
*   **Optimization**:
    *   Implement efficient inference techniques (e.g., `torch.bfloat16`, `torch.compile`, quantization).
    *   Consider using specialized serving frameworks like vLLM or TGI if performance is critical.
*   **`schemas.py`**: Update Pydantic schemas for LLM inputs (e.g., `prompt`, `max_new_tokens`, `temperature`) and outputs (e.g., `generated_text`).
*   **`Dockerfile`**: Ensure the Dockerfile includes all necessary dependencies (GPU drivers if applicable), and optimizes for LLM serving (e.g., larger memory limits).

## 8. CI/CD Pipeline (`.github/workflows/`, `dvc.yaml`)

The CI/CD pipeline needs to be updated to reflect the new training and deployment workflow.

**Proposed Changes:**
*   **`dvc.yaml`**: Update DVC stages for LLM data preprocessing, training, and model packaging.
*   **GitHub Actions/GitLab CI/CD**:
    *   Add steps for installing LLM-specific dependencies.
    *   Configure jobs to run on GPU-enabled runners if training in CI/CD.
    *   Update deployment steps to push LLM models to a model registry (e.g., MLflow Model Registry, Hugging Face Hub).
    *   Integrate model monitoring checks.

## 9. Infrastructure Considerations

LLMs require substantial computational resources.

**Proposed Changes:**
*   **GPU Resources**: Ensure access to powerful GPUs for training and efficient inference.
*   **Memory**: Plan for increased memory requirements for both training and serving.
*   **Distributed Systems**: Consider Kubernetes or other orchestration tools for scalable deployment.

## Next Steps

1.  **Create a new branch**: `git checkout -b feature/llm-migration`
2.  **Update `requirements.txt`**: Add the necessary LLM libraries.
3.  **Start with data preparation**: Adapt existing data scripts or create new ones for text data.
4.  **Develop a basic fine-tuning script**: Get a simple LLM fine-tuning process working.
5.  **Iteratively integrate**: Gradually incorporate these changes into the existing MLOps pipeline.
