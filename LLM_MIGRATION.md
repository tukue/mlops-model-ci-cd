# LLM Migration Plan: Adapting for Open-Source Hugging Face LLMs

This document outlines the necessary changes to adapt the current MLOps repository for training and deploying open-source Hugging Face Large Language Models (LLMs).

## 1. Dependency Updates (`requirements.txt`)

The existing `requirements.txt` is tailored for traditional ML models. We need to add libraries essential for Hugging Face LLMs and deep learning.

**Proposed Changes:**
*   Add `transformers`: Core Hugging Face library for models and tokenizers.
*   Add `datasets`: For efficient handling and loading of large text datasets.
*   Add `accelerate`: For simplified distributed training and mixed-precision training.
*   Add `peft`: Parameter-Efficient Fine-tuning (e.g., LoRA, QLoRA) for efficient LLM fine-tuning.
*   Add `bitsandbytes`: For 8-bit quantization, enabling training of larger models on consumer GPUs.
*   Add `torch`: The underlying deep learning framework.

## 2. Data Preparation Workflow

The current DVC-based data pipeline needs to be adapted for text data and LLM-specific preprocessing.

**Proposed Changes:**
*   **Data Ingestion**: Update scripts to ingest large text corpora.
*   **Text Preprocessing**:
    *   Tokenization using Hugging Face `AutoTokenizer`.
    *   Handling of special tokens (padding, EOS, BOS).
    *   Creation of input IDs, attention masks, and labels.
*   **Dataset Creation**: Utilize `datasets` library for efficient loading and batching.
*   **DVC Integration**: Ensure DVC tracks raw text data, preprocessed datasets, and tokenizer configurations.

## 3. Model Training Script (`train.py`)

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

## 4. Model Evaluation

Evaluation metrics and procedures need to be updated for LLMs.

**Proposed Changes:**
*   **Quantitative Metrics**: Implement metrics like perplexity, ROUGE, BLEU, or task-specific metrics (e.g., F1 for classification, exact match for QA).
*   **Human Evaluation**: Consider integrating a workflow for human evaluation, especially for generative tasks.
*   **MLflow Logging**: Log all evaluation results and potentially example generations.

## 5. Model Serving (`app/main.py`, `schemas.py`, `Dockerfile`)

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

## 6. CI/CD Pipeline (`.github/workflows/`, `dvc.yaml`)

The CI/CD pipeline needs to be updated to reflect the new training and deployment workflow.

**Proposed Changes:**
*   **`dvc.yaml`**: Update DVC stages for LLM data preprocessing, training, and model packaging.
*   **GitHub Actions/GitLab CI/CD**:
    *   Add steps for installing LLM-specific dependencies.
    *   Configure jobs to run on GPU-enabled runners if training in CI/CD.
    *   Update deployment steps to push LLM models to a model registry (e.g., MLflow Model Registry, Hugging Face Hub).
    *   Integrate model monitoring checks.

## 7. Infrastructure Considerations

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
