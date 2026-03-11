from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
import os

# Define paths relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
# Update path for the new model
MODEL_PATH = ARTIFACT_DIR / "Qwen2.5-0.5B-Instruct"

def main() -> None:
    """
    Downloads the Qwen/Qwen2.5-0.5B-Instruct model and tokenizer from Hugging Face,
    saves them to the artifacts directory, and logs them to MLflow.
    """
    print(f"Running training script from {__file__}")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Set tracking URI to a local directory if not already set for manual runs.
    if not mlflow.get_tracking_uri() or "databricks" in mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run():
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        mlflow.log_param("model_name", model_name)

        print(f"Downloading model and tokenizer for '{model_name}'...")

        # Download and save the model
        # trust_remote_code might be needed for some Qwen versions, but Qwen2.5 is usually standard.
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(MODEL_PATH)

        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(MODEL_PATH)

        print(f"Model and tokenizer saved to {MODEL_PATH}")

        # Log the model artifacts to MLflow
        mlflow.log_artifacts(str(MODEL_PATH), artifact_path="model")
        print("Logged model artifacts to MLflow.")

if __name__ == "__main__":
    main()
