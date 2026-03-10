from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
import os

# Define paths relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "DialoGPT-small"

def main() -> None:
    """
    Downloads the DialoGPT-small model and tokenizer from Hugging Face,
    saves them to the artifacts directory, and logs them to MLflow.
    """
    print(f"Running training script from {__file__}")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Set tracking URI to a local directory if not already set for manual runs.
    # The test will set its own tracking URI.
    if not mlflow.get_tracking_uri() or "databricks" in mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run():
        model_name = "microsoft/DialoGPT-small"
        mlflow.log_param("model_name", model_name)

        print(f"Downloading model and tokenizer for '{model_name}'...")

        # Download and save the model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(MODEL_PATH)

        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(MODEL_PATH)

        print(f"Model and tokenizer saved to {MODEL_PATH}")

        # Log the model artifacts to MLflow
        mlflow.log_artifacts(str(MODEL_PATH), artifact_path="model")
        print("Logged model artifacts to MLflow.")

if __name__ == "__main__":
    main()
