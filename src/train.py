from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define paths relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "DialoGPT-small"

def main() -> None:
    """
    Downloads the DialoGPT-small model and tokenizer from Hugging Face
    and saves them to the artifacts directory.
    """
    print(f"Running training script from {__file__}")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "microsoft/DialoGPT-small"

    print(f"Downloading model and tokenizer for '{model_name}'...")

    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(MODEL_PATH)

    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODEL_PATH)

    print(f"Model and tokenizer saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
