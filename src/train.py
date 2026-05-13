import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model"
MODEL_CLASSES = {
    "causal-lm": AutoModelForCausalLM,
    "seq2seq-lm": AutoModelForSeq2SeqLM,
    "masked-lm": AutoModelForMaskedLM,
}
MODEL_WEIGHT_PATTERNS = ("*.bin", "*.safetensors", "*.h5")
TOKENIZER_FILE_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
    "merges.txt",
}


class LLMPipelineError(RuntimeError):
    """Raised when the LLM artifact pipeline cannot produce a valid artifact."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a configurable LLM artifact for the MLOps pipeline."
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--model-task",
        choices=sorted(MODEL_CLASSES),
        default=os.getenv("LLM_MODEL_TASK", "causal-lm"),
        help="Model head to load for the artifact.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.getenv("LLM_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)),
        help="Directory where model and tokenizer files will be saved.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("LLM_DATASET_PATH", ""),
        help="Optional dataset path recorded for lineage.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=float(os.getenv("LLM_EPOCHS", "0")),
        help="Training epochs recorded for lineage. This script does not fine-tune by default.",
    )
    parser.add_argument(
        "--learning-rate",
        default=os.getenv("LLM_LEARNING_RATE", ""),
        help="Learning rate recorded for lineage.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.getenv("LLM_TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"},
        help="Allow custom remote model code when loading from Hugging Face.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove an existing artifact directory before saving.",
    )
    return parser.parse_args(argv)


def write_pipeline_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "model_name": args.model_name,
        "model_task": args.model_task,
        "dataset_path": args.dataset_path or None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate or None,
        "artifact_dir": str(args.artifact_dir),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    manifest_path = args.artifact_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_model_and_tokenizer(args: argparse.Namespace):
    model_class = MODEL_CLASSES[args.model_task]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
        )
        model = model_class.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        raise LLMPipelineError(
            f"Failed to load model or tokenizer for '{args.model_name}' "
            f"with task '{args.model_task}': {exc}"
        ) from exc

    return tokenizer, model


def save_model_artifact(tokenizer, model, artifact_dir: Path) -> None:
    try:
        tokenizer.save_pretrained(artifact_dir)
        model.save_pretrained(artifact_dir)
    except Exception as exc:
        raise LLMPipelineError(
            f"Failed to save model artifact to '{artifact_dir}': {exc}"
        ) from exc


def validate_artifact(artifact_dir: Path) -> None:
    missing: list[str] = []

    if not artifact_dir.is_dir():
        raise LLMPipelineError(f"Artifact directory was not created: {artifact_dir}")

    if not (artifact_dir / "config.json").is_file():
        missing.append("config.json")

    if not any(artifact_dir.glob(pattern) for pattern in MODEL_WEIGHT_PATTERNS):
        missing.append("model weight file (*.bin, *.safetensors, or *.h5)")

    if not any((artifact_dir / file_name).is_file() for file_name in TOKENIZER_FILE_NAMES):
        missing.append("tokenizer file")

    if not (artifact_dir / "pipeline_manifest.json").is_file():
        missing.append("pipeline_manifest.json")

    if missing:
        raise LLMPipelineError(
            f"Artifact validation failed for '{artifact_dir}'. Missing: {', '.join(missing)}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Running LLM pipeline script from {__file__}")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.model_task}")
    print(f"Artifact directory: {args.artifact_dir}")

    if args.artifact_dir.exists() and not args.keep_existing:
        print(f"Cleaning existing model directory: {args.artifact_dir}")
        shutil.rmtree(args.artifact_dir)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    if not mlflow.get_tracking_uri() or "databricks" in mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run():
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("model_task", args.model_task)
        if args.dataset_path:
            mlflow.log_param("dataset_path", args.dataset_path)
        if args.epochs:
            mlflow.log_param("epochs", args.epochs)
        if args.learning_rate:
            mlflow.log_param("learning_rate", args.learning_rate)

        tokenizer, model = load_model_and_tokenizer(args)

        save_model_artifact(tokenizer, model, args.artifact_dir)
        write_pipeline_manifest(args)
        validate_artifact(args.artifact_dir)

        mlflow.log_artifacts(str(args.artifact_dir), artifact_path="model")
        print(f"Saved and logged LLM artifact to {args.artifact_dir}")


if __name__ == "__main__":
    try:
        main()
    except LLMPipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
