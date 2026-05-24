import os
import shutil
from argparse import Namespace

import mlflow
import pytest

from src import train


class FakeTokenizer:
    def save_pretrained(self, artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "tokenizer.json"), "w", encoding="utf-8") as file:
            file.write("{}\n")


class FakeModel:
    def save_pretrained(self, artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "config.json"), "w", encoding="utf-8") as file:
            file.write("{}\n")
        with open(os.path.join(artifact_dir, "model.safetensors"), "w", encoding="utf-8") as file:
            file.write("fake weights\n")


def test_mlflow_tracking(monkeypatch, tmp_path):
    """
    Test that the generic LLM pipeline logs parameters and artifacts to MLflow.
    """
    monkeypatch.setattr(
        train.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setitem(train.MODEL_CLASSES, "causal-lm", FakeModelClass)

    # Setup: Use a temporary directory for MLflow runs
    mlruns_dir = "mlruns_test"
    # Ensure the directory is clean before the test
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    # Run training
    artifact_dir = tmp_path / "llm-artifact"
    train.main(
        [
            "--model-name",
            "example/test-llm",
            "--model-task",
            "causal-lm",
            "--artifact-dir",
            str(artifact_dir),
            "--dataset-path",
            "data/train.jsonl",
            "--epochs",
            "1",
            "--learning-rate",
            "2e-5",
        ]
    )

    # Verify: Check if an experiment was created and runs exist
    client = mlflow.tracking.MlflowClient()

    # The default experiment is '0' if not otherwise specified
    runs = client.search_runs(experiment_ids=["0"])

    assert len(runs) > 0, "No MLflow runs found"

    last_run = runs[0]

    # Check for logged parameter
    assert "model_name" in last_run.data.params, "model_name parameter not logged"
    assert last_run.data.params["model_name"] == "example/test-llm"
    assert last_run.data.params["model_task"] == "causal-lm"
    assert last_run.data.params["dataset_path"] == "data/train.jsonl"

    # Check for logged artifacts
    artifacts = client.list_artifacts(last_run.info.run_id, "model")
    assert len(artifacts) > 0, "Model artifacts not logged"

    # Cleanup
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)


class FakeModelClass:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeModel()


class BrokenModelClass:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise OSError("model not found")


def test_load_model_and_tokenizer_wraps_loader_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setitem(train.MODEL_CLASSES, "causal-lm", BrokenModelClass)

    args = Namespace(
        model_name="missing/model",
        model_task="causal-lm",
        artifact_dir=tmp_path,
        dataset_path="",
        epochs=0,
        learning_rate="",
        trust_remote_code=False,
    )

    with pytest.raises(train.LLMPipelineError, match="Failed to load model or tokenizer"):
        train.load_model_and_tokenizer(args)


def test_validate_artifact_rejects_missing_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "pipeline_manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(train.LLMPipelineError, match="model weight file"):
        train.validate_artifact(tmp_path)
