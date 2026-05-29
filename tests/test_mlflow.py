import os
import shutil
import mlflow

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


class FakeModelClass:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeModel()


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

    runs = client.search_runs(experiment_ids=["0"])

    assert len(runs) > 0, "No MLflow runs found"

    last_run = runs[0]
    assert last_run.data.params["model_name"] == "example/test-llm"
    assert last_run.data.params["model_task"] == "causal-lm"
    assert last_run.data.params["dataset_path"] == "data/train.jsonl"

    artifacts = client.list_artifacts(last_run.info.run_id, "model")
    assert len(artifacts) > 0, "Model artifacts not logged"

    # Cleanup
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
