import os
import shutil
import mlflow
from src.train import main


def test_mlflow_tracking():
    """
    Test that the training script successfully logs parameters and artifacts to MLflow.
    """
    # Setup: Use a temporary directory for MLflow runs
    mlruns_dir = "mlruns_test"
    # Ensure the directory is clean before the test
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    # Run training
    main()

    # Verify: Check if an experiment was created and runs exist
    client = mlflow.tracking.MlflowClient()

    # The default experiment is '0' if not otherwise specified
    runs = client.search_runs(experiment_ids=["0"])

    assert len(runs) > 0, "No MLflow runs found"

    last_run = runs[0]

    # Check for logged parameter
    assert "model_name" in last_run.data.params, "model_name parameter not logged"
    assert last_run.data.params["model_name"] == "microsoft/DialoGPT-small"

    # Check for logged artifacts
    artifacts = client.list_artifacts(last_run.info.run_id, "model")
    assert len(artifacts) > 0, "Model artifacts not logged"

    # Cleanup
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
