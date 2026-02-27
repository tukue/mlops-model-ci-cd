import os
import shutil
import mlflow
from src.train import main


def test_mlflow_tracking():
    """
    Test that the training script successfully logs to MLflow.
    """
    # Setup: Use a temporary directory for MLflow runs
    mlruns_dir = "mlruns_test"
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    # Run training
    main()

    # Verify: Check if an experiment was created and runs exist
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")  # Default experiment

    # If main() doesn't set experiment name, it goes to '0' (Default)
    # Let's check if any run exists
    runs = client.search_runs(experiment_ids=["0"])

    assert len(runs) > 0, "No MLflow runs found"

    last_run = runs[0]
    assert "accuracy" in last_run.data.metrics, "Accuracy metric not logged"
    assert "max_iter" in last_run.data.params or "max_iter" in last_run.data.params.keys(), "Params not logged"

    # Cleanup
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
