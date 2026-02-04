import os
import shutil
import mlflow
from pathlib import Path
from src.train import main, MODEL_PATH

def test_training_script():
    """
    Test the training script execution.
    Since src.train now uses MLflow, we need to configure a temporary tracking URI
    to avoid polluting the global state or failing if the directory doesn't exist.
    """
    # Setup: Use a temporary directory for MLflow runs
    mlruns_dir = "mlruns_test_model"
    if os.path.exists(mlruns_dir):
        shutil.rmtree(mlruns_dir)
    
    # Set tracking URI to local file system
    mlflow.set_tracking_uri(f"file:{os.path.abspath(mlruns_dir)}")
    
    # Ensure the directory exists
    os.makedirs(mlruns_dir, exist_ok=True)
    
    # Set the experiment to "Default" explicitly.
    # This creates it if it doesn't exist and sets it as active.
    mlflow.set_experiment("Default")

    # Clean up any existing model files
    if MODEL_PATH.exists():
        os.remove(MODEL_PATH)
    
    try:
        # Run the training function
        main()
        
        # Check that model was created
        assert MODEL_PATH.exists()
        assert MODEL_PATH.stat().st_size > 0
        
    finally:
        # Cleanup
        if os.path.exists(mlruns_dir):
            shutil.rmtree(mlruns_dir)
        # Reset tracking URI to default to avoid side effects on other tests
        mlflow.set_tracking_uri("")
