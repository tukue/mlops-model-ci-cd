import os
from pathlib import Path
from src.train import main, MODEL_PATH
from src.model_registry import ModelRegistry

def test_training_script():
    # Clean up any existing model files
    registry = ModelRegistry()
    if MODEL_PATH.exists():
        os.remove(MODEL_PATH)
    
    # Clean registry directory
    registry_path = Path("artifacts/registry")
    if registry_path.exists():
        for file in registry_path.glob("*"):
            if file.is_file():
                file.unlink()
    
    main()
    
    # Check that model was created and registered
    assert MODEL_PATH.exists()
    assert MODEL_PATH.stat().st_size > 0
    
    # Check registry has active model
    version, model = registry.get_active_model()
    assert version is not None
    assert model is not None
