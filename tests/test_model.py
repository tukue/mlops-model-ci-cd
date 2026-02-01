import os
from pathlib import Path
from src.train import main, MODEL_PATH

def test_training_script():
    # Ensure we can run the training script and it produces a file
    if MODEL_PATH.exists():
        os.remove(MODEL_PATH)
    
    main()
    
    assert MODEL_PATH.exists()
    assert MODEL_PATH.stat().st_size > 0
