import os
import json
from pathlib import Path
from src.detect_drift import detect_drift, DRIFT_REPORT_PATH
from src.train import main as train_main

def test_drift_detection_script():
    """
    Test that the drift detection script runs and produces a report.
    """
    # 1. Ensure reference data exists by running the training script
    train_main()
    
    # 2. Run the drift detection script
    detect_drift()
    
    # 3. Verify that the report was created
    assert DRIFT_REPORT_PATH.exists(), "Drift report was not created."
    assert DRIFT_REPORT_PATH.stat().st_size > 0, "Drift report is empty."
    
    # 4. Verify content
    with open(DRIFT_REPORT_PATH, 'r') as f:
        report = json.load(f)
        assert "drift_detected" in report
        assert "metrics" in report
    
    # Cleanup
    if DRIFT_REPORT_PATH.exists():
        os.remove(DRIFT_REPORT_PATH)
