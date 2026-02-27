import sys
import os

# Add the current directory to sys.path so we can import src and tests
sys.path.append(os.getcwd())

from tests.test_drift import test_drift_detection_script

if __name__ == "__main__":
    try:
        test_drift_detection_script()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
