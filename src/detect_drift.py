import pandas as pd
import json
from pathlib import Path
from scipy.stats import ks_2samp

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REFERENCE_DATA_PATH = ARTIFACT_DIR / "reference_dataset.csv"
DRIFT_REPORT_PATH = ARTIFACT_DIR / "drift_report.json"

def detect_drift():
    """
    Compares a reference dataset with a 'current' dataset to detect data drift
    using the Kolmogorov-Smirnov (KS) test.
    """
    # 1. Load reference data
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_DATA_PATH}. "
            "Run 'python -m src.train' first."
        )
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)

    # 2. Simulate current data with drift
    # For this demo, we'll take a slice of the reference data and modify it
    # to ensure drift is detected. In a real system, this would be live data.
    current_data = reference_data.copy().tail(50)
    current_data["sepal length (cm)"] = current_data["sepal length (cm)"] * 1.5
    current_data["petal width (cm)"] = current_data["petal width (cm)"] + 0.2

    print("Simulating drift in 'sepal length (cm)' and 'petal width (cm)'...")

    # 3. Perform Drift Detection (KS Test)
    drift_results = {
        "drift_detected": False,
        "drifted_features": {},
        "metrics": {}
    }
    
    # Check each numerical column
    numeric_columns = reference_data.select_dtypes(include=['number']).columns
    
    for col in numeric_columns:
        # Run KS test
        # Null hypothesis: the two distributions are identical
        # If p-value < 0.05, we reject null hypothesis -> Drift Detected
        stat, p_value = ks_2samp(reference_data[col], current_data[col])
        
        # Convert numpy bool to python bool for JSON serialization
        is_drifted = bool(p_value < 0.05)
        
        drift_results["metrics"][col] = {
            "ks_stat": float(stat),
            "p_value": float(p_value),
            "drift_detected": is_drifted
        }
        
        if is_drifted:
            drift_results["drifted_features"][col] = float(p_value)
            drift_results["drift_detected"] = True

    # 4. Save the report
    with open(DRIFT_REPORT_PATH, 'w') as f:
        json.dump(drift_results, f, indent=4)
        
    print(f"Drift report saved to {DRIFT_REPORT_PATH}")
    
    if drift_results["drift_detected"]:
        print("Data drift detected!")
        print(f"Drifted features: {list(drift_results['drifted_features'].keys())}")
    else:
        print("No data drift detected.")

if __name__ == "__main__":
    detect_drift()
