from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Define paths relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
REFERENCE_DATA_PATH = ARTIFACT_DIR / "reference_dataset.csv"

def main() -> None:
    print(f"Running training script from {__file__}")
    print(f"Artifact directory: {ARTIFACT_DIR}")
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Set tracking URI to a local directory to avoid issues in CI/CD
    # if not already set.
    if not mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run():
        X, y = load_iris(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save reference dataset for drift detection
        print(f"Saving reference dataset to {REFERENCE_DATA_PATH}")
        X_train.to_csv(REFERENCE_DATA_PATH, index=False)
        
        if REFERENCE_DATA_PATH.exists():
             print(f"Reference dataset successfully saved to {REFERENCE_DATA_PATH}")
        else:
             print(f"Failed to save reference dataset to {REFERENCE_DATA_PATH}")

        # Hyperparameters
        params = {
            "max_iter": 200,
            "solver": "lbfgs",
            "random_state": 42
        }
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save model locally for the API to use
        joblib.dump(model, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")
        
        # Log model to MLflow with signature and input example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)
        
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=input_example
        )

if __name__ == "__main__":
    main()
