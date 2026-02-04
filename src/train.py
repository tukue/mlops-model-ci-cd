from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define paths relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

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
        input_example = X_train[:5]
        
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=input_example
        )

if __name__ == "__main__":
    main()