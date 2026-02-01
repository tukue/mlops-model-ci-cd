from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.model_registry import ModelRegistry

# Define paths relative to this file
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "test_samples": len(y_test)
    }
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save with versioning
    version = registry.save_model(model, metrics)
    print(f"Model saved as version: {version}")
    
    # Deploy if first model or accuracy > 90%
    current_version, current_model = registry.get_active_model()
    
    if current_model is None or accuracy > 0.9:
        registry.set_active_model(version)
        print(f"Model v{version} deployed as active model")
    else:
        print(f"Model v{version} saved but not deployed (accuracy too low)")

if __name__ == "__main__":
    main()