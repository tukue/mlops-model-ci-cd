from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_iris(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    
    # Verify the model was saved correctly
    if MODEL_PATH.exists():
        print(f"Model artifact verified at {MODEL_PATH}")
    else:
        raise RuntimeError(f"Failed to create model artifact at {MODEL_PATH}")

if __name__ == "__main__":
    main()