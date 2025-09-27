from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
MODEL_PATH = Path("model.joblib")


def main() -> None:
    # Load data (features X, labels y)
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    # Train/test split for honest evaluation on unseen data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Pipeline: Standardize features -> k-NN classifier (k=5)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
    )

    # Fit on train set
    pipe.fit(X_tr, y_tr)

    # Evaluate on test set
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(f"Trained k-NN. Test accuracy: {acc:.3f}")

    # Save model so we can reuse it
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
