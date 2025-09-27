from __future__ import annotations
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main() -> None:
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = joblib.load("model.joblib")
    y_pred = model.predict(X_te)
    print("Accuracy:", round(accuracy_score(y_te, y_pred), 3))
    print("\nClassification report:\n", classification_report(y_te, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()
