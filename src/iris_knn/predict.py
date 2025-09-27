from __future__ import annotations
import sys
import numpy as np
import joblib
from sklearn.datasets import load_iris

USAGE = "Usage: python -m iris_knn.predict <sepal_len> <sepal_wid> <petal_len> <petal_wid>"

def main() -> None:
    # Basic CLI parsing and validation
    if len(sys.argv) != 5:
        print(USAGE)
        sys.exit(1)
    try:
        features = np.array([[float(x) for x in sys.argv[1:5]]])
    except ValueError:
        print("All 4 inputs must be numeric.\n" + USAGE)
        sys.exit(1)

    # Load the saved model and predict a class index
    model = joblib.load("model.joblib")
    pred_idx = int(model.predict(features)[0])

    # Map index -> human-readable species name
    names = load_iris().target_names
    print("Predicted species:", names[pred_idx])

if __name__ == "__main__":
    main()
