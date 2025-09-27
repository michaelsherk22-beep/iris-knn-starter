from pathlib import Path
import subprocess
import sys


def test_training_runs(tmp_path, monkeypatch):
    # Run in a temp directory to avoid polluting workspace
    monkeypatch.chdir(tmp_path)
    # Train and produce a model artifact
    subprocess.check_call([sys.executable, "-m", "iris_knn.train"])
    assert Path("model.joblib").exists()
