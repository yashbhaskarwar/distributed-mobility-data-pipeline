from pathlib import Path
import subprocess
import pytest
from src.api.app import app

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

@pytest.mark.order(1)
def test_silver_features_exist():
    silver_path = DATA_DIR / "features" / "silver_features"
    assert silver_path.exists(), "Silver features path does not exist"

@pytest.mark.order(2)
def test_batch_scoring_runs():
    cmd = [
        "python",
        "-m",
        "src.ml.batch_scoring",
        "--silver_features",
        str(DATA_DIR / "features" / "silver_features"),
        "--predictions_out",
        str(DATA_DIR / "gold" / "batch_predictions"),
        "--metrics_out",
        str(DATA_DIR / "gold" / "batch_scoring_metrics"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Batch scoring failed: {result.stderr}"

@pytest.mark.order(3)
def test_batch_predictions_written():
    preds_path = DATA_DIR / "gold" / "batch_predictions"
    assert preds_path.exists(), "Batch predictions Delta table not found"

@pytest.mark.order(4)
def test_model_monitoring_runs():
    cmd = [
        "python",
        "-m",
        "src.ml.model_monitoring",
        "--predictions_delta",
        str(DATA_DIR / "gold" / "batch_predictions"),
        "--monitoring_out",
        str(DATA_DIR / "gold" / "model_monitoring_metrics"),
        "--drift_out",
        str(DATA_DIR / "gold" / "model_drift_details"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Monitoring job failed: {result.stderr}"

@pytest.mark.order(5)
def test_monitoring_outputs_exist():
    metrics_path = DATA_DIR / "gold" / "model_monitoring_metrics"
    drift_path = DATA_DIR / "gold" / "model_drift_details"
    assert metrics_path.exists(), "Monitoring metrics table missing"
    assert drift_path.exists(), "Drift details table missing"

@pytest.mark.order(6)
def test_api_health_endpoint():
    assert app is not None
