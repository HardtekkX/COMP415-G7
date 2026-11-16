import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "training" / "artifacts"


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


def test_metrics_hold_no_model_parameters():
    metrics = load("metrics.json")
    assert not {"weights", "bias", "scaler_mean", "scaler_scale"} & set(metrics)
    assert {"test_accuracy", "test_auc", "baseline_accuracy", "n_train", "n_test", "seed"} <= set(metrics)


def test_metrics_features_match_preprocessing():
    assert load("metrics.json")["features"] == load("preprocessing.json")["feature_order"]
