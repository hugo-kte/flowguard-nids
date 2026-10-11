from __future__ import annotations

import json

import numpy as np
import pytest

from flowguard.features.schema import BENIGN, FEATURES
from flowguard.models.anomaly import AutoencoderDetector, IsolationForestDetector
from flowguard.models.preprocessing import signed_log1p
from flowguard.models.registry import MANIFEST, ModelBundle
from flowguard.training import calibrate_threshold, load_splits, xy


def test_signed_log_is_symmetric_and_monotonic():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = signed_log1p(x)
    assert np.allclose(y, -y[::-1]) and np.all(np.diff(y) > 0)


def test_threshold_calibration_respects_budget():
    scores = np.random.default_rng(0).normal(size=100_000)
    thr = calibrate_threshold(scores, 0.01)
    assert np.mean(scores > thr) <= 0.01


@pytest.mark.parametrize("cls", [AutoencoderDetector, IsolationForestDetector])
def test_anomaly_detectors_rank_outliers_higher(cls, synthetic_flows):
    benign = synthetic_flows[synthetic_flows["family"] == BENIGN][FEATURES].to_numpy(np.float32)
    params = {"max_iter": 30} if cls is AutoencoderDetector else {"n_estimators": 50}
    det = cls(**params).fit(benign)
    outliers = benign[:50] * 1e3  # wildly out-of-distribution flows
    assert np.median(det.score_samples(outliers)) > np.quantile(det.score_samples(benign), 0.99)


def test_bundle_contents_and_thresholds(trained_config):
    bundle = ModelBundle.load(trained_config.models_dir)
    assert set(bundle.model_names) == {
        "random_forest",
        "hist_gradient_boosting",
        "isolation_forest",
        "autoencoder",
    }
    assert bundle.features == FEATURES and BENIGN in bundle.classes
    val = load_splits(trained_config, ("val",)).val
    X, fam = xy(val)
    det = trained_config.detection
    for name in bundle.model_names:
        budget = det.target_fpr if name in bundle.supervised else det.anomaly_fpr
        fpr = np.mean(bundle.score(name, X[fam == BENIGN]) > bundle.thresholds[name])
        assert fpr <= budget + 1e-9


def test_supervised_model_detects_synthetic_attacks(trained_config):
    bundle = ModelBundle.load(trained_config.models_dir, models=["hist_gradient_boosting"])
    test = load_splits(trained_config, ("test",)).test
    X, fam = xy(test)
    alerts = bundle.score("hist_gradient_boosting", X) > bundle.thresholds["hist_gradient_boosting"]
    assert alerts[fam != BENIGN].mean() > 0.9


def test_tampered_model_is_refused(trained_config, tmp_path):
    import shutil

    copy = tmp_path / "models"
    shutil.copytree(trained_config.models_dir, copy)
    with (copy / "random_forest.joblib").open("ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ModelBundle.load(copy)
    # loading only the other models still works, and metadata-only loads unpickle nothing
    assert "random_forest" not in ModelBundle.load(copy, models=["autoencoder"]).anomaly
    manifest = json.loads((copy / MANIFEST).read_text())
    assert manifest["artifacts"]["random_forest"]["sha256"]
