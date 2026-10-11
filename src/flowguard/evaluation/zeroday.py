"""Leave-one-family-out experiment: how well is an attack detected when it is *unseen*?

Random train/test splits flatter supervised models: every attack in the test set has
close relatives in the training set. Here each attack family is removed from training in
turn, a binary supervised model is re-fitted, calibrated to the same false-positive budget,
and its detection rate on the held-out family is compared with:

* the same family when it *was* seen in training (main supervised model),
* the anomaly detector, which never sees any attack by construction,
* the hybrid (unseen supervised OR anomaly).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from flowguard.config import Config
from flowguard.data.split import cap_families
from flowguard.features.schema import BENIGN, FAMILY_COLUMN
from flowguard.models.registry import ModelBundle
from flowguard.models.supervised import build_hist_gradient_boosting
from flowguard.training import calibrate_threshold, load_splits, xy

log = logging.getLogger(__name__)

LOFO_DEFAULTS: dict[str, Any] = {"max_iter": 200, "class_weight": "balanced"}


def run_zero_day(cfg: Config) -> dict[str, Any]:
    zcfg = cfg.zero_day
    splits = load_splits(cfg)
    sup_name, ano_name = cfg.detection.supervised_model, cfg.detection.anomaly_model
    bundle = ModelBundle.load(cfg.models_dir, models=[sup_name, ano_name])

    X_test, fam_test = xy(splits.test)
    X_val, fam_val = xy(splits.val)
    benign_val = X_val[fam_val == BENIGN]
    seen_alerts = bundle.score(sup_name, X_test) > bundle.thresholds[sup_name]
    ano_alerts = bundle.score(ano_name, X_test) > bundle.thresholds[ano_name]

    counts = splits.test[FAMILY_COLUMN].value_counts()
    families = zcfg.families or [
        f for f, n in counts.items() if f != BENIGN and n >= zcfg.min_test_samples
    ]
    train = cap_families(splits.train, cfg.split.max_train_per_family, cfg.split.seed)

    results: dict[str, Any] = {}
    for family in families:
        X, y = xy(train[train[FAMILY_COLUMN] != family])
        t0 = time.perf_counter()
        model = build_hist_gradient_boosting({**LOFO_DEFAULTS, **zcfg.params}).fit(X, y != BENIGN)
        threshold = calibrate_threshold(
            model.predict_proba(benign_val)[:, 1], cfg.detection.target_fpr
        )
        mask = fam_test == family
        unseen = model.predict_proba(X_test[mask])[:, 1] > threshold
        results[str(family)] = {
            "n_test": int(mask.sum()),
            "supervised_seen": float(seen_alerts[mask].mean()),
            "supervised_unseen": float(unseen.mean()),
            "anomaly": float(ano_alerts[mask].mean()),
            "hybrid_unseen": float((unseen | ano_alerts[mask]).mean()),
            "fit_seconds": round(time.perf_counter() - t0, 1),
        }
        log.info("held out %-12s %s", family, results[str(family)])
    return {
        "supervised_model": sup_name,
        "anomaly_model": ano_name,
        "target_fpr": cfg.detection.target_fpr,
        "anomaly_fpr": cfg.detection.anomaly_fpr,
        "families": results,
        "mean": {
            key: float(np.mean([r[key] for r in results.values()])) if results else 0.0
            for key in ("supervised_seen", "supervised_unseen", "anomaly", "hybrid_unseen")
        },
    }
