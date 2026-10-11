"""Test-set evaluation of every model in the bundle, plus the hybrid detector."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from flowguard.config import Config
from flowguard.evaluation.metrics import binary_report, multiclass_report, per_family_rates
from flowguard.features.schema import BENIGN
from flowguard.models.registry import ModelBundle
from flowguard.training import calibrate_threshold, load_splits, xy

log = logging.getLogger(__name__)

BUDGETS = (0.001, 0.01, 0.05)  # false-positive budgets for the operating-point table


def evaluate(cfg: Config) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    """Returns (results, scores per model, test families)."""
    bundle = ModelBundle.load(cfg.models_dir)
    splits = load_splits(cfg, ("val", "test"))
    X, families = xy(splits.test)
    X_val, fam_val = xy(splits.val)
    benign_val = X_val[fam_val == BENIGN]
    is_attack = families != BENIGN
    labels = [c for c in bundle.classes if c in set(families)] or sorted(set(families))

    results: dict[str, Any] = {"n_test": len(X), "n_attacks": int(is_attack.sum()), "models": {}}
    scores: dict[str, np.ndarray] = {}
    alerts: dict[str, np.ndarray] = {}
    for name in bundle.model_names:
        t0 = time.perf_counter()
        if name in bundle.supervised:
            proba = bundle.proba(name, X)
            score = 1.0 - proba[:, bundle.classes.index(BENIGN)]
        else:
            score = bundle.score(name, X)
        elapsed = time.perf_counter() - t0
        scores[name] = score
        alerts[name] = score > bundle.thresholds[name]
        entry = binary_report(is_attack, alerts[name], score)
        entry["kind"] = "supervised" if name in bundle.supervised else "anomaly"
        entry["threshold"] = bundle.thresholds[name]
        entry["throughput_flows_per_s"] = round(len(X) / elapsed)
        entry["per_family"] = per_family_rates(families, alerts[name])
        entry["fpr_budget"] = (
            cfg.detection.target_fpr if name in bundle.supervised else cfg.detection.anomaly_fpr
        )
        val_scores = bundle.score(name, benign_val)
        entry["operating_points"] = {}
        for budget in BUDGETS:
            hit = score > calibrate_threshold(val_scores, budget)
            entry["operating_points"][f"{budget:g}"] = {
                "detection_rate": float(hit[is_attack].mean()),
                "false_positive_rate": float(hit[~is_attack].mean()),
            }
        if name in bundle.supervised:
            argmax_pred = bundle.supervised[name].classes_[proba.argmax(axis=1)]
            operational = np.where(alerts[name], bundle.predict_attack_family(name, proba), BENIGN)
            entry["multiclass_argmax"] = multiclass_report(families, argmax_pred, labels)
            entry["multiclass"] = multiclass_report(families, operational, labels)
        results["models"][name] = entry
        log.info(
            "%-24s DR=%.4f FPR=%.5f", name, entry["detection_rate"], entry["false_positive_rate"]
        )

    sup, ano = cfg.detection.supervised_model, cfg.detection.anomaly_model
    if sup in alerts and ano in alerts:
        hybrid = alerts[sup] | alerts[ano]
        entry = binary_report(is_attack, hybrid)
        entry["kind"] = "hybrid"
        entry["components"] = [sup, ano]
        entry["per_family"] = per_family_rates(families, hybrid)
        entry["anomaly_only_alerts"] = int(np.sum(alerts[ano] & ~alerts[sup]))
        results["models"]["hybrid"] = entry
    return results, scores, families
