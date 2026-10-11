"""Detection metrics with an operational focus (detection rate at a fixed false-positive rate)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_report(
    is_attack: np.ndarray, alerts: np.ndarray, scores: np.ndarray | None = None
) -> dict[str, Any]:
    """Confusion counts and rates for alert decisions; AUCs when raw scores are given."""
    is_attack = np.asarray(is_attack, dtype=bool)
    alerts = np.asarray(alerts, dtype=bool)
    tp = int(np.sum(alerts & is_attack))
    fp = int(np.sum(alerts & ~is_attack))
    fn = int(np.sum(~alerts & is_attack))
    tn = int(np.sum(~alerts & ~is_attack))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    report: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "detection_rate": recall,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "precision": precision,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "roc_auc": None,
        "pr_auc": None,
    }
    if scores is not None and 0 < is_attack.sum() < len(is_attack):
        report["roc_auc"] = float(roc_auc_score(is_attack, scores))
        report["pr_auc"] = float(average_precision_score(is_attack, scores))
    return report


def per_family_rates(families: np.ndarray, alerts: np.ndarray) -> dict[str, dict[str, Any]]:
    """Alert rate per family: detection rate for attacks, false-positive rate for BENIGN."""
    out = {}
    for family in sorted(np.unique(families)):
        mask = families == family
        out[str(family)] = {"n": int(mask.sum()), "alert_rate": float(alerts[mask].mean())}
    return out


def multiclass_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {label: {k: float(v) for k, v in report[label].items()} for label in labels},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
