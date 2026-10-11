"""Supervised attack-family classifiers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

RF_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 30,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}

HGB_DEFAULTS: dict[str, Any] = {
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
    "random_state": 42,
}


def build_random_forest(params: dict[str, Any] | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(**{**RF_DEFAULTS, **(params or {})})


def build_hist_gradient_boosting(
    params: dict[str, Any] | None = None,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**HGB_DEFAULTS, **(params or {})})


SUPERVISED_BUILDERS = {
    "random_forest": build_random_forest,
    "hist_gradient_boosting": build_hist_gradient_boosting,
}
