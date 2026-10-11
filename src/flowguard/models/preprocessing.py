"""Feature preprocessing for the distance/gradient based models.

Flow features are extremely heavy-tailed (durations in µs, byte rates in B/s), so a plain
StandardScaler would let a handful of huge values dominate. A signed log transform first
compresses the dynamic range; tree ensembles do not need any of this.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def signed_log1p(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def make_scaler() -> Pipeline:
    return make_pipeline(
        FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
