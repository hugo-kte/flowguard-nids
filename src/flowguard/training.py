"""Training pipeline: prepare -> split -> fit -> calibrate thresholds -> save bundle."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn

from flowguard import __version__
from flowguard.config import Config
from flowguard.data.split import cap_families, prepare, stratified_split
from flowguard.features.schema import BENIGN, FAMILY_COLUMN, FEATURES
from flowguard.models.anomaly import ANOMALY_BUILDERS
from flowguard.models.registry import ModelBundle
from flowguard.models.supervised import SUPERVISED_BUILDERS

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURES].to_numpy(dtype=np.float32), df[FAMILY_COLUMN].to_numpy()


def calibrate_threshold(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Alert threshold such that at most ``target_fpr`` of benign flows score above it."""
    return float(np.quantile(benign_scores, 1.0 - target_fpr, method="higher"))


def make_splits(cfg: Config, force: bool = False) -> tuple[Splits, dict]:
    df, report = prepare(cfg, force=force)
    train, val, test = stratified_split(
        df, val_size=cfg.split.val_size, test_size=cfg.split.test_size, seed=cfg.split.seed
    )
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in zip(SPLITS, (train, val, test), strict=True):
        part.to_parquet(cfg.splits_dir / f"{name}.parquet", index=False)
    return Splits(train, val, test), report


def load_splits(cfg: Config, which: tuple[str, ...] = SPLITS) -> Splits:
    missing = [s for s in which if not (cfg.splits_dir / f"{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"missing splits {missing} in {cfg.splits_dir}: run `train` first")
    parts = {s: pd.read_parquet(cfg.splits_dir / f"{s}.parquet") for s in which}
    return Splits(**{s: parts.get(s, pd.DataFrame()) for s in SPLITS})


def train(cfg: Config, force_prepare: bool = False) -> ModelBundle:
    splits, data_report = make_splits(cfg, force=force_prepare)
    train_df = cap_families(splits.train, cfg.split.max_train_per_family, cfg.split.seed)
    X, y = xy(train_df)
    X_val, y_val = xy(splits.val)
    benign_val = X_val[y_val == BENIGN]
    log.info("training on %d flows (%d features), validating on %d", len(X), X.shape[1], len(X_val))

    bundle = ModelBundle(features=FEATURES)
    fit_seconds: dict[str, float] = {}
    target_fpr = cfg.detection.target_fpr

    for name, build in SUPERVISED_BUILDERS.items():
        mcfg = getattr(cfg.models, name)
        if not mcfg.enabled:
            continue
        log.info("fitting %s ...", name)
        t0 = time.perf_counter()
        model = build(mcfg.params).fit(X, y)
        fit_seconds[name] = round(time.perf_counter() - t0, 2)
        bundle.supervised[name] = model
        bundle.classes = [str(c) for c in model.classes_]
        bundle.thresholds[name] = calibrate_threshold(bundle.score(name, benign_val), target_fpr)
        log.info("%s fitted in %.1fs", name, fit_seconds[name])

    X_benign = X[y == BENIGN]
    rng = np.random.default_rng(cfg.split.seed)
    for name, cls in ANOMALY_BUILDERS.items():
        mcfg = getattr(cfg.models, name)
        if not mcfg.enabled:
            continue
        X_fit = X_benign
        if mcfg.max_train_rows and len(X_fit) > mcfg.max_train_rows:
            X_fit = X_fit[rng.choice(len(X_fit), mcfg.max_train_rows, replace=False)]
        log.info("fitting %s on %d benign flows ...", name, len(X_fit))
        t0 = time.perf_counter()
        detector = cls(**mcfg.params).fit(X_fit)
        fit_seconds[name] = round(time.perf_counter() - t0, 2)
        bundle.anomaly[name] = detector
        bundle.thresholds[name] = calibrate_threshold(
            bundle.score(name, benign_val), cfg.detection.anomaly_fpr
        )
        log.info("%s fitted in %.1fs", name, fit_seconds[name])

    bundle.metadata = {
        "flowguard_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "dataset": cfg.dataset.kind,
        "data_report": data_report,
        "split_sizes": {
            name: part[FAMILY_COLUMN].value_counts().to_dict()
            for name, part in zip(SPLITS, (splits.train, splits.val, splits.test), strict=True)
        },
        "train_rows_after_caps": train_df[FAMILY_COLUMN].value_counts().to_dict(),
        "fit_seconds": fit_seconds,
        "target_fpr": target_fpr,
        "anomaly_fpr": cfg.detection.anomaly_fpr,
        "detection": cfg.detection.model_dump(),
        "config": cfg.model_dump(mode="json"),
    }
    bundle.save(cfg.models_dir)
    log.info("bundle saved to %s", cfg.models_dir)
    return bundle
