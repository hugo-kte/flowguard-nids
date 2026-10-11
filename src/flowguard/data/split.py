"""Dataset preparation helpers: caching, stratified splits, per-family caps."""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from flowguard.config import Config
from flowguard.features.schema import FAMILY_COLUMN

log = logging.getLogger(__name__)


def prepare(cfg: Config, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load + clean the configured dataset, cached as Parquet (seconds instead of minutes)."""
    cache = cfg.dataset.cache
    report_path = cache.with_suffix(".report.json")
    if cache.exists() and report_path.exists() and not force:
        log.info("using cached dataset %s", cache)
        return pd.read_parquet(cache), json.loads(report_path.read_text())

    if cfg.dataset.kind == "synthetic":
        from flowguard.data.synthetic import synthetic_dataset

        df = synthetic_dataset(seed=cfg.dataset.seed, scale=cfg.dataset.synthetic_scale)
        report = {"raw_rows": len(df), "clean_rows": len(df)}
    else:
        from flowguard.data import cicids2017

        df, report = cicids2017.load(cfg.dataset.source, attempted=cfg.dataset.attempted)

    report["families"] = df[FAMILY_COLUMN].value_counts().to_dict()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    report_path.write_text(json.dumps(report, indent=2))
    return df, report


def stratified_split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/validation/test split stratified on the attack family."""
    y = df[FAMILY_COLUMN]
    train_val, test = train_test_split(df, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[FAMILY_COLUMN], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def cap_families(df: pd.DataFrame, caps: dict[str, int], seed: int) -> pd.DataFrame:
    """Randomly subsample families above their cap (used to bound training time)."""
    if not caps:
        return df
    parts = []
    for family, group in df.groupby(FAMILY_COLUMN, sort=False):
        cap = caps.get(str(family))
        parts.append(group.sample(cap, random_state=seed) if cap and len(group) > cap else group)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
