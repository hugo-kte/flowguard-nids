"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Strict):
    kind: Literal["cicids2017", "synthetic"] = "cicids2017"
    source: Path = Path("data/raw/CICIDS2017_improved.zip")
    cache: Path = Path("data/processed/flows.parquet")
    attempted: Literal["benign", "drop"] = "benign"
    synthetic_scale: float = 1.0
    seed: int = 0


class SplitConfig(_Strict):
    val_size: float = 0.2
    test_size: float = 0.2
    seed: int = 42
    # Cap the number of *training* rows per family (validation/test keep the real mix).
    max_train_per_family: dict[str, int] = Field(default_factory=dict)


class ModelConfig(_Strict):
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    # anomaly detectors are fitted on benign flows only, optionally subsampled
    max_train_rows: int | None = None


class ModelsConfig(_Strict):
    random_forest: ModelConfig = ModelConfig()
    hist_gradient_boosting: ModelConfig = ModelConfig()
    isolation_forest: ModelConfig = ModelConfig()
    autoencoder: ModelConfig = ModelConfig()


class DetectionConfig(_Strict):
    # Alert thresholds are calibrated on validation benign traffic to a false-positive budget.
    # Two tiers: high-fidelity supervised alerts, and a looser budget for anomaly "leads".
    target_fpr: float = 0.001
    anomaly_fpr: float = 0.01
    supervised_model: str = "hist_gradient_boosting"
    anomaly_model: str = "autoencoder"


class ZeroDayConfig(_Strict):
    enabled: bool = True
    families: list[str] | None = None  # None = every attack family with enough test samples
    min_test_samples: int = 30
    params: dict[str, Any] = Field(default_factory=dict)


class Config(_Strict):
    name: str = "flowguard"
    dataset: DatasetConfig = DatasetConfig()
    split: SplitConfig = SplitConfig()
    models: ModelsConfig = ModelsConfig()
    detection: DetectionConfig = DetectionConfig()
    zero_day: ZeroDayConfig = ZeroDayConfig()
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def splits_dir(self) -> Path:
        return self.artifacts_dir / "splits"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})
