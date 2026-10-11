"""Model bundle: fitted models + calibrated thresholds + provenance, saved as one directory.

Model files are joblib pickles, and unpickling an untrusted file executes arbitrary code.
The manifest therefore stores a SHA-256 for every artifact and :meth:`ModelBundle.load`
refuses to deserialise anything that does not match. This does not replace signing (the
manifest itself could be swapped), but it catches corrupted or tampered model files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from flowguard.features.schema import BENIGN

MANIFEST = "bundle.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelBundle:
    features: list[str]
    classes: list[str] = field(default_factory=list)
    supervised: dict[str, Any] = field(default_factory=dict)
    anomaly: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- scoring
    @property
    def model_names(self) -> list[str]:
        return [*self.supervised, *self.anomaly]

    def _benign_index(self, model: Any) -> int:
        return int(np.flatnonzero(model.classes_ == BENIGN)[0])

    def proba(self, name: str, X: np.ndarray) -> np.ndarray:
        return self.supervised[name].predict_proba(X)

    def score(self, name: str, X: np.ndarray) -> np.ndarray:
        """Attack score, higher = more suspicious (``1 - P(benign)`` or anomaly score)."""
        if name in self.supervised:
            model = self.supervised[name]
            return 1.0 - model.predict_proba(X)[:, self._benign_index(model)]
        return self.anomaly[name].score_samples(X)

    def predict_attack_family(self, name: str, proba: np.ndarray) -> np.ndarray:
        """Most likely *attack* family, ignoring the benign column."""
        model = self.supervised[name]
        masked = proba.copy()
        masked[:, self._benign_index(model)] = -1.0
        return model.classes_[masked.argmax(axis=1)]

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, dict[str, str]] = {}
        for kind, models in (("supervised", self.supervised), ("anomaly", self.anomaly)):
            for name, model in models.items():
                path = directory / f"{name}.joblib"
                joblib.dump(model, path, compress=3)
                artifacts[name] = {"kind": kind, "file": path.name, "sha256": _sha256(path)}
        manifest = {
            "features": self.features,
            "classes": self.classes,
            "thresholds": self.thresholds,
            "artifacts": artifacts,
            "metadata": self.metadata,
        }
        (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))
        return directory

    @classmethod
    def load(
        cls, directory: str | Path, models: list[str] | None = None, verify: bool = True
    ) -> ModelBundle:
        directory = Path(directory)
        manifest_path = directory / MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no model bundle in {directory} (run `flowguard train`)")
        manifest = json.loads(manifest_path.read_text())
        bundle = cls(
            features=manifest["features"],
            classes=manifest["classes"],
            thresholds=manifest["thresholds"],
            metadata=manifest["metadata"],
        )
        for name, info in manifest["artifacts"].items():
            if models is not None and name not in models:
                continue
            path = directory / info["file"]
            if verify and _sha256(path) != info["sha256"]:
                raise ValueError(f"checksum mismatch for {path}: refusing to unpickle it")
            target = bundle.supervised if info["kind"] == "supervised" else bundle.anomaly
            target[name] = joblib.load(path)
        return bundle
