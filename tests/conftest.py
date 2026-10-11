from __future__ import annotations

from pathlib import Path

import pytest

from flowguard.config import Config
from flowguard.data.synthetic import TrafficSynth, synthetic_dataset, write_pcap


def make_config(root: Path) -> Config:
    return Config.model_validate(
        {
            "name": "test",
            "dataset": {
                "kind": "synthetic",
                "cache": str(root / "flows.parquet"),
                "synthetic_scale": 0.6,
                "seed": 1,
            },
            "split": {"val_size": 0.25, "test_size": 0.25, "seed": 3},
            "models": {
                "random_forest": {"params": {"n_estimators": 20, "max_depth": 10, "n_jobs": 1}},
                "hist_gradient_boosting": {"params": {"max_iter": 30}},
                "isolation_forest": {"params": {"n_estimators": 30, "n_jobs": 1}},
                "autoencoder": {"params": {"hidden_layer_sizes": [16, 6, 16], "max_iter": 25}},
            },
            "detection": {"target_fpr": 0.02},
            "zero_day": {"enabled": True, "min_test_samples": 10, "params": {"max_iter": 20}},
            "artifacts_dir": str(root / "artifacts"),
            "reports_dir": str(root / "reports"),
        }
    )


@pytest.fixture(scope="session")
def synthetic_flows():
    return synthetic_dataset(seed=11, scale=0.5)


@pytest.fixture(scope="session")
def trained_config(tmp_path_factory) -> Config:
    """A tiny synthetic experiment, trained once for the whole test session."""
    from flowguard.training import train

    cfg = make_config(tmp_path_factory.mktemp("experiment"))
    train(cfg)
    return cfg


@pytest.fixture(scope="session")
def demo_pcap(tmp_path_factory) -> Path:
    synth = TrafficSynth(seed=5).build(
        duration_s=120, n_web=60, n_dns=30, n_scan_ports=80, n_flood=40, n_ssh=10, n_slowloris=2
    )
    path = tmp_path_factory.mktemp("pcap") / "demo.pcap"
    write_pcap(synth.packets, path)
    return path
