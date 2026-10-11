from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from flowguard.detection.alerts import AlertSink, build_alert

APP = str(Path(__file__).parents[1] / "src" / "flowguard" / "dashboard" / "app.py")


def _flow(i: int) -> dict:
    return {
        "timestamp": f"2017-07-05T12:00:{i:02d}Z",
        "src_ip": "205.174.165.73",
        "src_port": 40000 + i,
        "dst_ip": "192.168.10.50",
        "dst_port": 80,
        "protocol": 6,
        "fwd_packets": 3,
        "bwd_packets": 2,
        "fwd_bytes": 300,
        "bwd_bytes": 11595,
        "flow_duration": 5000,
        "ground_truth": "DoS Hulk",
    }


def test_dashboard_renders_without_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No alerts" in info.value for info in at.info)


def test_dashboard_renders_alerts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sink = AlertSink(tmp_path / "runtime" / "alerts.jsonl", reset=True, source="test")
    sink.write(
        [build_alert(_flow(i), "DoS", "supervised", 0.99, 1.0, 2.0, ["x = 1"]) for i in range(30)]
    )
    sink.update(flows=1000, true_positives=30)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Alerts"] == "30" and metrics["Flows analysed"] == "1,000"
    assert json.loads((tmp_path / "runtime" / "stats.json").read_text())["alerts"] == 30
