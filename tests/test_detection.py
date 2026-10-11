from __future__ import annotations

import json

import numpy as np

from flowguard.detection.alerts import ATTACK_INFO, AlertSink, build_alert
from flowguard.detection.engine import DetectionEngine, LiveMonitor, analyze_pcap, replay
from flowguard.evaluation.report import build_report
from flowguard.features.pcap import read_pcap
from flowguard.training import load_splits


def test_pcap_analysis_flags_the_attacks(trained_config, demo_pcap):
    engine = DetectionEngine.from_dir(trained_config.models_dir)
    scored, alerts = analyze_pcap(engine, demo_pcap)
    assert len(scored) > 100 and alerts
    families = {a["flowguard"]["family"] for a in alerts}
    assert "PortScan" in families
    scan_alerts = [a for a in alerts if a["flowguard"]["family"] == "PortScan"]
    assert scan_alerts[0]["threat"]["technique"]["id"] == "T1046"
    # benign web/DNS clients should mostly stay quiet
    benign_src = scored["src_ip"].str.startswith("192.168.10.") & (scored["dst_port"] != 22)
    assert scored.loc[benign_src, "alert"].mean() < 0.2


def test_alert_document_shape():
    flow = {
        "timestamp": "2017-07-05T12:00:00Z",
        "src_ip": "1.2.3.4",
        "src_port": 1234,
        "dst_ip": "5.6.7.8",
        "dst_port": 80,
        "protocol": 6,
        "fwd_packets": 3,
        "bwd_packets": 2,
        "fwd_bytes": 10,
        "bwd_bytes": 20,
        "flow_duration": 99,
    }
    alert = build_alert(flow, "DoS", "both", 0.97, 4.0, 2.0, ["x"])
    assert alert["flowguard"]["severity"] == "critical"  # escalated: both detectors agree
    assert alert["network"] == {"transport": "tcp", "packets": 5, "bytes": 30, "duration_us": 99}
    assert alert["flowguard"]["anomaly_ratio"] == 2.0
    assert alert["threat"]["technique"]["id"] == ATTACK_INFO["DoS"][2]
    anomaly = build_alert(flow, "Anomaly", "anomaly", 0.01, 5.0, 2.0)
    assert "threat" not in anomaly and anomaly["event"]["severity"] == 47


def test_live_monitor_streams_alerts(trained_config, demo_pcap, tmp_path):
    engine = DetectionEngine.from_dir(trained_config.models_dir)
    sink = AlertSink(tmp_path / "alerts.jsonl", reset=True, source="test")
    monitor = LiveMonitor(engine, sink, idle_timeout_s=5, batch_interval_s=0.5)
    for packet in read_pcap(demo_pcap):
        monitor.on_packet(packet)
    monitor.close()
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert lines and stats["alerts"] == len(lines)
    assert stats["packets"] > 0 and stats["flows"] > 0


def test_replay_tracks_ground_truth(trained_config, tmp_path):
    engine = DetectionEngine.from_dir(trained_config.models_dir)
    test = load_splits(trained_config, ("test",)).test
    sink = AlertSink(tmp_path / "alerts.jsonl", reset=True)
    stats = replay(engine, test, sink, rate=1e9, batch=100)
    attacks = int((test["family"] != "BENIGN").sum())
    assert stats["flows"] == len(test)
    assert stats["true_positives"] + stats["missed_attacks"] == attacks
    first = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[0])
    assert "ground_truth" in first["flowguard"]


def test_report_end_to_end(trained_config):
    metrics = build_report(trained_config)
    reports = trained_config.reports_dir
    assert (reports / "metrics.json").exists() and (reports / "REPORT.md").exists()
    for fig in (
        "roc_curves",
        "per_family_detection",
        "confusion_matrix",
        "zero_day",
        "feature_importance",
        "flood_probe",
    ):
        assert (reports / "figures" / f"{fig}.png").exists(), fig
    hybrid = metrics["models"]["hybrid"]
    gb = metrics["models"]["hist_gradient_boosting"]
    assert hybrid["detection_rate"] >= gb["detection_rate"]
    assert metrics["zero_day"]["families"]
    assert metrics["flood_probe"]["variants"]
    assert np.isfinite(list(metrics["permutation_importance"].values())).all()
