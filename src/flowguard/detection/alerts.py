"""Alert model (ECS-flavoured JSON, MITRE ATT&CK mapped) and sinks."""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ANOMALY = "Anomaly"

SEVERITY_SCORE = {"low": 21, "medium": 47, "high": 73, "critical": 99}  # Elastic risk-score bands

# family -> (severity, ATT&CK tactic, technique id, technique name)
ATTACK_INFO: dict[str, tuple[str, str, str, str]] = {
    "DoS": ("high", "Impact", "T1499", "Endpoint Denial of Service"),
    "DDoS": ("high", "Impact", "T1498", "Network Denial of Service"),
    "PortScan": ("medium", "Discovery", "T1046", "Network Service Discovery"),
    "BruteForce": ("high", "Credential Access", "T1110", "Brute Force"),
    "WebAttack": ("high", "Initial Access", "T1190", "Exploit Public-Facing Application"),
    "Heartbleed": ("critical", "Initial Access", "T1190", "Exploit Public-Facing Application"),
    "Botnet": ("critical", "Command and Control", "T1071", "Application Layer Protocol"),
    "Infiltration": ("critical", "Command and Control", "T1071", "Application Layer Protocol"),
}

TRANSPORT = {1: "icmp", 6: "tcp", 17: "udp"}


def _iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_alert(
    flow: dict[str, Any],
    family: str,
    detector: str,
    attack_probability: float | None,
    anomaly_score: float | None,
    anomaly_threshold: float | None,
    explanation: list[str] | None = None,
) -> dict[str, Any]:
    """One alert document. ``flow`` holds the metadata and features of the flow."""
    severity, tactic, technique_id, technique = ATTACK_INFO.get(family, ("medium", "", "", ""))
    if detector == "both" and severity != "critical":
        severity = {"medium": "high", "high": "critical"}.get(severity, severity)
    alert: dict[str, Any] = {
        "@timestamp": _iso(flow.get("timestamp")),
        "event": {
            "kind": "alert",
            "category": ["network", "intrusion_detection"],
            "module": "flowguard",
            "severity": SEVERITY_SCORE[severity],
            "created": datetime.now(UTC).isoformat(timespec="milliseconds"),
        },
        "rule": {"name": f"FlowGuard ML - {family}"},
        "source": {"ip": flow.get("src_ip"), "port": int(flow.get("src_port", -1))},
        "destination": {"ip": flow.get("dst_ip"), "port": int(flow.get("dst_port", -1))},
        "network": {
            "transport": TRANSPORT.get(int(flow.get("protocol", 0)), "other"),
            "packets": int(flow.get("fwd_packets", 0) + flow.get("bwd_packets", 0)),
            "bytes": int(flow.get("fwd_bytes", 0) + flow.get("bwd_bytes", 0)),
            "duration_us": int(flow.get("flow_duration", 0)),
        },
        "flowguard": {
            "family": family,
            "detector": detector,
            "severity": severity,
            "attack_probability": None
            if attack_probability is None
            else round(attack_probability, 4),
            "anomaly_score": None if anomaly_score is None else round(anomaly_score, 4),
            "anomaly_ratio": (
                round(anomaly_score / anomaly_threshold, 2)
                if anomaly_score is not None and anomaly_threshold
                else None
            ),
            "explanation": explanation or [],
        },
    }
    if technique_id:
        alert["threat"] = {
            "framework": "MITRE ATT&CK",
            "tactic": {"name": tactic},
            "technique": {"id": technique_id, "name": technique},
        }
    if flow.get("ground_truth") is not None:
        alert["flowguard"]["ground_truth"] = flow["ground_truth"]
    return alert


class AlertSink:
    """Appends alerts to a JSONL file and keeps run statistics in a small JSON file.

    The stats file is replaced atomically so the dashboard never reads a half-written file.
    """

    def __init__(
        self,
        alerts_path: str | Path,
        stats_path: str | Path | None = None,
        reset: bool = False,
        source: str = "",
    ) -> None:
        self.alerts_path = Path(alerts_path)
        self.stats_path = (
            Path(stats_path) if stats_path else self.alerts_path.with_name("stats.json")
        )
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.alerts_path.write_text("")
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "source": source,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "packets": 0,
            "flows": 0,
            "alerts": 0,
            "by_family": Counter(),
            "true_positives": 0,
            "false_positives": 0,
            "missed_attacks": 0,
        }

    def write(self, alerts: list[dict[str, Any]]) -> None:
        if not alerts:
            return
        with self._lock, self.alerts_path.open("a", encoding="utf-8") as fh:
            for alert in alerts:
                fh.write(json.dumps(alert, default=str) + "\n")
                self.stats["alerts"] += 1
                self.stats["by_family"][alert["flowguard"]["family"]] += 1

    def update(self, **counters: int) -> None:
        with self._lock:
            for key, value in counters.items():
                self.stats[key] = self.stats.get(key, 0) + value
            self.stats["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            tmp = self.stats_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**self.stats, "by_family": dict(self.stats["by_family"])}))
            os.replace(tmp, self.stats_path)
