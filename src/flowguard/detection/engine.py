"""Hybrid detection engine: supervised classifier OR anomaly detector -> alerts.

* The supervised model recognises *known* attack families with very few false positives.
* The autoencoder, trained on benign traffic only, flags flows that do not look like the
  benign baseline, which is how unseen attacks get caught.
* Each detector raises an alert above its own threshold, calibrated on validation benign
  traffic to the configured false-positive budget.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flowguard.data.synthetic import flows_to_frame
from flowguard.detection.alerts import ANOMALY, AlertSink, build_alert
from flowguard.features.flowmeter import Flow, FlowMeter, Packet, iter_flows
from flowguard.features.pcap import read_pcap
from flowguard.features.schema import BENIGN, FEATURES, describe
from flowguard.models.anomaly import AutoencoderDetector
from flowguard.models.registry import ModelBundle

log = logging.getLogger(__name__)


class DetectionEngine:
    def __init__(
        self,
        bundle: ModelBundle,
        supervised: str | None = None,
        anomaly: str | None = None,
    ) -> None:
        defaults = bundle.metadata.get("detection", {})
        self.bundle = bundle
        self.supervised = supervised or defaults.get("supervised_model")
        self.anomaly = anomaly or defaults.get("anomaly_model")
        if self.supervised not in bundle.supervised:
            self.supervised = next(iter(bundle.supervised), None)
        if self.anomaly not in bundle.anomaly:
            self.anomaly = next(iter(bundle.anomaly), None)
        if self.supervised is None and self.anomaly is None:
            raise ValueError("the bundle contains no usable model")

    @classmethod
    def from_dir(
        cls, models_dir: str | Path, supervised: str | None = None, anomaly: str | None = None
    ) -> DetectionEngine:
        manifest = ModelBundle.load(models_dir, models=[])  # metadata only, nothing unpickled
        detection = manifest.metadata.get("detection", {})
        wanted = [
            m
            for m in (
                supervised or detection.get("supervised_model"),
                anomaly or detection.get("anomaly_model"),
            )
            if m
        ]
        return cls(ModelBundle.load(models_dir, models=wanted or None), supervised, anomaly)

    # --------------------------------------------------------------- scoring
    def score_frame(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Adds ``attack_probability``, ``anomaly_score``, ``family``, ``detector``, ``alert``."""
        out = flows.copy()
        n = len(out)
        if n == 0:
            for col in ("attack_probability", "anomaly_score", "family", "detector", "alert"):
                out[col] = pd.Series(dtype=object)
            return out
        X = out[FEATURES].to_numpy(dtype=np.float32)
        b = self.bundle

        sup_alert = np.zeros(n, dtype=bool)
        family = np.full(n, BENIGN, dtype=object)
        if self.supervised:
            proba = b.proba(self.supervised, X)
            attack_prob = 1.0 - proba[:, b.classes.index(BENIGN)]
            sup_alert = attack_prob > b.thresholds[self.supervised]
            family = np.where(sup_alert, b.predict_attack_family(self.supervised, proba), BENIGN)
            out["attack_probability"] = attack_prob
        else:
            out["attack_probability"] = np.nan

        ano_alert = np.zeros(n, dtype=bool)
        if self.anomaly:
            ano_score = b.score(self.anomaly, X)
            ano_alert = ano_score > b.thresholds[self.anomaly]
            out["anomaly_score"] = ano_score
        else:
            out["anomaly_score"] = np.nan

        out["family"] = np.where(~sup_alert & ano_alert, ANOMALY, family)
        out["detector"] = np.select(
            [sup_alert & ano_alert, sup_alert, ano_alert], ["both", "supervised", "anomaly"], ""
        )
        out["alert"] = sup_alert | ano_alert
        return out

    def explain(self, flows: pd.DataFrame, top_k: int = 3) -> list[list[str]]:
        """Features that deviate most from the benign baseline (autoencoder residuals)."""
        model = self.bundle.anomaly.get(self.anomaly) if self.anomaly else None
        if not isinstance(model, AutoencoderDetector) or flows.empty:
            return [[] for _ in range(len(flows))]
        X = flows[FEATURES].to_numpy(dtype=np.float32)
        return [
            [f"{describe(f)} = {_fmt(row[f])}" for f in names]
            for names, (_, row) in zip(
                model.explain(X, FEATURES, top_k), flows.iterrows(), strict=True
            )
        ]

    def to_alerts(self, scored: pd.DataFrame) -> list[dict[str, Any]]:
        hits = scored[scored["alert"]]
        if hits.empty:
            return []
        explanations = self.explain(hits)
        threshold = self.bundle.thresholds.get(self.anomaly) if self.anomaly else None
        alerts = []
        for (_, row), why in zip(hits.iterrows(), explanations, strict=True):
            flow = row.to_dict()
            if "label_raw" in flow:
                flow["ground_truth"] = flow["label_raw"]
            alerts.append(
                build_alert(
                    flow,
                    family=row["family"],
                    detector=row["detector"],
                    attack_probability=_maybe(row["attack_probability"]),
                    anomaly_score=_maybe(row["anomaly_score"]),
                    anomaly_threshold=threshold,
                    explanation=why,
                )
            )
        return alerts


def _fmt(value: float) -> str:
    value = float(value)
    return f"{value:,.0f}" if abs(value) >= 100 or value.is_integer() else f"{value:.2f}"


def _maybe(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


# ------------------------------------------------------------------ front-ends
def analyze_pcap(
    engine: DetectionEngine, path: str | Path, meter: FlowMeter | None = None
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Offline analysis of a capture: exact CICFlowMeter flow semantics, then scoring."""
    flows = flows_to_frame(iter_flows(read_pcap(path), meter))
    scored = engine.score_frame(flows)
    return scored, engine.to_alerts(scored)


class LiveMonitor:
    """Streaming detection for live capture: meters packets, scores flows in micro-batches.

    Unlike offline analysis, quiet flows must be closed by the clock, so an idle timeout and
    a short grace period after TCP teardown are applied (configurable approximations).
    """

    def __init__(
        self,
        engine: DetectionEngine,
        sink: AlertSink,
        idle_timeout_s: float = 30.0,
        termination_grace_s: float = 2.0,
        batch_interval_s: float = 1.0,
        on_alerts: Any = None,
    ) -> None:
        self.engine, self.sink = engine, sink
        self.meter = FlowMeter()
        self.idle_timeout = int(idle_timeout_s * 1e6)
        self.grace = int(termination_grace_s * 1e6)
        self.interval = int(batch_interval_s * 1e6)
        self.on_alerts = on_alerts
        self._pending: list[Flow] = []
        self._last_sweep = 0
        self._packets_since_update = 0

    def on_packet(self, packet: Packet) -> None:
        self._pending += self.meter.add(packet)
        self._packets_since_update += 1
        if packet.ts - self._last_sweep >= self.interval:
            self._pending += self.meter.expire(packet.ts, self.idle_timeout, self.grace)
            self._last_sweep = packet.ts
            self.process_pending()

    def process_pending(self) -> list[dict[str, Any]]:
        flows, self._pending = self._pending, []
        alerts = self.process_flows(flows_to_frame(flows)) if flows else []
        self.sink.update(packets=self._packets_since_update)
        self._packets_since_update = 0
        return alerts

    def process_flows(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        scored = self.engine.score_frame(frame)
        alerts = self.engine.to_alerts(scored)
        self.sink.write(alerts)
        self.sink.update(flows=len(scored))
        if alerts and self.on_alerts:
            self.on_alerts(alerts)
        return alerts

    def close(self) -> list[dict[str, Any]]:
        self._pending += self.meter.flush()
        return self.process_pending()


def replay(
    engine: DetectionEngine,
    flows: pd.DataFrame,
    sink: AlertSink,
    rate: float = 2_000.0,
    batch: int = 500,
    on_alerts: Any = None,
) -> dict[str, int]:
    """Stream labelled dataset flows through the engine at ``rate`` flows/s (demo mode).

    Ground truth travels with each alert, so the dashboard can show hits and false alarms.
    """
    for start in range(0, len(flows), batch):
        t0 = time.perf_counter()
        chunk = flows.iloc[start : start + batch]
        scored = engine.score_frame(chunk)
        alerts = engine.to_alerts(scored)
        sink.write(alerts)
        truth = (chunk["family"] != BENIGN).to_numpy() if "family" in chunk else None
        counters = {"flows": len(chunk)}
        if truth is not None:
            hit = scored["alert"].to_numpy(dtype=bool)
            counters |= {
                "true_positives": int(np.sum(hit & truth)),
                "false_positives": int(np.sum(hit & ~truth)),
                "missed_attacks": int(np.sum(~hit & truth)),
            }
        sink.update(**counters)
        if alerts and on_alerts:
            on_alerts(alerts)
        pause = len(chunk) / rate - (time.perf_counter() - t0)
        if pause > 0:
            time.sleep(pause)
    return {k: v for k, v in sink.stats.items() if isinstance(v, int)}
