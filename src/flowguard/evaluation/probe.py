"""Robustness probe: is the classifier detecting the attack, or one tool's implementation?

The same HTTP flood (one connection per request, 300 connections from one source) is
generated in several variants. The baseline replicates the dominant DoS Hulk flow of
CICIDS2017; each variant changes one detail that has nothing to do with the attack being
a flood: the victim's TCP window or page size, server latency, how the client closes the
connection, or how the response is segmented (a capture taken behind a network tap sees
MSS-sized segments instead of the large offloaded segments seen on the server).
"""

from __future__ import annotations

from typing import Any

from flowguard.config import Config
from flowguard.data.synthetic import TrafficSynth, flows_to_frame
from flowguard.detection.engine import DetectionEngine
from flowguard.features.flowmeter import iter_flows

VARIANTS: dict[str, dict[str, Any]] = {
    "CICIDS2017 Hulk replica": {},
    "Different victim TCP window": {"server_window": 8192},
    "Different victim page size (5 kB)": {"response_bytes": 5000},
    "Fast server (1 ms)": {"server_delay_us": (500, 2_000)},
    "Graceful close (FIN)": {"teardown": "fin"},
    "MSS-sized segments": {"segment": 1448},
    "Graceful close + MSS segments": {"teardown": "fin", "segment": 1448},
}


def http_flood_probe(cfg: Config, n_connections: int = 300, seed: int = 7) -> dict[str, Any]:
    engine = DetectionEngine.from_dir(cfg.models_dir)
    results: dict[str, Any] = {}
    for name, params in VARIANTS.items():
        synth = TrafficSynth(seed=seed)
        for i in range(n_connections):
            synth.http_flood(synth.start + i * 40_000, **params)
        synth.packets.sort(key=lambda p: p.ts)
        scored = engine.score_frame(flows_to_frame(iter_flows(synth.packets)))
        results[name] = {
            "flows": len(scored),
            "supervised": float(scored["detector"].isin(["supervised", "both"]).mean()),
            "anomaly": float(scored["detector"].isin(["anomaly", "both"]).mean()),
            "hybrid": float(scored["alert"].mean()),
        }
    return {
        "supervised_model": engine.supervised,
        "anomaly_model": engine.anomaly,
        "n_connections": n_connections,
        "variants": results,
    }
