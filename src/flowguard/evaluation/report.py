"""Evaluation report: metrics.json (machine-readable) + REPORT.md + figures."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from flowguard.config import Config
from flowguard.data.labels import FAMILIES
from flowguard.evaluation import plots
from flowguard.evaluation.evaluate import evaluate
from flowguard.evaluation.probe import http_flood_probe
from flowguard.evaluation.zeroday import run_zero_day
from flowguard.features.schema import BENIGN, FAMILY_COLUMN, FEATURES
from flowguard.models.registry import MANIFEST, ModelBundle
from flowguard.training import load_splits

log = logging.getLogger(__name__)


def _pct(x: float | None, digits: int = 2) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 4) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _ordered_families(present: set[str]) -> list[str]:
    known = [f for f in FAMILIES if f in present]
    return known + sorted(present - set(known))


def permutation_importances(
    cfg: Config, model_name: str, per_family: int = 2_000, seed: int = 0
) -> dict[str, float]:
    bundle = ModelBundle.load(cfg.models_dir, models=[model_name])
    test = load_splits(cfg, ("test",)).test
    parts = [
        g.sample(min(len(g), per_family), random_state=seed) for _, g in test.groupby(FAMILY_COLUMN)
    ]
    sample = pd.concat(parts)
    result = permutation_importance(
        bundle.supervised[model_name],
        sample[FEATURES].to_numpy(np.float32),
        sample[FAMILY_COLUMN].to_numpy(),
        scoring="balanced_accuracy",
        n_repeats=3,
        random_state=seed,
    )
    return {f: float(v) for f, v in zip(FEATURES, result.importances_mean, strict=True)}


def build_report(cfg: Config, zero_day: bool | None = None) -> dict[str, Any]:
    reports_dir, fig_dir = cfg.reports_dir, cfg.reports_dir / "figures"
    manifest = json.loads((cfg.models_dir / MANIFEST).read_text())
    meta = manifest["metadata"]
    target_fpr = meta["target_fpr"]
    anomaly_fpr = meta.get("anomaly_fpr", target_fpr)

    results, scores, families = evaluate(cfg)
    models = results["models"]
    is_attack = families != BENIGN
    attack_families = _ordered_families(set(families) - {BENIGN})
    sup = cfg.detection.supervised_model

    plots.roc_curves(
        is_attack,
        scores,
        {m: (models[m]["false_positive_rate"], models[m]["detection_rate"]) for m in scores},
        {"supervised": target_fpr, "anomaly": anomaly_fpr},
        fig_dir / "roc_curves.png",
    )
    plots.per_family_heatmap(models, attack_families, fig_dir / "per_family_detection.png")
    if sup in models:
        mc = models[sup]["multiclass"]
        plots.confusion_matrix_plot(
            mc["confusion_matrix"],
            mc["labels"],
            f"Confusion matrix: {plots.label(sup)}",
            fig_dir / "confusion_matrix.png",
        )

    importances: dict[str, float] = {}
    if sup in manifest["artifacts"]:
        log.info("computing permutation importances for %s ...", sup)
        importances = permutation_importances(cfg, sup)
        plots.feature_importance_bars(
            importances, f"What {plots.label(sup)} relies on", fig_dir / "feature_importance.png"
        )

    log.info("probing robustness to HTTP-flood implementation details ...")
    probe = http_flood_probe(cfg)
    plots.probe_bars(probe, fig_dir / "flood_probe.png")

    zd = None
    if cfg.zero_day.enabled if zero_day is None else zero_day:
        log.info("running the leave-one-family-out (zero-day) experiment ...")
        zd = run_zero_day(cfg)
        if zd["families"]:
            plots.zero_day_bars(zd, fig_dir / "zero_day.png")

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": cfg.name,
        "dataset": meta["data_report"],
        "split_sizes": meta["split_sizes"],
        "train_rows_after_caps": meta["train_rows_after_caps"],
        "fit_seconds": meta["fit_seconds"],
        "target_fpr": target_fpr,
        "anomaly_fpr": anomaly_fpr,
        "n_test": results["n_test"],
        "n_test_attacks": results["n_attacks"],
        "models": models,
        "zero_day": zd,
        "flood_probe": probe,
        "permutation_importance": importances,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports_dir / "REPORT.md").write_text(render_markdown(metrics, attack_families, sup))
    log.info("report written to %s", reports_dir)
    return metrics


def render_markdown(m: dict[str, Any], attack_families: list[str], sup: str) -> str:
    models = m["models"]
    names = list(models)
    lines = [
        f"# FlowGuard evaluation report ({m['config']})",
        "",
        f"_Generated {m['generated_at']} by `flowguard evaluate`. "
        "Every number below is recomputed from the saved models and test split._",
        "",
        "## Data",
        "",
        "| Step | Rows |",
        "|---|---:|",
    ]
    for key, value in m["dataset"].items():
        if isinstance(value, int):
            lines.append(f"| {key.replace('_', ' ')} | {value:,} |")
    lines += [
        "",
        "| Family | Train | Train (after caps) | Validation | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    sizes, capped = m["split_sizes"], m["train_rows_after_caps"]
    for fam in [BENIGN, *attack_families]:
        lines.append(
            f"| {fam} | {sizes['train'].get(fam, 0):,} | {capped.get(fam, 0):,} | "
            f"{sizes['val'].get(fam, 0):,} | {sizes['test'].get(fam, 0):,} |"
        )

    lines += [
        "",
        "## Binary detection (test set)",
        "",
        "Alert thresholds are calibrated on *validation* benign traffic for a false-positive "
        f"budget: **{m['target_fpr']:.2%}** for supervised models (high-fidelity alerts) and "
        f"**{m['anomaly_fpr']:.2%}** for anomaly detectors (leads to triage). "
        "The test FPR shows how well each budget holds on unseen traffic.",
        "",
        "| Model | Type | FPR budget | Detection rate | Test FPR | Precision | F1 | ROC-AUC | PR-AUC | Flows/s |",  # noqa: E501
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        r = models[name]
        tput = f"{r['throughput_flows_per_s']:,}" if "throughput_flows_per_s" in r else "–"
        budget = _pct(r["fpr_budget"], 1) if "fpr_budget" in r else "–"
        lines.append(
            f"| {plots.label(name)} | {r['kind']} | {budget} | {_pct(r['detection_rate'])} | "
            f"{_pct(r['false_positive_rate'], 3)} | {_pct(r['precision'])} | {_num(r['f1'])} | "
            f"{_num(r['roc_auc'])} | {_num(r['pr_auc'])} | {tput} |"
        )
    budgets = [b for b in next(iter(models.values())).get("operating_points", {})]
    if budgets:
        lines += [
            "",
            "### Detection rate at fixed false-positive budgets",
            "",
            "Same models, thresholds re-calibrated on validation for each budget "
            "(the actual test FPR is in brackets).",
            "",
            "| Model | " + " | ".join(f"FPR budget {float(b):.1%}" for b in budgets) + " |",
            "|---|" + "---:|" * len(budgets),
        ]
        for name in names:
            ops = models[name].get("operating_points")
            if not ops:
                continue
            cells = " | ".join(
                f"{_pct(ops[b]['detection_rate'])} ({_pct(ops[b]['false_positive_rate'], 2)})"
                for b in budgets
            )
            lines.append(f"| {plots.label(name)} | {cells} |")

    lines += ["", "![ROC curves](figures/roc_curves.png)", "", "## Detection rate per family", ""]
    header = "| Family | Test flows | " + " | ".join(plots.label(n) for n in names) + " |"
    lines += [header, "|---|---:|" + "---:|" * len(names)]
    for fam in [*attack_families, BENIGN]:
        n = models[names[0]]["per_family"].get(fam, {}).get("n", 0)
        cells = " | ".join(
            _pct(models[x]["per_family"].get(fam, {}).get("alert_rate")) for x in names
        )
        name = f"{fam} (false-positive rate)" if fam == BENIGN else fam
        lines.append(f"| {name} | {n:,} | {cells} |")
    lines += ["", "![Per-family detection](figures/per_family_detection.png)", ""]

    if sup in models:
        mc = models[sup]["multiclass"]
        lines += [
            f"## Attack-family classification ({plots.label(sup)})",
            "",
            "Operational predictions (alert threshold applied): "
            f"macro-F1 **{mc['macro_f1']:.4f}**, weighted-F1 **{mc['weighted_f1']:.4f}** "
            f"(plain argmax: macro-F1 {models[sup]['multiclass_argmax']['macro_f1']:.4f}).",
            "",
            "| Family | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for fam in mc["labels"]:
            c = mc["per_class"][fam]
            lines.append(
                f"| {fam} | {_num(c['precision'])} | {_num(c['recall'])} | "
                f"{_num(c['f1-score'])} | {int(c['support']):,} |"
            )
        lines += ["", "![Confusion matrix](figures/confusion_matrix.png)", ""]

    zd = m.get("zero_day")
    if zd and zd["families"]:
        lines += [
            "## Zero-day simulation (leave one family out)",
            "",
            "Each attack family is removed from training in turn; a supervised model is re-fitted "
            "and re-calibrated, then scored on the held-out family. The autoencoder never sees "
            "attacks, so its numbers are the same with or without the family.",
            "",
            "| Held-out family | Test flows | Supervised (seen) | Supervised (unseen) | "
            f"{plots.label(zd['anomaly_model'])} | Hybrid (unseen) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for fam, r in zd["families"].items():
            lines.append(
                f"| {fam} | {r['n_test']:,} | {_pct(r['supervised_seen'], 1)} | "
                f"{_pct(r['supervised_unseen'], 1)} | {_pct(r['anomaly'], 1)} | "
                f"{_pct(r['hybrid_unseen'], 1)} |"
            )
        mean = zd["mean"]
        lines += [
            f"| **Mean** | | **{_pct(mean['supervised_seen'], 1)}** | "
            f"**{_pct(mean['supervised_unseen'], 1)}** | **{_pct(mean['anomaly'], 1)}** | "
            f"**{_pct(mean['hybrid_unseen'], 1)}** |",
            "",
            "![Zero-day simulation](figures/zero_day.png)",
            "",
        ]

    probe = m.get("flood_probe")
    if probe:
        lines += [
            "## Robustness probe: one HTTP flood, seven implementations",
            "",
            f"{probe['n_connections']} connections of a Hulk-style flood from a single source, "
            "generated as packets and metered by FlowGuard. The baseline replicates the dominant "
            "DoS Hulk flow of CICIDS2017; every other row changes one implementation detail "
            "that has nothing to do with the traffic being a flood.",
            "",
            f"| Variant | {plots.label(probe['supervised_model'])} | "
            f"{plots.label(probe['anomaly_model'])} | Hybrid |",
            "|---|---:|---:|---:|",
        ]
        for name, r in probe["variants"].items():
            lines.append(
                f"| {name} | {_pct(r['supervised'], 1)} | {_pct(r['anomaly'], 1)} | "
                f"{_pct(r['hybrid'], 1)} |"
            )
        lines += ["", "![HTTP flood probe](figures/flood_probe.png)", ""]

    if m.get("permutation_importance"):
        top = sorted(m["permutation_importance"].items(), key=lambda kv: -kv[1])[:10]
        lines += [
            "## Most important features",
            "",
            "| Feature | Balanced-accuracy drop |",
            "|---|---:|",
        ]
        lines += [f"| `{f}` | {v:.4f} |" for f, v in top]
        lines += ["", "![Feature importance](figures/feature_importance.png)", ""]

    lines += ["## Training cost", "", "| Model | Fit time (s) |", "|---|---:|"]
    lines += [f"| {plots.label(k)} | {v:,.1f} |" for k, v in m["fit_seconds"].items()]
    return "\n".join(lines) + "\n"
