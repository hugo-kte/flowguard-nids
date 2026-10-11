"""Static figures for the evaluation report (matplotlib, Agg backend).

Styling follows a small, consistent spec: colour follows the model (never its rank), one
sequential blue ramp for rates, hairline solid grids, thin marks with rounded data-ends.
Every figure has a table twin in REPORT.md, so no value is readable from colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve

from flowguard.evaluation.naming import MODEL_LABELS, label

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("flowguard_blues", BLUES)

# Colour is bound to the model identity so it never changes between figures.
MODEL_COLORS = dict(zip(MODEL_LABELS, SERIES, strict=False))

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _figure(width: float, height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def _titles(ax: Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(
        title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=28 if subtitle else 12
    )
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _rounded_bars(
    ax: Axes,
    positions: np.ndarray,
    values: np.ndarray,
    thickness: float,
    color: str,
    horizontal: bool = False,
    radius_pt: float = 3.0,
) -> None:
    """Bars with a rounded data-end and a square baseline, each drawn as a single path.

    Axis limits must be final: the corner radius is converted from points to data units.
    """
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    r_px = radius_pt * ax.figure.dpi / 72
    # radius along the bar's length / across its thickness, in data units
    along_px = bbox.width / (x1 - x0) if horizontal else bbox.height / (y1 - y0)
    across_px = bbox.height / (y1 - y0) if horizontal else bbox.width / (x1 - x0)
    k = 0.5523  # cubic Bezier approximation of a quarter ellipse
    for pos, val in zip(positions, values, strict=True):
        if not np.isfinite(val) or val <= 0:
            continue
        length = float(val)
        ra = min(r_px / along_px, length / 2)
        rc = min(r_px / across_px, thickness / 2)
        lo, hi = pos - thickness / 2, pos + thickness / 2
        # (along, across) outline: square at 0, rounded at `length`
        outline = [
            (0, lo),
            (length - ra, lo),
            (length - ra + k * ra, lo),
            (length, lo + rc - k * rc),
            (length, lo + rc),
            (length, hi - rc),
            (length, hi - rc + k * rc),
            (length - ra + k * ra, hi),
            (length - ra, hi),
            (0, hi),
            (0, lo),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            *[MplPath.CURVE4] * 3,
            MplPath.LINETO,
            *[MplPath.CURVE4] * 3,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        verts = outline if horizontal else [(c, a) for a, c in outline]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, lw=0, snap=False))


def _cell_text_color(value: float) -> str:
    return "#ffffff" if value >= 0.55 else INK


# ------------------------------------------------------------------- figures
def roc_curves(
    is_attack: np.ndarray,
    scores: dict[str, np.ndarray],
    operating_points: dict[str, tuple[float, float]],
    budgets: dict[str, float],
    path: Path,
) -> Path:
    """ROC on a log FPR axis: the low-FPR region is the only one an IDS can operate in."""
    fig, ax = _figure(7.2, 4.6)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    lo = 1e-5
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(is_attack, score)
        keep = fpr >= lo
        ax.plot(
            np.r_[lo, fpr[keep]],
            np.r_[np.interp(lo, fpr, tpr), tpr[keep]],
            color=MODEL_COLORS.get(name, MUTED),
            lw=2,
            solid_capstyle="round",
            label=label(name),
        )
    for name, (fp, dr) in operating_points.items():
        ax.scatter(
            [max(fp, lo)],
            [dr],
            s=64,
            color=MODEL_COLORS.get(name, MUTED),
            edgecolors=SURFACE,
            linewidths=2,
            zorder=5,
        )
    for name, budget in budgets.items():
        ax.axvline(budget, color=MUTED, lw=1)
        ax.text(
            budget * 0.88,
            0.5,
            f"{name} alert budget: {budget:.1%} FPR",
            rotation=90,
            ha="right",
            va="center",
            color=INK2,
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate (log scale)")
    ax.set_ylabel("Detection rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.62), fontsize=9, labelcolor=INK2)
    _titles(
        ax,
        "Detection rate vs. false-positive rate",
        "Test set; dots mark each model's calibrated operating point",
    )
    return _save(fig, path)


def per_family_heatmap(results: dict[str, Any], families: list[str], path: Path) -> Path:
    models = list(results)
    data = np.array(
        [
            [results[m]["per_family"].get(f, {}).get("alert_rate", np.nan) for f in families]
            for m in models
        ]
    )
    fig, ax = _figure(1.05 * len(families) + 2.4, 0.55 * len(models) + 1.6)
    ax.pcolormesh(data, cmap=SEQUENTIAL, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i in range(len(models)):
        for j in range(len(families)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{v:.1%}" if v < 0.9995 else "100%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=_cell_text_color(v),
                )
    ax.set_xticks(np.arange(len(families)) + 0.5, families)
    ax.set_yticks(np.arange(len(models)) + 0.5, [label(m) for m in models])
    ax.invert_yaxis()
    _titles(
        ax,
        "Detection rate per attack family",
        "Share of test flows of each family that raised an alert",
    )
    return _save(fig, path)


def confusion_matrix_plot(cm: list[list[int]], labels: list[str], title: str, path: Path) -> Path:
    cm_arr = np.asarray(cm, dtype=float)
    norm = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)
    n = len(labels)
    fig, ax = _figure(0.75 * n + 2.6, 0.62 * n + 1.6)
    ax.pcolormesh(norm, cmap=SEQUENTIAL, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i in range(n):
        for j in range(n):
            if cm_arr[i, j] > 0:
                v = norm[i, j]
                text = "100" if v >= 0.9995 else f"{100 * v:.1f}" if v >= 0.001 else "<0.1"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=_cell_text_color(v),
                )
    ax.set_xticks(np.arange(n) + 0.5, labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(n) + 0.5, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    _titles(ax, title, "Row-normalised, % of each true family")
    return _save(fig, path)


def zero_day_bars(zero_day: dict[str, Any], path: Path) -> Path:
    fams = list(zero_day["families"])
    series = [
        ("supervised_seen", "Supervised, family seen in training"),
        ("supervised_unseen", "Supervised, family held out"),
        ("anomaly", f"{label(zero_day['anomaly_model'])} (never sees attacks)"),
        ("hybrid_unseen", "Hybrid, family held out"),
    ]
    fig, ax = _figure(max(7.2, 1.25 * len(fams) + 1.5), 4.4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(fams) - 0.4)
    ax.set_ylim(0, 1.0)
    fig.canvas.draw()
    width, step = 0.17, 0.19
    for k, (key, _name) in enumerate(series):
        offs = (k - (len(series) - 1) / 2) * step
        vals = np.array([zero_day["families"][f][key] for f in fams])
        _rounded_bars(ax, np.arange(len(fams)) + offs, vals, width, SERIES[k])
    handles = [Patch(color=SERIES[k], label=name) for k, (_, name) in enumerate(series)]
    ax.set_xticks(np.arange(len(fams)), fams)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Detection rate on the held-out family")
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        fontsize=9,
        labelcolor=INK2,
    )
    _titles(
        ax,
        "Zero-day simulation: detecting attack families never seen in training",
        f"Each family removed from training in turn; supervised at {zero_day['target_fpr']:.1%} "
        f"FPR, anomaly detector at {zero_day['anomaly_fpr']:.0%} FPR",
    )
    return _save(fig, path)


def feature_importance_bars(importances: dict[str, float], title: str, path: Path) -> Path:
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15][::-1]
    names, vals = [k for k, _ in items], np.array([v for _, v in items])
    fig, ax = _figure(7.2, 0.32 * len(items) + 1.4)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlim(0, max(vals.max() * 1.12, 1e-9))
    fig.canvas.draw()
    _rounded_bars(ax, np.arange(len(items)), vals, 0.6, SERIES[0], horizontal=True)
    ax.set_yticks(np.arange(len(items)), names)
    ax.set_xlabel("Drop in balanced accuracy when the feature is shuffled")
    _titles(ax, title, "Permutation importance on a stratified test sample, top 15")
    return _save(fig, path)


def probe_bars(probe: dict[str, Any], path: Path) -> Path:
    """Detection of the same flood under implementation variants (single series)."""
    variants = list(probe["variants"])[::-1]
    vals = np.array([probe["variants"][v]["supervised"] for v in variants])
    model = probe["supervised_model"]
    fig, ax = _figure(7.2, 0.42 * len(variants) + 1.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(variants) - 0.4)
    ax.set_xlim(0, 1.0)
    fig.canvas.draw()
    _rounded_bars(
        ax,
        np.arange(len(variants)),
        vals,
        0.55,
        MODEL_COLORS.get(model, SERIES[0]),
        horizontal=True,
    )
    for y, v in enumerate(vals):
        ax.text(min(v, 1.0) + 0.012, y, f"{v:.0%}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(np.arange(len(variants)), variants)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(f"Flood connections detected by {label(model)}")
    _titles(
        ax,
        "Same attack, different implementation",
        "Hulk-style HTTP flood, 300 connections; one detail changed per row",
    )
    return _save(fig, path)
