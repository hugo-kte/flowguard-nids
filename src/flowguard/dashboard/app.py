"""FlowGuard SOC dashboard (Streamlit).

Run with ``flowguard dashboard`` (or ``streamlit run src/flowguard/dashboard/app.py``).
Three views: live alert monitoring (follows the JSONL written by ``sniff``/``replay``),
model performance (reads reports/metrics.json) and on-demand PCAP analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import deque
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from flowguard.evaluation.naming import label as model_label

# --------------------------------------------------------------------------- config
parser = argparse.ArgumentParser()
parser.add_argument("--alerts", default="runtime/alerts.jsonl")
parser.add_argument("--config", default="configs/cicids2017.yaml")
ARGS, _ = parser.parse_known_args(sys.argv[1:])

SURFACE, INK2, MUTED, GRID, BASELINE = "#1a1a19", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
# Colour follows the family (fixed order, dark-mode steps of the validated categorical palette).
FAMILY_COLORS = {
    "DoS": "#3987e5",
    "DDoS": "#d95926",
    "PortScan": "#199e70",
    "BruteForce": "#c98500",
    "WebAttack": "#d55181",
    "Botnet": "#008300",
    "Anomaly": "#9085e9",
    "Other": "#e66767",  # rare families (Infiltration, Heartbleed) are folded here
}
SEVERITY_ICON = {
    "critical": "🔴 critical",
    "high": "🟠 high",
    "medium": "🟡 medium",
    "low": "🟢 low",
}

st.set_page_config(page_title="FlowGuard SOC", page_icon="🛡️", layout="wide")


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=GRID,
            domainColor=BASELINE,
            tickColor=BASELINE,
            labelColor=INK2,
            titleColor=INK2,
            labelFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(labelColor=INK2, titleColor=INK2, orient="top")
    )


def family_scale() -> alt.Scale:
    return alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values()))


# ----------------------------------------------------------------------- data access
def tail_jsonl(path: Path, max_lines: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        lines = deque(fh, maxlen=max_lines)
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line being written right now
    return out


def alerts_frame(raw: list[dict]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.json_normalize(raw)
    df["time"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce", format="mixed")
    df["family"] = df["flowguard.family"].where(df["flowguard.family"].isin(FAMILY_COLORS), "Other")
    df["severity"] = df["flowguard.severity"]
    df["source"] = df["source.ip"].astype(str) + ":" + df["source.port"].astype(str)
    df["destination"] = df["destination.ip"].astype(str) + ":" + df["destination.port"].astype(str)
    df["technique"] = df.get("threat.technique.id", pd.Series("", index=df.index)).fillna("")
    df["why"] = df["flowguard.explanation"].map(
        lambda xs: "; ".join(xs) if isinstance(xs, list) else ""
    )
    return df


@st.cache_data(show_spinner=False, max_entries=4)
def load_alerts(path: str, mtime_ns: int, size: int, max_lines: int) -> pd.DataFrame:
    """Parse the alert tail once per file version (auto-refresh reruns are then cheap)."""
    return alerts_frame(tail_jsonl(Path(path), max_lines))


def current_alerts(path: Path, max_lines: int) -> pd.DataFrame:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return pd.DataFrame()
    return load_alerts(str(path), stat.st_mtime_ns, stat.st_size, max_lines)


def read_stats(alerts_path: Path) -> dict:
    path = alerts_path.with_name("stats.json")
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=30)
def load_metrics(reports_dir: str) -> dict | None:
    path = Path(reports_dir) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def load_engine(models_dir: str):
    from flowguard.detection.engine import DetectionEngine

    return DetectionEngine.from_dir(models_dir)


def config_paths() -> tuple[Path, Path]:
    try:
        from flowguard.config import Config

        cfg = Config.load(ARGS.config)
        return cfg.models_dir, cfg.reports_dir
    except Exception:  # dashboard must still render without a config
        return Path("artifacts/cicids2017/models"), Path("reports")


MODELS_DIR, REPORTS_DIR = config_paths()

# --------------------------------------------------------------------------- layout
st.title("🛡️ FlowGuard: network intrusion detection")
st.caption(
    "Hybrid detection: a supervised attack classifier plus an autoencoder that flags "
    "anything unlike the benign baseline. Thresholds are calibrated to a fixed "
    "false-positive budget."
)

with st.sidebar:
    st.header("Filters")
    alerts_path = Path(st.text_input("Alerts file (JSONL)", ARGS.alerts))
    max_alerts = st.select_slider("Alerts loaded", [1_000, 5_000, 20_000, 50_000], value=20_000)
    families = st.multiselect("Families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    severities = st.multiselect("Severity", list(SEVERITY_ICON), default=list(SEVERITY_ICON))
    detectors = st.multiselect(
        "Detector", ["supervised", "anomaly", "both"], default=["supervised", "anomaly", "both"]
    )
    refresh = st.select_slider(
        "Auto-refresh (s)", [0, 2, 5, 10, 30], value=5, help="0 disables auto-refresh"
    )

live_tab, perf_tab, pcap_tab = st.tabs(
    ["📡 Live alerts", "📊 Model performance", "🔍 Analyse a PCAP"]
)


@st.fragment(run_every=refresh or None)
def live_view() -> None:
    stats = read_stats(alerts_path)
    df = current_alerts(alerts_path, max_alerts)
    if not df.empty:
        df = df[
            df["family"].isin(families)
            & df["severity"].isin(severities)
            & df["flowguard.detector"].isin(detectors)
        ]

    flows = int(stats.get("flows", 0))
    n_alerts = int(stats.get("alerts", len(df)))
    tp, fp = int(stats.get("true_positives", 0)), int(stats.get("false_positives", 0))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analysed", f"{flows:,}")
    c2.metric("Alerts", f"{n_alerts:,}")
    c3.metric("Alert rate", f"{n_alerts / flows:.2%}" if flows else "–")
    if tp + fp:
        missed = int(stats.get("missed_attacks", 0))
        c4.metric(
            "Alert precision (ground truth)",
            f"{tp / (tp + fp):.2%}",
            help=f"{tp:,} true alerts, {fp:,} false alarms, {missed:,} attacks missed",
        )
    else:
        c4.metric("Source", stats.get("source", "–") or "–")

    if df.empty:
        st.info(
            f"No alerts in `{alerts_path}` yet. Start a feed with `flowguard replay` "
            "(labelled test traffic) or `sudo flowguard sniff -i <iface>`."
        )
        return

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Alerts over time")
        span = (
            (df["time"].max() - df["time"].min()).total_seconds() if df["time"].notna().any() else 0
        )
        freq, fmt = ("h", "%b %d %H:00") if span > 6 * 3600 else ("min", "%H:%M")
        if span <= 900:
            freq, fmt = "s", "%H:%M:%S"
        buckets = (
            df.assign(bucket=df["time"].dt.floor(freq))
            .groupby(["bucket", "family"], as_index=False)
            .size()
            .rename(columns={"size": "alerts"})
        )
        buckets["when"] = buckets["bucket"].dt.strftime(fmt)
        order = list(dict.fromkeys(buckets.sort_values("bucket")["when"]))
        timeline = (
            alt.Chart(buckets)
            .mark_bar(stroke=SURFACE, strokeWidth=1)
            .encode(
                x=alt.X("when:O", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("sum(alerts):Q", title="Alerts"),
                color=alt.Color(
                    "family:N",
                    scale=family_scale(),
                    title=None,
                    legend=alt.Legend(orient="top", columns=4),
                ),
                tooltip=["when:O", "family:N", alt.Tooltip("sum(alerts):Q", title="alerts")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_theme(timeline), use_container_width=True)
    with right:
        st.subheader("By family")
        counts = df["family"].value_counts().rename_axis("family").reset_index(name="alerts")
        bars = (
            alt.Chart(counts)
            .mark_bar(cornerRadiusEnd=4, size=16)
            .encode(
                y=alt.Y("family:N", sort="-x", title=None),
                x=alt.X("alerts:Q", title=None),
                color=alt.Color("family:N", scale=family_scale(), legend=None),
                tooltip=["family:N", "alerts:Q"],
            )
        )
        # values stay in text ink; the coloured bar beside them carries the identity
        labels = bars.mark_text(align="left", dx=4).encode(text="alerts:Q", color=alt.value(INK2))
        st.altair_chart(
            chart_theme((bars + labels).properties(height=260)), use_container_width=True
        )

    st.subheader("Latest alerts")
    cols = {
        "time": "Time",
        "severity": "Severity",
        "family": "Family",
        "flowguard.detector": "Detector",
        "source": "Source",
        "destination": "Destination",
        "technique": "ATT&CK",
        "flowguard.attack_probability": "P(attack)",
        "flowguard.anomaly_ratio": "Anomaly ×thr",
        "why": "Why it is unusual",
    }
    if "flowguard.ground_truth" in df:
        cols["flowguard.ground_truth"] = "Ground truth"
    latest = df.sort_values("time", ascending=False).head(200)
    view = latest[[c for c in cols if c in latest]].rename(columns=cols)
    view["Severity"] = view["Severity"].map(SEVERITY_ICON).fillna(view["Severity"])
    st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        column_config={
            "P(attack)": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
        },
    )

    st.subheader("Top talkers")
    talkers = (
        df.groupby("source.ip")
        .agg(
            alerts=("family", "size"),
            families=("family", lambda s: ", ".join(sorted(set(s)))),
            targets=("destination.ip", "nunique"),
        )
        .sort_values("alerts", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"source.ip": "Source IP"})
    )
    st.dataframe(talkers, hide_index=True, use_container_width=True)


with live_tab:
    live_view()

with perf_tab:
    metrics = load_metrics(str(REPORTS_DIR))
    if metrics is None:
        st.info(f"No `{REPORTS_DIR / 'metrics.json'}` yet: run `flowguard evaluate`.")
    else:
        st.caption(
            f"Test set: {metrics['n_test']:,} flows ({metrics['n_test_attacks']:,} attacks). "
            "Alert thresholds are calibrated on validation traffic for a false-positive budget "
            f"of {metrics['target_fpr']:.1%} (supervised) and "
            f"{metrics.get('anomaly_fpr', metrics['target_fpr']):.1%} (anomaly leads)."
        )
        rows = []
        for name, r in metrics["models"].items():
            rows.append(
                {
                    "Model": model_label(name),
                    "Type": r["kind"],
                    "Detection rate": r["detection_rate"],
                    "False-positive rate": r["false_positive_rate"],
                    "Precision": r["precision"],
                    "F1": r["f1"],
                    "ROC-AUC": r["roc_auc"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.4f")
                for c in ("Detection rate", "False-positive rate", "Precision", "F1", "ROC-AUC")
            },
        )
        fig_dir = REPORTS_DIR / "figures"
        figures = [
            ("roc_curves.png", "ROC (log FPR)"),
            ("per_family_detection.png", "Per-family detection"),
            ("zero_day.png", "Zero-day simulation"),
            ("confusion_matrix.png", "Confusion matrix"),
            ("feature_importance.png", "Feature importance"),
        ]
        available = [(f, t) for f, t in figures if (fig_dir / f).exists()]
        for i in range(0, len(available), 2):
            cols = st.columns(2)
            for col, (fname, title) in zip(cols, available[i : i + 2], strict=False):
                col.markdown(f"**{title}**")
                col.image(str(fig_dir / fname), use_container_width=True)

with pcap_tab:
    st.write(
        "Upload a capture: flows are rebuilt with the CICFlowMeter-compatible flow meter, "
        "then scored by the hybrid engine."
    )
    upload = st.file_uploader("pcap / pcapng", type=["pcap", "pcapng", "cap"])
    if upload is not None:
        from flowguard.detection.engine import analyze_pcap

        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.getbuffer())
            tmp.flush()
            with st.spinner("Metering flows and scoring..."):
                try:
                    scored, alerts = analyze_pcap(load_engine(str(MODELS_DIR)), tmp.name)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
        a, b = st.columns(2)
        a.metric("Flows", f"{len(scored):,}")
        b.metric("Alerts", f"{len(alerts):,}")
        if alerts:
            adf = alerts_frame(alerts)
            st.dataframe(
                adf[
                    [
                        "time",
                        "severity",
                        "family",
                        "flowguard.detector",
                        "source",
                        "destination",
                        "technique",
                        "why",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download alerts (JSONL)",
                "".join(json.dumps(x, default=str) + "\n" for x in alerts),
                file_name="flowguard_alerts.jsonl",
            )
        else:
            st.success("No alert raised for this capture.")
