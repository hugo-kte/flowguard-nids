"""FlowGuard command-line interface."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    help="FlowGuard - hybrid ML network intrusion detection.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Experiment YAML.")]
ModelsOpt = Annotated[
    Path | None,
    typer.Option("--models", "-m", help="Model bundle dir (default: from the config)."),
]
DEFAULT_CONFIG = Path("configs/cicids2017.yaml")
RUNTIME = Path("runtime")

SEVERITY_STYLE = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _config(path: Path):
    from flowguard.config import Config

    if not path.exists():
        raise typer.BadParameter(f"config file not found: {path}")
    return Config.load(path)


def _models_dir(models: Path | None, config: Path) -> Path:
    return models if models is not None else _config(config).models_dir


def _print_alerts(alerts: list[dict], limit: int) -> None:
    if not alerts:
        console.print("[green]No alerts.[/green]")
        return
    table = Table(
        title=f"{len(alerts)} alert(s)" + (f", showing {limit}" if len(alerts) > limit else "")
    )
    for col in ("time", "severity", "family", "detector", "source", "destination", "ATT&CK", "why"):
        table.add_column(col, overflow="fold")
    for a in alerts[:limit]:
        fg = a["flowguard"]
        technique = a.get("threat", {}).get("technique", {}).get("id", "")
        table.add_row(
            str(a["@timestamp"])[11:23],
            f"[{SEVERITY_STYLE.get(fg['severity'], '')}]{fg['severity']}[/]",
            fg["family"],
            fg["detector"],
            f"{a['source']['ip']}:{a['source']['port']}",
            f"{a['destination']['ip']}:{a['destination']['port']}",
            technique,
            "; ".join(fg["explanation"][:2]),
        )
    console.print(table)


# --------------------------------------------------------------------- pipeline
@app.command()
def prepare(
    config: ConfigOpt = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option(help="Rebuild the Parquet cache.")] = False,
) -> None:
    """Load, clean and cache the dataset; print the cleaning report."""
    from flowguard.data.split import prepare as _prepare

    _, report = _prepare(_config(config), force=force)
    console.print_json(json.dumps(report))


@app.command("data-quality")
def data_quality(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Recompute the data-quality facts quoted in the README -> reports/data_quality.json."""
    from flowguard.data.quality import data_quality as _data_quality
    from flowguard.data.split import prepare as _prepare

    cfg = _config(config)
    df, report = _prepare(cfg)
    if cfg.dataset.kind != "synthetic" and "duplicates_by_family" not in report:
        df, report = _prepare(cfg, force=True)  # cache written by an older version
    facts = _data_quality(df, report)
    out = cfg.reports_dir / "data_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2) + "\n")
    console.print_json(json.dumps({k: v for k, v in facts.items() if k != "cleaning"}))
    console.print(f"Written to [bold]{out}[/bold]")


@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG,
    force_prepare: Annotated[bool, typer.Option(help="Rebuild the dataset cache first.")] = False,
) -> None:
    """Split the data, fit every enabled model, calibrate thresholds, save the bundle."""
    from flowguard.training import train as _train

    cfg = _config(config)
    bundle = _train(cfg, force_prepare=force_prepare)
    table = Table(title=f"Model bundle -> {cfg.models_dir}")
    table.add_column("model")
    table.add_column("alert threshold", justify="right")
    table.add_column("fit time (s)", justify="right")
    for name in bundle.model_names:
        table.add_row(
            name,
            f"{bundle.thresholds[name]:.6g}",
            str(bundle.metadata["fit_seconds"].get(name, "")),
        )
    console.print(table)


@app.command()
def evaluate(
    config: ConfigOpt = DEFAULT_CONFIG,
    zero_day: Annotated[
        bool | None, typer.Option("--zero-day/--no-zero-day", help="Run the LOFO experiment.")
    ] = None,
) -> None:
    """Evaluate on the test split; write reports/metrics.json, REPORT.md and figures."""
    from flowguard.evaluation.plots import label
    from flowguard.evaluation.report import build_report

    cfg = _config(config)
    metrics = build_report(cfg, zero_day=zero_day)
    table = Table(
        title=f"Test set: {metrics['n_test']:,} flows, {metrics['n_test_attacks']:,} attacks"
    )
    for col in ("model", "detection rate", "false-positive rate", "precision", "ROC-AUC"):
        table.add_column(col, justify="left" if col == "model" else "right")
    for name, r in metrics["models"].items():
        auc = "-" if r["roc_auc"] is None else f"{r['roc_auc']:.4f}"
        table.add_row(
            label(name),
            f"{r['detection_rate']:.2%}",
            f"{r['false_positive_rate']:.3%}",
            f"{r['precision']:.2%}",
            auc,
        )
    console.print(table)
    console.print(f"Report: [bold]{cfg.reports_dir / 'REPORT.md'}[/bold]")


@app.command()
def pipeline(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """prepare + train + evaluate."""
    train(config=config, force_prepare=False)
    evaluate(config=config, zero_day=None)


# -------------------------------------------------------------------- detection
@app.command()
def detect(
    pcap: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="pcap/pcapng file.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    models: ModelsOpt = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write alerts as JSONL.")] = None,
    flows_out: Annotated[Path | None, typer.Option(help="Write all scored flows (CSV).")] = None,
    show: Annotated[int, typer.Option(help="Alerts to print.")] = 20,
) -> None:
    """Analyse a capture file offline and report alerts."""
    from flowguard.detection.engine import DetectionEngine, analyze_pcap

    engine = DetectionEngine.from_dir(_models_dir(models, config))
    scored, alerts = analyze_pcap(engine, pcap)
    console.print(f"{len(scored):,} flows analysed, [bold]{len(alerts)}[/bold] alert(s).")
    if alerts:
        console.print(scored.loc[scored["alert"], "family"].value_counts().to_string())
    _print_alerts(alerts, show)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(a, default=str) + "\n" for a in alerts))
        console.print(f"Alerts written to {out}")
    if flows_out:
        scored.to_csv(flows_out, index=False)


@app.command()
def sniff(
    iface: Annotated[
        str | None, typer.Option("--iface", "-i", help="Interface (e.g. en0).")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    models: ModelsOpt = None,
    out: Annotated[Path, typer.Option("--out", "-o")] = RUNTIME / "alerts.jsonl",
    idle_timeout: Annotated[float, typer.Option(help="Close flows idle for N seconds.")] = 30.0,
) -> None:
    """Live capture (requires root): stream alerts to a JSONL file and the console."""
    from flowguard.detection.alerts import AlertSink
    from flowguard.detection.engine import DetectionEngine, LiveMonitor
    from flowguard.features.pcap import sniff_live

    engine = DetectionEngine.from_dir(_models_dir(models, config))
    sink = AlertSink(out, source=f"live:{iface or 'default'}")
    monitor = LiveMonitor(
        engine,
        sink,
        idle_timeout_s=idle_timeout,
        on_alerts=lambda alerts: _print_alerts(alerts, 10),
    )
    console.print(f"Sniffing on [bold]{iface or 'default interface'}[/bold] - Ctrl+C to stop")
    try:
        sniff_live(iface, monitor.on_packet)
    except PermissionError:
        console.print("[red]Permission denied: live capture needs root (try sudo).[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        console.print(f"Stopped. Stats: {sink.stats_path}")


@app.command()
def replay(
    config: ConfigOpt = DEFAULT_CONFIG,
    models: ModelsOpt = None,
    rate: Annotated[float, typer.Option(help="Flows per second.")] = 2_000.0,
    limit: Annotated[int | None, typer.Option(help="Stop after N flows.")] = None,
    day: Annotated[str | None, typer.Option(help="Only replay one capture day.")] = None,
    sample: Annotated[
        int | None, typer.Option(help="Random sample of N flows, kept in time order.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o")] = RUNTIME / "alerts.jsonl",
) -> None:
    """Replay the labelled test split as if it were live traffic (feeds the dashboard)."""
    from flowguard.detection.alerts import AlertSink
    from flowguard.detection.engine import DetectionEngine
    from flowguard.detection.engine import replay as _replay
    from flowguard.training import load_splits

    cfg = _config(config)
    test = load_splits(cfg, ("test",)).test
    if day:
        test = test[test["day"] == day]
    if sample and sample < len(test):
        test = test.sample(n=sample, random_state=0)
    if "timestamp" in test:
        test = test.sort_values("timestamp", kind="stable")
    if limit:
        test = test.head(limit)
    engine = DetectionEngine.from_dir(models if models is not None else cfg.models_dir)
    sink = AlertSink(out, reset=True, source=f"replay:{cfg.name}")
    console.print(f"Replaying {len(test):,} flows at {rate:,.0f} flows/s -> {out}")
    try:
        stats = _replay(engine, test.reset_index(drop=True), sink, rate=rate)
    except KeyboardInterrupt:
        stats = {k: v for k, v in sink.stats.items() if isinstance(v, int)}
    console.print_json(json.dumps(stats))


@app.command("demo-pcap")
def demo_pcap(
    out: Annotated[Path, typer.Argument(help="Output pcap path.")] = Path("demo/demo.pcap"),
    seed: int = 7,
    scale: Annotated[float, typer.Option(help="Traffic volume multiplier.")] = 0.2,
) -> None:
    """Generate a pcap with benign traffic + port scan, HTTP flood, SSH brute force, slowloris."""
    from flowguard.data.synthetic import TrafficSynth, write_pcap

    synth = TrafficSynth(seed).build(
        duration_s=300,
        n_web=int(1200 * scale),
        n_dns=int(600 * scale),
        n_scan_ports=int(500 * scale),
        n_flood=int(300 * scale),
        n_ssh=int(120 * scale),
        n_slowloris=max(1, int(30 * scale)),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    count = write_pcap(synth.packets, out)
    console.print(f"Wrote {count:,} packets to {out}")


@app.command()
def dashboard(
    port: int = 8501,
    host: Annotated[str, typer.Option(help="Bind address (0.0.0.0 in Docker).")] = "localhost",
    alerts: Annotated[Path, typer.Option(help="Alerts JSONL to follow.")] = RUNTIME
    / "alerts.jsonl",
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Launch the Streamlit SOC dashboard."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    theme = {
        "base": "dark",
        "primaryColor": "#3987e5",
        "backgroundColor": "#0d0d0d",
        "secondaryBackgroundColor": "#1a1a19",
        "textColor": "#ffffff",
    }
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "viewer",
        *[arg for key, value in theme.items() for arg in (f"--theme.{key}", value)],
        "--",
        "--alerts",
        str(alerts),
        "--config",
        str(config),
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def info(config: ConfigOpt = DEFAULT_CONFIG, models: ModelsOpt = None) -> None:
    """Show what a model bundle contains (no model is unpickled)."""
    from flowguard.models.registry import MANIFEST

    manifest = json.loads((_models_dir(models, config) / MANIFEST).read_text())
    meta = manifest["metadata"]
    console.print(
        f"created {meta.get('created_at')}  sklearn {meta.get('sklearn')}  "
        f"dataset {meta.get('dataset')}  target FPR {meta.get('target_fpr')}"
    )
    console.print(f"classes: {', '.join(manifest['classes'])}")
    table = Table()
    for col in ("model", "kind", "threshold", "sha256"):
        table.add_column(col)
    for name, art in manifest["artifacts"].items():
        table.add_row(name, art["kind"], f"{manifest['thresholds'][name]:.6g}", art["sha256"][:16])
    console.print(table)


if __name__ == "__main__":
    app()
