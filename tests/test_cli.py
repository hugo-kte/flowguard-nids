from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from flowguard.cli import app

runner = CliRunner()


def _config_file(cfg, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return path


def test_demo_pcap_then_detect(trained_config, tmp_path):
    pcap = tmp_path / "demo.pcap"
    result = runner.invoke(app, ["demo-pcap", str(pcap), "--scale", "0.05"])
    assert result.exit_code == 0, result.output
    assert pcap.stat().st_size > 0

    out = tmp_path / "alerts.jsonl"
    config = _config_file(trained_config, tmp_path)
    result = runner.invoke(app, ["detect", str(pcap), "-c", str(config), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "flows analysed" in result.output
    for line in out.read_text().splitlines():
        assert json.loads(line)["event"]["kind"] == "alert"


def test_info_reads_manifest_only(trained_config, tmp_path):
    config = _config_file(trained_config, tmp_path)
    result = runner.invoke(app, ["info", "-c", str(config)])
    assert result.exit_code == 0, result.output
    assert "hist_gradient_boosting" in result.output


def test_missing_config_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["info", "-c", str(tmp_path / "missing.yaml")])
    assert result.exit_code != 0
