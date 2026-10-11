from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from flowguard.data import cicids2017
from flowguard.data.labels import FAMILY_MAP, base_label, is_attempted, normalize_label, to_family
from flowguard.data.split import cap_families, stratified_split
from flowguard.features.schema import CICIDS2017_COLUMNS, FEATURES, METADATA_COLUMNS


def raw_rows(n: int = 6) -> pd.DataFrame:
    """Rows in the improved-CICIDS2017 CSV layout."""
    rng = np.random.default_rng(0)
    data = {col: rng.integers(1, 1000, n).astype(float) for col in CICIDS2017_COLUMNS.values()}
    data["Protocol"] = np.full(n, 6.0)
    df = pd.DataFrame(data)
    df["Src IP"], df["Dst IP"] = "10.0.0.1", "10.0.0.2"
    df["Src Port"], df["Dst Port"] = 40000, 80
    df["Timestamp"] = "2017-07-05 12:00:00.000001"
    df["Label"] = "BENIGN"
    return df


def test_label_normalisation():
    assert normalize_label("Web Attack � Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("  Web Attack – XSS ") == "Web Attack - XSS"
    assert normalize_label("benign") == "BENIGN"
    assert is_attempted("DoS Hulk - Attempted") and base_label("DoS Hulk - Attempted") == "DoS Hulk"
    assert to_family("Infiltration - Portscan") == "PortScan"
    assert to_family("SSH-Patator") == "BruteForce"
    assert set(FAMILY_MAP.values()) >= {"DoS", "DDoS", "PortScan", "WebAttack", "Botnet"}


def test_clean_applies_every_rule():
    df = raw_rows(7)
    df.loc[1, "Label"] = "DoS Hulk - Attempted"
    df.loc[2, "Label"] = "Portscan"
    df.loc[3, "Protocol"] = 0  # non-IP artefact
    df.loc[4, "Fwd Header Length"] = -32_000  # wrapped 16-bit counter: kept
    df.loc[5, "Flow Bytes/s"] = np.inf  # zero-duration flow
    df.loc[6] = df.loc[0]  # exact duplicate
    clean, report = cicids2017.clean(df)
    assert report == {
        "raw_rows": 7,
        "attempted_flows": 1,
        "dropped_other_protocols": 1,
        "wrapped_header_lengths": 1,
        "zero_duration_infinite_rates": 1,
        "dropped_non_finite_rows": 0,
        "dropped_duplicates": 1,
        "clean_rows": 5,  # attempted relabelled, infinite rate fixed, wrapped header kept
    }
    assert list(clean.columns) == cicids2017.OUTPUT_COLUMNS
    assert clean["family"].tolist() == ["BENIGN", "BENIGN", "PortScan", "BENIGN", "BENIGN"]
    assert clean.loc[3, "fwd_header_len"] == -32_000
    assert clean.loc[4, "flow_bytes_per_s"] == 0
    assert clean.loc[1, "label_raw"] == "DoS Hulk - Attempted"  # relabelled, raw label kept
    assert all(clean[f].dtype == np.float32 for f in FEATURES)
    assert np.isfinite(clean[FEATURES].to_numpy()).all()


def test_attempted_can_be_dropped():
    df = raw_rows(3)
    df.loc[1, "Label"] = "FTP-Patator - Attempted"
    clean, report = cicids2017.clean(df, attempted="drop")
    assert report["clean_rows"] == 2 and "FTP-Patator - Attempted" not in clean["label"].values


def test_clean_rejects_other_formats():
    with pytest.raises(ValueError, match="missing columns"):
        cicids2017.clean(pd.DataFrame({"Label": ["BENIGN"]}))


def test_read_raw_from_zip_with_extra_columns(tmp_path):
    df = raw_rows(4)
    df["id"] = range(4)  # columns FlowGuard ignores must not be loaded
    df["Total TCP Flow Time"] = 1
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("monday.csv", df.to_csv(index=False))
        zf.writestr("tuesday.csv", df.to_csv(index=False))
    raw = cicids2017.read_raw(archive)
    assert len(raw) == 8 and set(raw["day"]) == {"monday", "tuesday"}
    assert "id" not in raw and "Total TCP Flow Time" not in raw
    assert set(METADATA_COLUMNS.values()) <= set(raw.columns)


def test_missing_dataset_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="distrinet"):
        cicids2017.read_raw(tmp_path / "nope.zip")


def test_splits_are_stratified_and_disjoint(synthetic_flows):
    train, val, test = stratified_split(synthetic_flows, 0.2, 0.2, seed=0)
    assert len(train) + len(val) + len(test) == len(synthetic_flows)
    for part in (train, val, test):
        assert set(part["family"]) == set(synthetic_flows["family"])
    capped = cap_families(train, {"BENIGN": 50}, seed=0)
    assert (capped["family"] == "BENIGN").sum() == 50
