"""Loading and cleaning of the improved CICIDS2017 dataset (Liu et al., 2022).

Download: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/

Cleaning steps, each counted in the returned report so the README numbers are traceable:

1. "Attempted" flows (attack traffic that carries no malicious behaviour, e.g. a TCP
   handshake that never sent a payload) are relabelled BENIGN, as recommended by the
   dataset authors, or dropped (``attempted="drop"``);
2. only TCP/UDP/ICMP flows are kept (a few non-IP artefacts are labelled protocol 0);
3. infinite rates of zero-duration flows are set to 0, the value our flow meter emits;
4. exact duplicates (same features and same family) are removed. Duplicates inflate scores
   when the same vector lands in both the train and the test split.

Header-length sums are stored in a signed 16-bit field and wrap around on long flows
(negative values). They are kept as-is: the flow meter reproduces the overflow.
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import IO, Literal

import numpy as np
import pandas as pd

from flowguard.data.labels import base_label, is_attempted, normalize_label, to_family
from flowguard.features.schema import (
    BENIGN,
    CICIDS2017_COLUMNS,
    FAMILY_COLUMN,
    FEATURES,
    LABEL_COLUMN,
    METADATA_COLUMNS,
)

log = logging.getLogger(__name__)

RAW_LABEL = "Label"
RATE_FEATURES = ["flow_bytes_per_s", "flow_packets_per_s", "fwd_packets_per_s", "bwd_packets_per_s"]
KEPT_PROTOCOLS = (1, 6, 17)
OUTPUT_COLUMNS = [*METADATA_COLUMNS, "day", "label_raw", LABEL_COLUMN, FAMILY_COLUMN, *FEATURES]


@contextmanager
def _csv_sources(source: Path) -> Iterator[list[tuple[str, Callable[[], IO[bytes]]]]]:
    """Yield (day, opener) pairs for every CSV in a zip archive or a directory.

    Openers rather than handles: each file is opened twice (header, then data).
    """
    if source.is_dir():
        yield [(p.stem, partial(p.open, "rb")) for p in sorted(source.glob("*.csv"))]
        return
    with zipfile.ZipFile(source) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        yield [(Path(n).stem, partial(zf.open, n)) for n in names]


def read_raw(source: str | Path) -> pd.DataFrame:
    """Read all daily CSVs, keeping only the columns FlowGuard uses."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Download CICIDS2017_improved.zip from "
            "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/ "
            "(or run `make data`)."
        )
    wanted = {*CICIDS2017_COLUMNS.values(), *METADATA_COLUMNS.values(), RAW_LABEL}
    frames = []
    with _csv_sources(source) as sources:
        if not sources:
            raise FileNotFoundError(f"no CSV file found in {source}")
        for day, opener in sources:
            with opener() as fh:
                header = fh.readline().decode("utf-8", errors="replace").rstrip("\r\n")
            usecols = [c for c in header.split(",") if c.strip() in wanted]
            with opener() as fh:
                df = pd.read_csv(fh, engine="pyarrow", usecols=usecols)
            df.columns = [c.strip() for c in df.columns]
            df["day"] = day
            log.info("read %s: %d flows", day, len(df))
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(
    raw: pd.DataFrame,
    attempted: Literal["benign", "drop"] = "benign",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn raw CICIDS2017 rows into the canonical schema. Returns (frame, report)."""
    missing = {*CICIDS2017_COLUMNS.values(), RAW_LABEL} - set(raw.columns)
    if missing:
        raise ValueError(f"not an improved-CICIDS2017 file, missing columns: {sorted(missing)}")

    report: dict[str, int] = {"raw_rows": len(raw)}
    rename = {v: k for k, v in {**CICIDS2017_COLUMNS, **METADATA_COLUMNS}.items()}
    df = raw.rename(columns=rename)
    if "day" not in df:
        df["day"] = "unknown"
    for col in METADATA_COLUMNS:
        if col not in df:
            df[col] = pd.NA

    df["label_raw"] = df[RAW_LABEL].map(normalize_label)
    attempted_mask = df["label_raw"].map(is_attempted).to_numpy(dtype=bool)
    report["attempted_flows"] = int(attempted_mask.sum())
    if attempted == "drop":
        df = df.loc[~attempted_mask]
        df[LABEL_COLUMN] = df["label_raw"]
    else:
        df[LABEL_COLUMN] = np.where(attempted_mask, BENIGN, df["label_raw"].map(base_label))
    df[FAMILY_COLUMN] = df[LABEL_COLUMN].map(to_family)

    proto_ok = df["protocol"].isin(KEPT_PROTOCOLS)
    report["dropped_other_protocols"] = int((~proto_ok).sum())
    df = df.loc[proto_ok]

    # Negative header lengths are 16-bit overflows of long flows, not corrupt rows: the
    # flow meter reproduces the same wrap-around (see flowmeter.wrap_int16), so keep them.
    report["wrapped_header_lengths"] = int(
        ((df["fwd_header_len"] < 0) | (df["bwd_header_len"] < 0)).sum()
    )
    df = df.copy()

    rates = df[RATE_FEATURES].to_numpy(dtype=np.float64)
    report["zero_duration_infinite_rates"] = int((~np.isfinite(rates)).any(axis=1).sum())
    df[RATE_FEATURES] = np.where(np.isfinite(rates), rates, 0.0)

    feats = df[FEATURES].to_numpy(dtype=np.float64)
    finite = np.isfinite(feats).all(axis=1)
    report["dropped_non_finite_rows"] = int((~finite).sum())
    df = df.loc[finite]

    dup = df.duplicated(subset=[*FEATURES, FAMILY_COLUMN])
    report["dropped_duplicates"] = int(dup.sum())
    df = df.loc[~dup]

    df[FEATURES] = df[FEATURES].astype(np.float32)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("src_port", "dst_port"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(np.int32)
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


def load(
    source: str | Path,
    attempted: Literal["benign", "drop"] = "benign",
) -> tuple[pd.DataFrame, dict[str, int]]:
    return clean(read_raw(source), attempted=attempted)
