"""Data-quality facts quoted in the README, recomputed from the cleaning report and the data.

* share of each family's rows that were exact duplicates (removed by ``cicids2017.clean``);
* the most common responder initial window per family: the CSVs store the window of the *last*
  backward packet, which is why most DoS flows show the same scaled value.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowguard.features.schema import FAMILY_COLUMN


def data_quality(df: pd.DataFrame, report: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, int] = report.get("families", {})
    dups: dict[str, int] = report.get("duplicates_by_family", {})
    share = {
        fam: round(dups.get(fam, 0) / (kept.get(fam, 0) + dups.get(fam, 0)), 4)
        for fam in sorted(set(kept) | set(dups))
    }
    window = {}
    for fam, values in df.groupby(FAMILY_COLUMN)["bwd_init_win_bytes"]:
        counts = values.value_counts()
        window[str(fam)] = {
            "most_common": int(counts.index[0]),
            "share": round(float(counts.iloc[0]) / len(values), 4),
        }
    return {
        "cleaning": report,
        "duplicate_share_of_raw_rows": round(
            report.get("dropped_duplicates", 0) / max(report.get("raw_rows", 1), 1), 4
        ),
        "duplicate_share_by_family": share,
        "bwd_init_win_bytes_most_common": window,
    }
