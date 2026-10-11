"""Label normalisation and attack-family taxonomy for CICIDS2017."""

from __future__ import annotations

import re

from flowguard.features.schema import BENIGN

ATTEMPTED_SUFFIX = " - Attempted"

# Fine-grained CICIDS2017 labels -> attack families used as learning targets.
# "Infiltration - Portscan" is the nmap sweep run from the compromised host: its flows are
# behaviourally a port scan, so it joins PortScan (the raw label is kept for analysis).
FAMILY_MAP: dict[str, str] = {
    BENIGN: BENIGN,
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS Slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "Portscan": "PortScan",
    "Infiltration - Portscan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - SQL Injection": "WebAttack",
    "Botnet": "Botnet",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

FAMILIES: list[str] = list(dict.fromkeys(FAMILY_MAP.values()))
ATTACK_FAMILIES: list[str] = [f for f in FAMILIES if f != BENIGN]

_ODD_DASHES = re.compile("[–—�\x96]")  # en/em dash, mojibake from cp1252
_SPACES = re.compile(r"\s+")


def normalize_label(raw: str) -> str:
    """Canonicalise spacing and the odd dash encodings found in CICIDS2017 releases."""
    label = _SPACES.sub(" ", _ODD_DASHES.sub("-", str(raw))).strip()
    return BENIGN if label.upper() == BENIGN else label


def is_attempted(label: str) -> bool:
    return label.endswith(ATTEMPTED_SUFFIX)


def base_label(label: str) -> str:
    return label.removesuffix(ATTEMPTED_SUFFIX)


def to_family(label: str) -> str:
    """Map a (normalised, non-attempted) label to its family; unknown labels keep their name."""
    return FAMILY_MAP.get(label, label)
