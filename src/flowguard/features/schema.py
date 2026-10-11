"""Canonical flow feature schema shared by training data and the live flow meter.

Both the CICIDS2017 loader and :mod:`flowguard.features.flowmeter` emit exactly these
columns, which is what guarantees train/serve parity: a model never sees a feature at
inference time that was computed differently from the one it was trained on.

Deliberately excluded CICFlowMeter columns:

* identifiers (flow id, IPs, ports, timestamp): they let a model memorise the lab
  topology instead of learning traffic behaviour (shortcut learning);
* exact duplicates (``Average Packet Size``, ``Fwd/Bwd Segment Size Avg``,
  ``Packet Length Variance``);
* bulk and subflow features, which are known to be unreliable in CICFlowMeter;
* ``Total TCP Flow Time``, which requires connection tracking across flow splits.
"""

from __future__ import annotations

# canonical name -> column name in the improved CICIDS2017 CSVs (Engelen et al., 2022)
CICIDS2017_COLUMNS: dict[str, str] = {
    "protocol": "Protocol",
    "flow_duration": "Flow Duration",
    "fwd_packets": "Total Fwd Packet",
    "bwd_packets": "Total Bwd packets",
    "fwd_bytes": "Total Length of Fwd Packet",
    "bwd_bytes": "Total Length of Bwd Packet",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_bytes_per_s": "Flow Bytes/s",
    "flow_packets_per_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_total": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_total": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_rst_flags": "Fwd RST Flags",
    "bwd_rst_flags": "Bwd RST Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_packets_per_s": "Fwd Packets/s",
    "bwd_packets_per_s": "Bwd Packets/s",
    "pkt_len_min": "Packet Length Min",
    "pkt_len_max": "Packet Length Max",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "fin_flag_count": "FIN Flag Count",
    "syn_flag_count": "SYN Flag Count",
    "rst_flag_count": "RST Flag Count",
    "psh_flag_count": "PSH Flag Count",
    "ack_flag_count": "ACK Flag Count",
    "urg_flag_count": "URG Flag Count",
    "cwr_flag_count": "CWR Flag Count",
    "ece_flag_count": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "fwd_init_win_bytes": "FWD Init Win Bytes",
    "bwd_init_win_bytes": "Bwd Init Win Bytes",
    "fwd_act_data_pkts": "Fwd Act Data Pkts",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}

FEATURES: list[str] = list(CICIDS2017_COLUMNS)

# Flow metadata carried alongside the features (used for alerts, never for learning).
METADATA_COLUMNS: dict[str, str] = {
    "src_ip": "Src IP",
    "src_port": "Src Port",
    "dst_ip": "Dst IP",
    "dst_port": "Dst Port",
    "timestamp": "Timestamp",
}

LABEL_COLUMN = "label"  # raw dataset label
FAMILY_COLUMN = "family"  # attack family used as the learning target
BENIGN = "BENIGN"

# Human-readable descriptions surfaced in alerts and the dashboard.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "flow_duration": "flow duration (µs)",
    "fwd_packets": "packets sent by the initiator",
    "bwd_packets": "packets sent by the responder",
    "fwd_bytes": "payload bytes sent by the initiator",
    "bwd_bytes": "payload bytes sent by the responder",
    "flow_bytes_per_s": "payload throughput (B/s)",
    "flow_packets_per_s": "packet rate (pkt/s)",
    "flow_iat_mean": "mean inter-arrival time (µs)",
    "syn_flag_count": "SYN flags",
    "rst_flag_count": "RST flags",
    "fin_flag_count": "FIN flags",
    "psh_flag_count": "PSH flags",
    "ack_flag_count": "ACK flags",
    "fwd_init_win_bytes": "initiator initial TCP window",
    "bwd_init_win_bytes": "responder initial TCP window",
    "down_up_ratio": "responder/initiator packet ratio",
    "pkt_len_max": "largest payload (B)",
    "pkt_len_mean": "mean payload (B)",
}


def describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " "))
