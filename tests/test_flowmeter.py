"""The flow meter must reproduce CICFlowMeter semantics exactly: hand-computed cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flowguard.features.flowmeter import (
    ACK,
    FIN,
    PSH,
    RST,
    SYN,
    TCP,
    UDP,
    FlowMeter,
    Packet,
    RunningStats,
    iter_flows,
)
from flowguard.features.schema import FEATURES

C, S = ("10.0.0.1", 40000), ("10.0.0.2", 80)


def tcp(ts, frm, to, flags, payload=0, header=32, window=502):
    return Packet(ts, frm[0], to[0], frm[1], to[1], TCP, payload, header, flags, window)


def http_session(t0: int = 0) -> list[Packet]:
    return [
        tcp(t0 + 0, C, S, SYN, header=40, window=64240),
        tcp(t0 + 1000, S, C, SYN | ACK, header=40, window=65160),
        tcp(t0 + 1100, C, S, ACK),
        tcp(t0 + 2000, C, S, PSH | ACK, payload=300),
        tcp(t0 + 5000, S, C, PSH | ACK, payload=1000),
        tcp(t0 + 5100, C, S, ACK),
        tcp(t0 + 6000, C, S, FIN | ACK),
        tcp(t0 + 7000, S, C, FIN | ACK),
        tcp(t0 + 7100, C, S, ACK),
    ]


def single_flow(packets: list[Packet]) -> dict[str, float]:
    flows = list(iter_flows(packets))
    assert len(flows) == 1
    return flows[0].features()


def test_feature_vector_matches_schema():
    assert list(single_flow(http_session())) == FEATURES


def test_tcp_session_features_hand_computed():
    f = single_flow(http_session())
    assert f["protocol"] == TCP
    assert f["flow_duration"] == 7100
    assert (f["fwd_packets"], f["bwd_packets"]) == (6, 3)
    assert (f["fwd_bytes"], f["bwd_bytes"]) == (300, 1000)
    assert f["fwd_pkt_len_max"] == 300 and f["fwd_pkt_len_min"] == 0
    assert f["fwd_pkt_len_mean"] == pytest.approx(50.0)
    assert f["fwd_pkt_len_std"] == pytest.approx(np.std([0, 0, 300, 0, 0, 0], ddof=1))
    assert f["bwd_pkt_len_std"] == pytest.approx(np.std([0, 1000, 0], ddof=1))
    # flow IATs: 1000, 100, 900, 3000, 100, 900, 1000, 100
    assert f["flow_iat_mean"] == pytest.approx(887.5)
    assert (f["flow_iat_max"], f["flow_iat_min"]) == (3000, 100)
    # forward IATs: 1100, 900, 3100, 900, 1100 ; backward: 4000, 2000
    assert f["fwd_iat_total"] == 7100 and f["fwd_iat_mean"] == pytest.approx(1420)
    assert (f["fwd_iat_max"], f["fwd_iat_min"]) == (3100, 900)
    assert f["bwd_iat_total"] == 6000 and f["bwd_iat_mean"] == pytest.approx(3000)
    assert (f["syn_flag_count"], f["fin_flag_count"], f["ack_flag_count"]) == (2, 2, 8)
    assert (f["psh_flag_count"], f["fwd_psh_flags"], f["bwd_psh_flags"]) == (2, 1, 1)
    assert (f["fwd_header_len"], f["bwd_header_len"]) == (200, 104)
    # forward: window of the first packet (SYN); backward: window of the *last* packet
    assert (f["fwd_init_win_bytes"], f["bwd_init_win_bytes"]) == (64240, 502)
    assert f["fwd_act_data_pkts"] == 1
    assert f["fwd_seg_size_min"] == 32
    assert f["down_up_ratio"] == pytest.approx(0.5)
    assert f["pkt_len_max"] == 1000 and f["pkt_len_mean"] == pytest.approx(1300 / 9)
    assert f["flow_bytes_per_s"] == pytest.approx(1300 / 0.0071)
    assert f["flow_packets_per_s"] == pytest.approx(9 / 0.0071)
    assert f["active_mean"] == 0 and f["idle_mean"] == 0


def test_syn_after_teardown_opens_new_flow():
    packets = http_session(0) + http_session(10_000)  # same 5-tuple reused
    flows = list(iter_flows(packets))
    assert [fl.packet_count for fl in flows] == [9, 9]


def test_rst_terminates_and_syn_reopens():
    packets = [
        tcp(0, C, S, SYN, header=40, window=1024),
        tcp(100, S, C, RST | ACK, header=20, window=0),
        tcp(5000, C, S, SYN, header=40, window=1024),
        tcp(5100, S, C, RST | ACK, header=20, window=0),
    ]
    flows = list(iter_flows(packets))
    assert len(flows) == 2
    assert flows[0].features()["rst_flag_count"] == 1
    assert flows[0].features()["bwd_rst_flags"] == 1


def test_leading_syn_ack_flips_orientation():
    packets = [tcp(0, S, C, SYN | ACK, header=40, window=65160), tcp(50, C, S, ACK)]
    flow = next(iter_flows(packets))
    assert (flow.src, flow.sport) == C  # the client is the initiator
    f = flow.features()
    # parity with CICFlowMeter: forward-only fields are set by a forward *first* packet
    assert f["fwd_init_win_bytes"] == 0 and f["fwd_seg_size_min"] == 0
    assert f["bwd_init_win_bytes"] == 65160


def test_active_timeout_splits_flow_and_keeps_orientation():
    meter = FlowMeter(flow_timeout_us=1_000_000)
    packets = [
        Packet(t, "10.0.0.5", "8.8.8.8", 5353, 53, UDP, 40, 8) for t in (0, 400_000, 900_000)
    ]
    packets.append(Packet(1_200_000, "8.8.8.8", "10.0.0.5", 53, 5353, UDP, 90, 8))
    packets.append(Packet(1_300_000, "10.0.0.5", "8.8.8.8", 5353, 53, UDP, 40, 8))
    flows = list(iter_flows(packets, meter))
    assert [fl.packet_count for fl in flows] == [3, 2]
    assert flows[1].src == "10.0.0.5"  # continuation keeps the original initiator
    assert flows[1].features()["bwd_packets"] == 1


def test_active_and_idle_periods():
    packets = [
        Packet(t, "10.0.0.5", "10.0.0.6", 1000, 2000, UDP, 10, 8)
        for t in (0, 1_000_000, 8_000_000, 8_500_000)
    ]
    f = single_flow(packets)
    assert f["active_mean"] == 1_000_000  # first burst 0 -> 1 s, closed by a 7 s gap
    assert f["idle_mean"] == 7_000_000
    assert (f["fwd_init_win_bytes"], f["fwd_seg_size_min"]) == (0, 8)


def test_single_packet_flows_are_dropped_like_in_cicids2017():
    assert list(iter_flows([tcp(0, C, S, SYN, header=40)])) == []
    assert len(list(iter_flows([tcp(0, C, S, SYN, header=40)], FlowMeter(min_packets=1)))) == 1


def test_zero_duration_rates_are_zero():
    f = single_flow([tcp(0, C, S, SYN, header=40), tcp(0, S, C, RST | ACK, header=20)])
    assert f["flow_duration"] == 0
    assert f["flow_bytes_per_s"] == 0 and f["flow_packets_per_s"] == 0


def test_expire_closes_terminated_flows_after_grace():
    meter = FlowMeter()
    for p in http_session():
        assert meter.add(p) == []
    assert meter.expire(7100 + 500, termination_grace_us=1_000) == []
    assert len(meter.expire(7100 + 5_000, termination_grace_us=1_000)) == 1
    assert len(meter) == 0


def test_running_stats_matches_numpy():
    values = np.random.default_rng(0).lognormal(5, 2, 1000)
    stats = RunningStats()
    for v in values:
        stats.add(float(v))
    assert stats.mean == pytest.approx(values.mean())
    assert stats.std == pytest.approx(values.std(ddof=1))
    assert (stats.min, stats.max) == (values.min(), values.max())
    assert math.isclose(stats.total, values.sum())


def test_pcap_round_trip_is_lossless(tmp_path):
    from flowguard.data.synthetic import write_pcap
    from flowguard.features.pcap import read_pcap

    packets = http_session(1_700_000_000_123_456)
    path = tmp_path / "s.pcap"
    write_pcap(packets, path)
    assert list(read_pcap(path)) == packets


def test_header_length_wraps_like_the_dataset():
    from flowguard.features.flowmeter import wrap_int16

    assert wrap_int16(32_767) == 32_767 and wrap_int16(32_768) == -32_768
    packets = [tcp(i * 10, C, S, ACK) for i in range(1_100)]  # 1,100 x 32-byte headers
    f = single_flow(packets)
    assert f["fwd_header_len"] == 1_100 * 32 - 65_536
