"""Bidirectional flow meter replicating the (fixed) CICFlowMeter feature semantics.

The models are trained on CICIDS2017 flows produced by the CICFlowMeter version fixed by
Engelen et al. (WTMC 2021 / CNS 2022). To score live traffic we must rebuild flows and
compute features *exactly* the same way, otherwise the model silently degrades
(train/serve skew). This module is a dependency-free Python port of the relevant logic:

* flows are bidirectional; the first packet defines the forward direction, except a
  leading SYN-ACK (missed SYN), which flips the orientation;
* a flow ends after the 120 s active timeout, or when a SYN re-opens a TCP connection
  that was ready for termination (FIN in both directions + final ACK, or RST);
* lengths are transport *payload* bytes (Ethernet padding excluded), header lengths
  are transport header bytes, times are microseconds, std/var are sample statistics;
* active/idle periods are split on 5 s gaps, like CICFlowMeter's activity timeout.

Known quirks of the reference implementation are reproduced on purpose and flagged
with ``# parity:`` comments.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

FIN, SYN, RST, PSH, ACK, URG, ECE, CWR = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
ICMP, TCP, UDP = 1, 6, 17

DEFAULT_FLOW_TIMEOUT_US = 120_000_000
DEFAULT_ACTIVITY_TIMEOUT_US = 5_000_000


@dataclass(slots=True, frozen=True)
class Packet:
    """The handful of fields the flow meter needs, independent of any capture library."""

    ts: int  # microseconds since the epoch
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    payload: int  # transport payload bytes
    header: int  # transport header bytes (TCP data offset * 4, UDP 8)
    flags: int = 0  # TCP flags bitmask
    window: int = 0  # TCP window; 0 for other protocols, as in CICFlowMeter
    icmp_type: int = -1
    icmp_code: int = -1

    def has(self, flag: int) -> bool:
        return bool(self.flags & flag)


class RunningStats:
    """Streaming count/sum/mean/sample-std/min/max (Welford's algorithm).

    Mirrors Apache Commons ``SummaryStatistics`` used by CICFlowMeter: the standard
    deviation is the bias-corrected sample estimate and is 0 for a single value.
    """

    __slots__ = ("_m2", "max", "mean", "min", "n", "total")

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0

    def summary(self) -> tuple[float, float, float, float]:
        """(mean, std, max, min), all zero when empty."""
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        return self.mean, self.std, self.max, self.min


class TcpState(Enum):
    OPEN = 0
    FIRST_FIN = 1
    SECOND_FIN = 2
    READY_FOR_TERMINATION = 3


_FLAG_ORDER = (FIN, SYN, RST, PSH, ACK, URG, CWR, ECE)


class Flow:
    """Accumulates the statistics of one bidirectional flow."""

    __slots__ = (
        "active",
        "bwd_header",
        "bwd_iat",
        "bwd_init_win",
        "bwd_last",
        "bwd_len",
        "dir_counts",
        "dport",
        "dst",
        "end_active",
        "flag_counts",
        "flow_iat",
        "fwd_act_data",
        "fwd_header",
        "fwd_iat",
        "fwd_init_win",
        "fwd_last",
        "fwd_len",
        "fwd_seg_min",
        "icmp_code",
        "icmp_type",
        "idle",
        "last_seen",
        "len_all",
        "proto",
        "sport",
        "src",
        "start",
        "start_active",
        "tcp_state",
    )

    def __init__(self, first: Packet, src: str, sport: int, dst: str, dport: int) -> None:
        self.src, self.sport, self.dst, self.dport = src, sport, dst, dport
        self.proto = first.proto
        self.start = self.last_seen = first.ts
        self.start_active = self.end_active = first.ts
        self.fwd_len, self.bwd_len, self.len_all = RunningStats(), RunningStats(), RunningStats()
        self.flow_iat, self.fwd_iat, self.bwd_iat = RunningStats(), RunningStats(), RunningStats()
        self.active, self.idle = RunningStats(), RunningStats()
        self.fwd_last = self.bwd_last = 0
        self.fwd_header = self.bwd_header = 0
        # per-direction PSH, URG, RST, FIN counters: [fwd_psh, bwd_psh, fwd_urg, bwd_urg, ...]
        self.dir_counts = [0] * 8
        self.flag_counts = [0] * 8
        self.fwd_act_data = 0
        self.fwd_seg_min = 0  # parity: stays 0 when the first packet is backward
        self.fwd_init_win = 0  # parity: only set by a forward *first* packet
        self.bwd_init_win = 0
        self.tcp_state = TcpState.OPEN
        self.icmp_type, self.icmp_code = first.icmp_type, first.icmp_code
        self._add(first, first_packet=True)

    # ------------------------------------------------------------------ updates
    def is_forward(self, p: Packet) -> bool:
        return p.src == self.src and p.sport == self.sport

    def _add(self, p: Packet, first_packet: bool = False) -> None:
        for i, flag in enumerate(_FLAG_ORDER):
            if p.flags & flag:
                self.flag_counts[i] += 1
        self.len_all.add(p.payload)
        forward = self.is_forward(p)
        offset = 0 if forward else 1
        if forward:
            if p.payload >= 1:
                self.fwd_act_data += 1
            self.fwd_len.add(p.payload)
            self.fwd_header += p.header
            if first_packet:
                self.fwd_seg_min = p.header
                self.fwd_init_win = p.window
            else:
                self.fwd_seg_min = min(p.header, self.fwd_seg_min)
            if self.fwd_len.n > 1:
                self.fwd_iat.add(p.ts - self.fwd_last)
            self.fwd_last = p.ts
        else:
            self.bwd_len.add(p.payload)
            self.bwd_header += p.header
            # parity: despite its name, the dataset's "Bwd Init Win Bytes" holds the window
            # of the *last* backward packet (e.g. 98% of DoS flows report 235, a scaled
            # window, although their SYN-ACK is part of the flow), so overwrite every time.
            self.bwd_init_win = p.window
            if self.bwd_len.n > 1:
                self.bwd_iat.add(p.ts - self.bwd_last)
            self.bwd_last = p.ts
        for j, flag in enumerate((PSH, URG, RST, FIN)):
            if p.flags & flag:
                self.dir_counts[2 * j + offset] += 1
        if not first_packet:
            self.flow_iat.add(p.ts - self.last_seen)
        self.last_seen = p.ts

    def add(self, p: Packet, activity_timeout: int) -> None:
        # CICFlowMeter updates active/idle periods before adding the packet.
        if p.ts - self.end_active > activity_timeout:
            if self.end_active - self.start_active > 0:
                self.active.add(self.end_active - self.start_active)
            self.idle.add(p.ts - self.end_active)
            self.start_active = self.end_active = p.ts
        else:
            self.end_active = p.ts
        self._add(p)

    # ---------------------------------------------------------------- features
    @property
    def packet_count(self) -> int:
        return self.fwd_len.n + self.bwd_len.n

    @property
    def fwd_fin(self) -> int:
        return self.dir_counts[6]

    @property
    def bwd_fin(self) -> int:
        return self.dir_counts[7]

    def features(self) -> dict[str, float]:
        duration = self.last_seen - self.start
        seconds = duration / 1e6
        fwd_n, bwd_n = self.fwd_len.n, self.bwd_len.n
        n = fwd_n + bwd_n

        def rate(count: float) -> float:
            return count / seconds if duration > 0 else 0.0

        fwd_mean, fwd_std, fwd_max, fwd_min = self.fwd_len.summary()
        bwd_mean, bwd_std, bwd_max, bwd_min = self.bwd_len.summary()
        iat_mean, iat_std, iat_max, iat_min = self.flow_iat.summary()
        fiat_mean, fiat_std, fiat_max, fiat_min = self.fwd_iat.summary() if fwd_n > 1 else (0,) * 4
        biat_mean, biat_std, biat_max, biat_min = self.bwd_iat.summary() if bwd_n > 1 else (0,) * 4
        len_mean, len_std, len_max, len_min = self.len_all.summary()
        act_mean, act_std, act_max, act_min = self.active.summary()
        idle_mean, idle_std, idle_max, idle_min = self.idle.summary()
        fwd_psh, bwd_psh, fwd_urg, bwd_urg, fwd_rst, bwd_rst, _, _ = self.dir_counts
        fin, syn, rst, psh, ack, urg, cwr, ece = self.flag_counts
        return {
            "protocol": self.proto,
            "flow_duration": duration,
            "fwd_packets": fwd_n,
            "bwd_packets": bwd_n,
            "fwd_bytes": self.fwd_len.total,
            "bwd_bytes": self.bwd_len.total,
            "fwd_pkt_len_max": fwd_max,
            "fwd_pkt_len_min": fwd_min,
            "fwd_pkt_len_mean": fwd_mean,
            "fwd_pkt_len_std": fwd_std,
            "bwd_pkt_len_max": bwd_max,
            "bwd_pkt_len_min": bwd_min,
            "bwd_pkt_len_mean": bwd_mean,
            "bwd_pkt_len_std": bwd_std,
            "flow_bytes_per_s": rate(self.fwd_len.total + self.bwd_len.total),
            "flow_packets_per_s": rate(n),
            "flow_iat_mean": iat_mean,
            "flow_iat_std": iat_std,
            "flow_iat_max": iat_max,
            "flow_iat_min": iat_min,
            "fwd_iat_total": self.fwd_iat.total if fwd_n > 1 else 0.0,
            "fwd_iat_mean": fiat_mean,
            "fwd_iat_std": fiat_std,
            "fwd_iat_max": fiat_max,
            "fwd_iat_min": fiat_min,
            "bwd_iat_total": self.bwd_iat.total if bwd_n > 1 else 0.0,
            "bwd_iat_mean": biat_mean,
            "bwd_iat_std": biat_std,
            "bwd_iat_max": biat_max,
            "bwd_iat_min": biat_min,
            "fwd_psh_flags": fwd_psh,
            "bwd_psh_flags": bwd_psh,
            "fwd_urg_flags": fwd_urg,
            "bwd_urg_flags": bwd_urg,
            "fwd_rst_flags": fwd_rst,
            "bwd_rst_flags": bwd_rst,
            "fwd_header_len": wrap_int16(self.fwd_header),
            "bwd_header_len": wrap_int16(self.bwd_header),
            "fwd_packets_per_s": rate(fwd_n),
            "bwd_packets_per_s": rate(bwd_n),
            "pkt_len_min": len_min,
            "pkt_len_max": len_max,
            "pkt_len_mean": len_mean,
            "pkt_len_std": len_std,
            "fin_flag_count": fin,
            "syn_flag_count": syn,
            "rst_flag_count": rst,
            "psh_flag_count": psh,
            "ack_flag_count": ack,
            "urg_flag_count": urg,
            "cwr_flag_count": cwr,
            "ece_flag_count": ece,
            "down_up_ratio": bwd_n / fwd_n if fwd_n > 0 else 0.0,
            "fwd_init_win_bytes": self.fwd_init_win,
            "bwd_init_win_bytes": self.bwd_init_win,
            "fwd_act_data_pkts": self.fwd_act_data,
            "fwd_seg_size_min": self.fwd_seg_min,
            "active_mean": act_mean,
            "active_std": act_std,
            "active_max": act_max,
            "active_min": act_min,
            "idle_mean": idle_mean,
            "idle_std": idle_std,
            "idle_max": idle_max,
            "idle_min": idle_min,
        }

    def record(self) -> dict[str, float | int | str]:
        """Metadata + features, ready to become a DataFrame row."""
        return {
            "src_ip": self.src,
            "src_port": self.sport,
            "dst_ip": self.dst,
            "dst_port": self.dport,
            "timestamp_us": self.start,
            **self.features(),
        }


def wrap_int16(value: int) -> int:
    """parity: the CICIDS2017 CSVs store header-length sums in a signed 16-bit field.

    Values wrap around past 32,767 bytes (roughly 1,000 packets), so long flows carry
    wrapped, even negative, header lengths. Reproducing the overflow keeps live features
    on the scale the models were trained on.
    """
    return ((int(value) + 32768) % 65536) - 32768


FlowKey = tuple[int, str, int, str, int]


def flow_key(p: Packet) -> FlowKey:
    """Direction-independent key: both directions of a conversation map to one flow."""
    a, b = (p.src, p.sport), (p.dst, p.dport)
    lo, hi = (a, b) if a <= b else (b, a)
    return (p.proto, lo[0], lo[1], hi[0], hi[1])


class FlowMeter:
    """Turns a time-ordered packet stream into finished :class:`Flow` objects.

    ``add`` returns the flows closed by the packet (timeouts, TCP connection re-use).
    ``expire`` closes flows based on the clock, which is needed for live capture where
    a quiet flow would otherwise never be reported; ``flush`` closes everything.
    """

    def __init__(
        self,
        flow_timeout_us: int = DEFAULT_FLOW_TIMEOUT_US,
        activity_timeout_us: int = DEFAULT_ACTIVITY_TIMEOUT_US,
        min_packets: int = 2,
    ) -> None:
        self.flow_timeout = flow_timeout_us
        self.activity_timeout = activity_timeout_us
        # CICIDS2017 contains no single-packet flows, so the models never saw any.
        self.min_packets = min_packets
        self._flows: dict[FlowKey, Flow] = {}
        self.packets_seen = 0
        self.flows_emitted = 0

    def __len__(self) -> int:
        return len(self._flows)

    @staticmethod
    def _new_flow(p: Packet) -> Flow:
        if p.has(SYN) and p.has(ACK):
            # SYN-ACK seen first: the sender is the responder, flip the orientation.
            return Flow(p, p.dst, p.dport, p.src, p.sport)
        return Flow(p, p.src, p.sport, p.dst, p.dport)

    def _emit(self, flows: list[Flow]) -> list[Flow]:
        kept = [f for f in flows if f.packet_count >= self.min_packets]
        self.flows_emitted += len(kept)
        return kept

    def add(self, p: Packet) -> list[Flow]:
        self.packets_seen += 1
        key = flow_key(p)
        flow = self._flows.get(key)
        if flow is None:
            self._flows[key] = self._new_flow(p)
            return []

        reopened = flow.tcp_state is TcpState.READY_FOR_TERMINATION and p.has(SYN)
        if p.ts - flow.start > self.flow_timeout or reopened:
            new_udp_dialogue = flow.proto == UDP and p.ts - flow.last_seen > self.flow_timeout
            if reopened or new_udp_dialogue or p.proto not in (TCP, UDP):
                self._flows[key] = self._new_flow(p)
            else:
                # Same TCP connection / UDP dialogue split by the timeout: keep orientation.
                self._flows[key] = Flow(p, flow.src, flow.sport, flow.dst, flow.dport)
            return self._emit([flow])

        if p.has(FIN):
            flow.add(p, self.activity_timeout)
            if flow.tcp_state is TcpState.OPEN:
                flow.tcp_state = TcpState.FIRST_FIN
            elif flow.tcp_state is TcpState.FIRST_FIN and flow.fwd_fin > 0 and flow.bwd_fin > 0:
                flow.tcp_state = TcpState.SECOND_FIN
        elif p.has(RST):
            flow.add(p, self.activity_timeout)
            flow.tcp_state = TcpState.READY_FOR_TERMINATION
        elif p.has(ACK):
            flow.add(p, self.activity_timeout)
            if flow.tcp_state is TcpState.SECOND_FIN:
                flow.tcp_state = TcpState.READY_FOR_TERMINATION
        elif (
            flow.proto == ICMP
            and flow.icmp_code != p.icmp_code
            and flow.icmp_type != p.icmp_type  # parity: both must differ
        ):
            self._flows[key] = self._new_flow(p)
            return self._emit([flow])
        else:
            flow.add(p, self.activity_timeout)
        return []

    def expire(
        self,
        now_us: int,
        idle_timeout_us: int | None = None,
        termination_grace_us: int | None = None,
    ) -> list[Flow]:
        """Close flows that are over the active timeout (exact: no later packet could be
        added to them anyway) and, optionally, idle or terminated flows (an approximation
        needed for timely live alerts)."""

        def stale(flow: Flow) -> bool:
            idle = now_us - flow.last_seen
            return (
                now_us - flow.start > self.flow_timeout
                or (idle_timeout_us is not None and idle > idle_timeout_us)
                or (
                    termination_grace_us is not None
                    and flow.tcp_state is TcpState.READY_FOR_TERMINATION
                    and idle > termination_grace_us
                )
            )

        done = [key for key, flow in self._flows.items() if stale(flow)]
        return self._emit([self._flows.pop(k) for k in done])

    def flush(self) -> list[Flow]:
        flows = list(self._flows.values())
        self._flows.clear()
        return self._emit(flows)


def iter_flows(
    packets: Iterable[Packet],
    meter: FlowMeter | None = None,
    sweep_every: int = 50_000,
) -> Iterator[Flow]:
    """Offline helper: meter a finite packet stream, yielding flows as they finish."""
    if meter is None:  # not `meter or ...`: an empty FlowMeter is falsy (__len__)
        meter = FlowMeter()
    last_ts = 0
    for i, p in enumerate(packets, 1):
        last_ts = p.ts
        yield from meter.add(p)
        if i % sweep_every == 0:  # bound memory on long captures
            yield from meter.expire(last_ts)
    yield from meter.flush()
