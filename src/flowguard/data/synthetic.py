"""Synthetic network traffic for tests, CI and demos.

Generates *packets* (not feature rows) for benign sessions and a few attack scenarios,
so the whole pipeline - flow metering, training, detection - can run without the
CICIDS2017 download. Attack patterns follow the tools used in CICIDS2017 (nmap SYN scan,
Hulk-style HTTP flood, Patator SSH brute force, slowloris) closely enough to be
recognisable, but this is a smoke-test fixture, not a benchmark.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from flowguard.data.labels import to_family
from flowguard.features.flowmeter import (
    ACK,
    FIN,
    PSH,
    RST,
    SYN,
    TCP,
    UDP,
    Flow,
    FlowMeter,
    Packet,
    flow_key,
    iter_flows,
)
from flowguard.features.schema import BENIGN, FEATURES

MSS = 1448
MAC_A, MAC_B = "02:00:00:00:00:01", "02:00:00:00:00:02"  # locally administered
START_US = 1_700_000_000_000_000  # 2023-11-14, arbitrary but stable

WEB_SERVER, DNS_SERVER, SSH_SERVER = "192.168.10.50", "192.168.10.3", "192.168.10.51"
ATTACKER = "205.174.165.73"


class TrafficSynth:
    """Builds a labelled, time-ordered packet capture."""

    def __init__(self, seed: int = 0, start_us: int = START_US) -> None:
        self.rng = random.Random(seed)
        self.start = start_us
        self.packets: list[Packet] = []
        self.labels: dict[tuple, str] = {}
        self._next_port = 32768

    # ------------------------------------------------------------- primitives
    def _ephemeral_port(self) -> int:
        self._next_port += 1
        if self._next_port > 60999:
            self._next_port = 32768
        return self._next_port

    def _emit(self, p: Packet, label: str) -> None:
        self.packets.append(p)
        self.labels.setdefault(flow_key(p), label)

    def _tcp(
        self,
        t: int,
        src: str,
        dst: str,
        sport: int,
        dport: int,
        flags: int,
        payload: int = 0,
        header: int = 32,
        window: int = 502,
    ) -> Packet:
        return Packet(t, src, dst, sport, dport, TCP, payload, header, flags, window)

    def tcp_session(
        self,
        t: int,
        client: str,
        server: str,
        dport: int,
        exchanges: Iterable[tuple[bool, int, int]],
        label: str,
        rtt_us: int = 2_000,
        client_win: int = 64240,
        server_win: int = 65160,
        close: str = "client_fin",
        client_data_win: int = 502,
        server_data_win: int = 502,
    ) -> int:
        """Handshake, data exchanges ``(from_client, payload, gap_us)``, teardown.

        Returns the timestamp of the last packet.
        """
        c, s = (client, self._ephemeral_port()), (server, dport)
        jitter = lambda: max(1, int(self.rng.gauss(rtt_us, rtt_us * 0.2)))  # noqa: E731

        def send(
            frm: tuple[str, int],
            to: tuple[str, int],
            flags: int,
            payload: int = 0,
            header: int = 32,
            window: int = 502,
        ) -> None:
            self._emit(
                self._tcp(t, frm[0], to[0], frm[1], to[1], flags, payload, header, window), label
            )

        send(c, s, SYN, header=40, window=client_win)
        t += jitter()
        send(s, c, SYN | ACK, header=40, window=server_win)
        t += jitter() // 10
        send(c, s, ACK)
        exchanges = list(exchanges)
        for i, (from_client, payload, gap) in enumerate(exchanges):
            t += gap
            frm, to = (c, s) if from_client else (s, c)
            push = payload > 0 and self._ends_write(exchanges, i)
            win = client_data_win if from_client else server_data_win
            send(frm, to, (PSH | ACK) if push else ACK, payload, window=win)
        t += jitter()
        if close == "rst":
            send(c, s, RST | ACK, header=20, window=0)
            return t
        first, second = (c, s) if close == "client_fin" else (s, c)
        send(first, second, FIN | ACK)
        t += jitter()
        send(second, first, FIN | ACK)
        t += jitter() // 10
        send(first, second, ACK)
        return t

    @staticmethod
    def _ends_write(exchanges: list[tuple[bool, int, int]], i: int, burst_us: int = 2_000) -> bool:
        """PSH marks the last segment of an application write, like a real TCP stack: a
        segment is pushed unless more data follows from the same side within ``burst_us``."""
        from_client, elapsed = exchanges[i][0], 0
        for other, payload, gap in exchanges[i + 1 :]:
            elapsed += gap
            if elapsed > burst_us:
                return True
            if payload:
                return other != from_client
        return True

    # ---------------------------------------------------------------- benign
    def benign_web(self, t: int, client: str) -> int:
        exchanges: list[tuple[bool, int, int]] = []
        rtt = self.rng.randint(500, 60_000)
        for _ in range(self.rng.choice([1, 1, 1, 2, 3, 5])):  # keep-alive requests
            exchanges.append((True, self.rng.randint(280, 900), self.rng.randint(1_000, 400_000)))
            remaining = int(self.rng.lognormvariate(8.5, 1.4))  # response size
            while remaining > 0:
                chunk = min(MSS, remaining)
                remaining -= chunk
                exchanges.append((False, chunk, self.rng.randint(50, rtt // 4 + 60)))
                if self.rng.random() < 0.5:
                    exchanges.append((True, 0, self.rng.randint(20, 200)))
        port = self.rng.choice([80, 443, 443, 8080])
        return self.tcp_session(
            t,
            client,
            WEB_SERVER,
            port,
            exchanges,
            BENIGN,
            rtt_us=rtt,
            client_win=self.rng.choice([64240, 65535, 29200, 8192]),
            close=self.rng.choice(["client_fin", "server_fin"]),
        )

    def benign_dns(self, t: int, client: str) -> int:
        sport = self._ephemeral_port()
        query = self.rng.randint(25, 50)
        self._emit(Packet(t, client, DNS_SERVER, sport, 53, UDP, query, 8), BENIGN)
        t += self.rng.randint(300, 30_000)
        self._emit(
            Packet(t, DNS_SERVER, client, 53, sport, UDP, query + self.rng.randint(16, 220), 8),
            BENIGN,
        )
        return t

    # --------------------------------------------------------------- attacks
    def port_scan(self, t: int, target: str, n_ports: int) -> int:
        """nmap -sS: one SYN (window 1024, MSS option) per port, answered by RST or SYN-ACK."""
        sport = self.rng.randint(40000, 60000)
        ports = self.rng.sample(range(1, 10000), n_ports)
        open_ports = {22, 80, 443, 445, 3389}
        label = "Portscan"
        for port in ports:
            probe = self._tcp(t, ATTACKER, target, sport, port, SYN, header=24, window=1024)
            self._emit(probe, label)
            t2 = t + self.rng.randint(40, 400)
            if port in open_ports:
                syn_ack = self._tcp(t2, target, ATTACKER, port, sport, SYN | ACK, 0, 24, 29200)
                reset = self._tcp(t2 + 60, ATTACKER, target, sport, port, RST, 0, 20, 0)
                self._emit(syn_ack, label)
                self._emit(reset, label)
            else:
                self._emit(self._tcp(t2, target, ATTACKER, port, sport, RST | ACK, 0, 20, 0), label)
            t += self.rng.randint(200, 3_000)
        return t

    def http_flood(
        self,
        t: int,
        attacker: str = ATTACKER,
        label: str = "DoS Hulk",
        server_window: int = 235,
        response_bytes: int = 11595,
        segment: int = 2896,
        server_delay_us: tuple[int, int] = (20_000, 150_000),
        teardown: str = "rst",
    ) -> int:
        """One Hulk request, modelled on the dominant CICIDS2017 pattern.

        Unique GET, 11,595-byte response in 4 segments (offloaded captures show segments
        larger than the MSS), server FIN, client tears down with RST. ``server_window`` is
        the (scaled) TCP window of the victim; 235 is what the CICIDS2017 web server used,
        and 11,595 bytes is the size of the page it returned.
        """
        c, s = (attacker, self._ephemeral_port()), (WEB_SERVER, 80)

        def send(
            at: int,
            fwd: bool,
            flags: int,
            payload: int = 0,
            header: int = 32,
            window: int | None = None,
        ) -> None:
            frm, to = (c, s) if fwd else (s, c)
            win = window if window is not None else (229 if fwd else server_window)
            self._emit(
                self._tcp(at, frm[0], to[0], frm[1], to[1], flags, payload, header, win), label
            )

        jitter = self.rng.randint
        send(t, True, SYN, header=40, window=29200)
        t += jitter(200, 600)
        send(t, False, SYN | ACK, header=40, window=28960)
        t += jitter(20, 80)
        send(t, True, ACK)
        send(t + 5, True, PSH | ACK, payload=self.rng.randint(300, 420))
        t += jitter(*server_delay_us)  # the server is busy
        send(t, False, ACK)
        n_full, rest = divmod(response_bytes - 1, segment)
        chunks = [segment] * n_full + [rest + 1]
        for k, chunk in enumerate(chunks):
            t += jitter(30, 300)
            last = k == len(chunks) - 1
            send(t, False, PSH | ACK if last else ACK, payload=chunk)
            if not last:
                send(t + 15, True, ACK)
        t += jitter(100, 2_000)
        send(t, False, FIN | ACK)
        t += jitter(50, 500)
        if teardown == "rst":  # Hulk drops the socket
            send(t, True, RST | ACK, header=20, window=0)
            send(t + 10, True, RST, header=20, window=0)
        else:  # orderly close
            send(t, True, FIN | ACK)
            send(t + 200, False, ACK)
        return t

    def ssh_bruteforce(self, t: int) -> int:
        """Patator ssh_login, modelled on the dominant CICIDS2017 flow (22 packets out,
        33 back, about 12 s): banners, key exchange, then several failed passwords."""
        exchanges = [
            (False, 41, 3_000),  # server banner
            (True, 25, 300),  # client banner
            (True, 640, 200),  # KEXINIT
            (False, 976, 2_000),
            (False, 0, 100),
            (True, 48, 700),  # ECDH init
            (False, 1000, 6_000),  # ECDH reply + NEWKEYS
            (True, 16, 400),  # NEWKEYS
            (False, 0, 100),
            (True, 44, 300),  # service request
            (False, 44, 1_500),
            (True, 0, 100),
        ]
        for _ in range(self.rng.randint(6, 8)):  # failed password attempts
            exchanges += [
                (True, self.rng.choice([164, 180, 196]), self.rng.randint(20_000, 80_000)),
                (False, 0, 300),
                (False, 52, self.rng.randint(1_400_000, 1_900_000)),  # sshd failure delay
                (False, 0, 200),
                (True, 0, 200),
            ]
        return self.tcp_session(
            t,
            ATTACKER,
            SSH_SERVER,
            22,
            exchanges,
            "SSH-Patator",
            rtt_us=600,
            client_win=29200,
            server_win=28960,
            close="server_fin",
            client_data_win=229,
            server_data_win=247,
        )

    def slowloris(self, t: int) -> int:
        """Slow HTTP headers, modelled on CICIDS2017: a partial request, then a header
        chunk every 15-25 s, rarely acknowledged, and a reset after about 100 s."""
        exchanges: list[tuple[bool, int, int]] = [(True, 231, 300), (False, 0, 400)]
        for k in range(self.rng.randint(10, 12)):
            exchanges.append(
                (True, self.rng.randint(190, 231), self.rng.randint(5_000_000, 12_000_000))
            )
            if k == 5:
                exchanges.append((False, 0, 300))
        return self.tcp_session(
            t,
            ATTACKER,
            WEB_SERVER,
            80,
            exchanges,
            "DoS Slowloris",
            rtt_us=400,
            client_win=29200,
            server_win=28960,
            close="rst",
            client_data_win=229,
            server_data_win=235,
        )

    # ---------------------------------------------------------------- driver
    def build(
        self,
        duration_s: float = 600,
        n_web: int = 1200,
        n_dns: int = 600,
        n_scan_ports: int = 500,
        n_flood: int = 300,
        n_ssh: int = 120,
        n_slowloris: int = 30,
    ) -> TrafficSynth:
        span = int(duration_s * 1e6)
        clients = [f"192.168.10.{i}" for i in range(5, 25)]

        def at() -> int:
            return self.start + self.rng.randint(0, span)

        for _ in range(n_web):
            self.benign_web(at(), self.rng.choice(clients))
        for _ in range(n_dns):
            self.benign_dns(at(), self.rng.choice(clients))
        if n_scan_ports:
            self.port_scan(self.start + span // 3, WEB_SERVER, n_scan_ports)
        burst = self.start + span // 2
        for _ in range(n_flood):
            self.http_flood(burst + self.rng.randint(0, span // 10))
        for _ in range(n_ssh):
            self.ssh_bruteforce(self.start + self.rng.randint(0, span // 2))
        for _ in range(n_slowloris):
            self.slowloris(self.start + span // 4 + self.rng.randint(0, 5_000_000))
        self.packets.sort(key=lambda p: p.ts)
        return self


def flows_to_frame(flows: Iterable[Flow], labels: dict[tuple, str] | None = None) -> pd.DataFrame:
    """Convert metered flows to the canonical DataFrame used everywhere else."""
    rows = []
    for flow in flows:
        row = flow.record()
        if labels is not None:
            key = flow_key(Packet(0, flow.src, flow.dst, flow.sport, flow.dport, flow.proto, 0, 0))
            row["label_raw"] = labels.get(key, BENIGN)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df.pop("timestamp_us"), unit="us", utc=True)
    df[FEATURES] = df[FEATURES].astype(np.float32)
    if labels is not None:
        df["label"] = df["label_raw"]
        df["family"] = df["label"].map(to_family)
    return df


def synthetic_dataset(seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    """Labelled synthetic flows in the canonical schema (benign + 3 attack families)."""
    synth = TrafficSynth(seed).build(
        n_web=int(1200 * scale),
        n_dns=int(600 * scale),
        n_scan_ports=int(500 * scale),
        n_flood=int(300 * scale),
        n_ssh=int(120 * scale),
        n_slowloris=max(1, int(30 * scale)),
    )
    df = flows_to_frame(iter_flows(synth.packets, FlowMeter()), synth.labels)
    df["day"] = "synthetic"
    return df


def write_pcap(packets: Iterable[Packet], path: str | Path) -> int:
    """Serialise packets to a pcap with scapy (payload bytes are zero-filled)."""
    from scapy.layers.inet import IP
    from scapy.layers.inet import TCP as STCP
    from scapy.layers.inet import UDP as SUDP
    from scapy.layers.l2 import Ether
    from scapy.packet import Raw
    from scapy.utils import PcapWriter

    options = {
        40: [("MSS", 1460), ("SAckOK", b""), ("Timestamp", (0, 0)), ("NOP", None), ("WScale", 7)],
        32: [("NOP", None), ("NOP", None), ("Timestamp", (0, 0))],
        24: [("MSS", 1460)],
        20: [],
    }
    count = 0
    with PcapWriter(str(path), linktype=1, sync=False) as writer:
        for p in packets:
            base = Ether(src=MAC_A, dst=MAC_B) / IP(src=p.src, dst=p.dst)  # fixed MACs: no ARP
            if p.proto == TCP:
                l4 = STCP(
                    sport=p.sport,
                    dport=p.dport,
                    flags=p.flags,
                    window=p.window,
                    options=options.get(p.header, []),
                )
            else:
                l4 = SUDP(sport=p.sport, dport=p.dport)
            frame = base / l4 / Raw(b"\x00" * p.payload) if p.payload else base / l4
            frame.time = Decimal(p.ts) / 1_000_000  # exact microseconds
            writer.write(frame)
            count += 1
    return count
