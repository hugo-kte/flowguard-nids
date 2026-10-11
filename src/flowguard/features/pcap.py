"""Scapy adapters: PCAP files and live interfaces -> :class:`Packet` stream."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from flowguard.features.flowmeter import ICMP, TCP, UDP, Packet

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.layers import inet  # noqa: E402
from scapy.layers.inet6 import IPv6  # noqa: E402
from scapy.packet import Packet as ScapyPacket  # noqa: E402
from scapy.utils import PcapReader  # noqa: E402


def from_scapy(pkt: ScapyPacket) -> Packet | None:
    """Convert a dissected frame; returns ``None`` for non TCP/UDP/ICMP traffic.

    Payload sizes are derived from the IP length fields rather than the captured bytes,
    so Ethernet padding on short frames is not counted (same as CICFlowMeter).
    """
    if inet.IP in pkt:
        ip = pkt[inet.IP]
        src, dst, proto = ip.src, ip.dst, int(ip.proto)
        l4_len = ip.len - ip.ihl * 4 if ip.len else len(bytes(ip.payload))
    elif IPv6 in pkt:
        ip = pkt[IPv6]
        src, dst = ip.src, ip.dst
        # extension headers: fall back to the first transport layer found
        proto = int(ip.nh) if ip.nh in (TCP, UDP) else _transport_of(ip)
        l4_len = ip.plen if ip.plen else len(bytes(ip.payload))
    else:
        return None

    # Dispatch on the *outer* protocol: an ICMP error quoting a TCP header is ICMP.
    ts = round(float(pkt.time) * 1_000_000)
    if proto == TCP and inet.TCP in pkt:
        tcp = pkt[inet.TCP]
        header = tcp.dataofs * 4 if tcp.dataofs else 20
        return Packet(
            ts,
            src,
            dst,
            int(tcp.sport),
            int(tcp.dport),
            TCP,
            payload=max(0, l4_len - header),
            header=header,
            flags=int(tcp.flags),
            window=int(tcp.window),
        )
    if proto == UDP and inet.UDP in pkt:
        udp = pkt[inet.UDP]
        udp_len = udp.len if udp.len else l4_len
        return Packet(
            ts,
            src,
            dst,
            int(udp.sport),
            int(udp.dport),
            UDP,
            payload=max(0, udp_len - 8),
            header=8,
        )
    if proto == ICMP and inet.ICMP in pkt:
        icmp = pkt[inet.ICMP]
        return Packet(
            ts,
            src,
            dst,
            0,
            0,
            ICMP,
            payload=max(0, l4_len - 8),
            header=8,
            icmp_type=int(icmp.type),
            icmp_code=int(icmp.code),
        )
    return None


def _transport_of(layer: ScapyPacket) -> int:
    if inet.TCP in layer:
        return TCP
    if inet.UDP in layer:
        return UDP
    return -1


def read_pcap(path: str | Path) -> Iterator[Packet]:
    """Stream packets from a pcap/pcapng file without loading it in memory."""
    with PcapReader(str(path)) as reader:
        for frame in reader:
            packet = from_scapy(frame)
            if packet is not None:
                yield packet


def sniff_live(
    iface: str | None,
    on_packet: Callable[[Packet], None],
    bpf_filter: str = "ip or ip6",
    stop: Callable[[], bool] | None = None,
) -> None:
    """Capture from a network interface (needs root / CAP_NET_RAW)."""
    from scapy.sendrecv import sniff

    def _handle(frame: ScapyPacket) -> None:
        packet = from_scapy(frame)
        if packet is not None:
            on_packet(packet)

    sniff(
        iface=iface,
        filter=bpf_filter,
        prn=_handle,
        store=False,
        stop_filter=(lambda _: stop()) if stop else None,
    )
