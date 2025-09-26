#!/usr/bin/env python3
"""
    The scripts extracts the features from the feature menu
    python pcap_to_flows_iot.py input.pcap -o flows.csv --tcp-idle 300 --udp-idle 60 --max-age 3600 \
        --k-payload 512 --internal-nets 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16
"""

import argparse
import csv
import ipaddress
import math
import os
from pathlib import Path
from config import RAW_DATA_DIRECTORY, PREPROCESSED_DATA_DIRECTORY

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.packet import Raw

INTERNAL_NETS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
TCP_IDLE_S, UDP_IDLE_S, MAX_AGE_S = 300, 60, 3600   # seconds
K_PAYLOAD_BYTES = 512                                # bytes per dir for entropy window (0 disables)
BATCH_ROWS = 5000                                    # CSV buffer
SWEEP_EVERY_PKTS = 20000                             # periodic idle sweep

def is_internal(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in net for net in INTERNAL_NETS)
    except ValueError:
        return False

def is_multicast_ip(ip_str: str) -> bool:
    try:
        return ipaddress.ip_address(ip_str).is_multicast
    except ValueError:
        return False

def port_bucket(p: int) -> str:
    if 0 <= p <= 1023: return "well_known"
    if 1024 <= p <= 49151: return "registered"
    return "ephemeral"

def iot_service_id(proto: int, pa: int, pb: int) -> str:
    """
    Coarse IoT service signature from (proto, port).
    Picks the more "service-like" (smaller) port among the two endpoints.
    """
    cand = pa if pa <= pb else pb
    m = {
        (17, 53): "DNS53",     (17,123): "NTP123",   (17,1900): "SSDP1900",
        (17,5353): "mDNS5353", (17,5355): "LLMNR5355", (17,5683): "CoAP5683",
        (6,  80): "HTTP80",    (6, 443): "TLS443",   (6, 554): "RTSP554",
        (6,8554): "RTSP8554",  (6,8008): "Cast8008", (6,8009): "Cast8009",
        (6,9100): "Printer9100",(6,1883): "MQTT1883",(6,8883): "MQTT8883",
    }
    return m.get((proto, cand), "Other")

def discovery_proto_id(proto: int, dst_port: int) -> str:
    if proto != 17: return "None"
    if dst_port == 5353: return "mDNS"
    if dst_port == 1900: return "SSDP"
    if dst_port == 3702: return "WS-Discovery"
    if dst_port == 5355: return "LLMNR"
    return "None"

def five_tuple(pkt):
    """
    Return (src_ip, src_port, dst_ip, dst_port, proto).
    Ports = 0 for non-TCP/UDP traffic.
    """
    ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
    if ip is None:
        return None
    if TCP in pkt:
        proto = 6
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif UDP in pkt:
        proto = 17
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
    else:
        sport = dport = 0
        proto = int(ip.proto) if isinstance(ip, IP) else int(ip.nh)
    return (str(ip.src), sport, str(ip.dst), dport, proto)

def canonize(k):
    """
    Merge directions by sorting endpoints.
    Returns: ((ip_lo,port_lo), (ip_hi,port_hi), proto)
    """
    a = (k[0], k[1]); b = (k[2], k[3]); p = k[4]
    return (a, b, p) if a <= b else (b, a, p)

class OnlineStats:
    __slots__ = ("n","mean","M2","min","max")
    def __init__(self):
        self.n=0; self.mean=0.0; self.M2=0.0; self.min=float("inf"); self.max=float("-inf")
    def update(self, x: float):
        self.n += 1
        if x < self.min: self.min = x
        if x > self.max: self.max = x
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def stats(self):  # (min, mean, max, std)
        var = self.M2/(self.n-1) if self.n>1 else 0.0
        std = math.sqrt(var)
        mmin = 0.0 if self.n==0 or self.min==float("inf") else self.min
        mmax = 0.0 if self.n==0 or self.max==float("-inf") else self.max
        mmean= 0.0 if self.n==0 else self.mean
        return mmin, mmean, mmax, std

class EntropyCounter:
    __slots__ = ("bins","taken","K")

    def __init__(self, K):
        self.bins=[0]*256; self.taken=0; self.K=K

    def ingest(self, payload: bytes):
        if self.K <= 0 or self.taken >= self.K or not payload: return
        room = self.K - self.taken
        chunk = payload[:room]
        for b in chunk: self.bins[b]+=1
        self.taken += len(chunk)

    def entropy(self):
        if self.taken==0: return 0.0
        H=0.0; inv=1.0/self.taken
        for c in self.bins:
            if c:
                p=c*inv; H -= p*math.log2(p)
        return H

    def nonzero_frac(self):
        if self.taken==0: return 0.0
        return 1.0 - (self.bins[0]/self.taken)

class FlowState:
    """
    Everything needed to compute features for a single conversation (flow).
    Key = ((ip_lo,port_lo),(ip_hi,port_hi),proto)
    """
    __slots__ = (
        "a","b","proto","first","last","first_src_ip","forward_is_a",
        # directional counts
        "pkts_fwd","pkts_bwd","bytes_fwd","bytes_bwd",
        # pkt length stats
        "len_fwd","len_bwd",
        # IAT stats
        "iat_fwd","iat_bwd","iat_tot","last_ts_fwd","last_ts_bwd","last_ts_any",
        # small-packet counters
        "small_fwd","small_bwd",
        # TCP flags
        "syn_cnt","ack_cnt","fin_cnt","rst_cnt","psh_cnt","urg_cnt","ece_cnt","cwr_cnt",
        # payload entropy
        "ent_fwd","ent_bwd",
        # presence flags
        "has_fwd","has_bwd",
        # misc hints
        "ttl_fwd_first","saw_multicast","saw_broadcast_l2","saw_broadcast_ip",
        # TLS placeholders (left 0 unless you add a parser)
        "tls_ext_cnt","tls_cipher_cnt","tls_sni_len",
        # TCP SYN fingerprint
        "tcp_syn_mss","tcp_syn_wscale","tcp_syn_sack_perm","tcp_syn_opt_len",
        # eviction hint
        "seen_fin_rst",
    )

    def __init__(self, ck, ts, first_src_ip):
        (self.a, self.b, self.proto) = ck
        self.first = self.last = ts
        self.first_src_ip = first_src_ip

        # Decide forward: internal endpoint → forward; else initiator
        a_ip, b_ip = self.a[0], self.b[0]
        if is_internal(a_ip) ^ is_internal(b_ip):
            self.forward_is_a = is_internal(a_ip)
        else:
            self.forward_is_a = (first_src_ip == a_ip)

        # init counters
        self.pkts_fwd=self.pkts_bwd=self.bytes_fwd=self.bytes_bwd=0
        self.len_fwd, self.len_bwd = OnlineStats(), OnlineStats()
        self.iat_fwd, self.iat_bwd, self.iat_tot = OnlineStats(), OnlineStats(), OnlineStats()
        self.last_ts_fwd=self.last_ts_bwd=self.last_ts_any=None
        self.small_fwd=self.small_bwd=0

        self.syn_cnt=self.ack_cnt=self.fin_cnt=self.rst_cnt=0
        self.psh_cnt=self.urg_cnt=self.ece_cnt=self.cwr_cnt=0

        self.ent_fwd, self.ent_bwd = EntropyCounter(K_PAYLOAD_BYTES), EntropyCounter(K_PAYLOAD_BYTES)
        self.has_fwd=self.has_bwd=0

        self.ttl_fwd_first = 0
        self.saw_multicast=False
        self.saw_broadcast_l2=False
        self.saw_broadcast_ip=False

        self.tls_ext_cnt=self.tls_cipher_cnt=self.tls_sni_len=0
        self.tcp_syn_mss=0; self.tcp_syn_wscale=0; self.tcp_syn_sack_perm=0; self.tcp_syn_opt_len=0

        self.seen_fin_rst = False

    def _is_fwd_ip(self, pkt_src_ip: str) -> bool:
        return (pkt_src_ip == self.a[0]) if self.forward_is_a else (pkt_src_ip == self.b[0])

    def update(self, pkt):
        ts = float(getattr(pkt, "time", 0.0))
        self.first = min(self.first, ts); self.last = max(self.last, ts)

        ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
        if ip is None:
            return
        src_ip, dst_ip = str(ip.src), str(ip.dst)
        fwd = self._is_fwd_ip(src_ip)

        # total IAT
        if self.last_ts_any is not None and ts >= self.last_ts_any:
            self.iat_tot.update(ts - self.last_ts_any)
        self.last_ts_any = ts

        # lengths / counts
        plen = int(len(pkt))
        if fwd:
            self.has_fwd = 1
            self.pkts_fwd += 1; self.bytes_fwd += plen; self.len_fwd.update(plen)
            if self.last_ts_fwd is not None and ts >= self.last_ts_fwd:
                self.iat_fwd.update(ts - self.last_ts_fwd)
            self.last_ts_fwd = ts
            if self.ttl_fwd_first == 0:
                ttl = int(getattr(ip, "ttl", getattr(ip, "hlim", 0)) or 0)
                self.ttl_fwd_first = ttl
        else:
            self.has_bwd = 1
            self.pkts_bwd += 1; self.bytes_bwd += plen; self.len_bwd.update(plen)
            if self.last_ts_bwd is not None and ts >= self.last_ts_bwd:
                self.iat_bwd.update(ts - self.last_ts_bwd)
            self.last_ts_bwd = ts

        # small packet counters (<= 100 bytes)
        if fwd: self.small_fwd += (1 if plen <= 100 else 0)
        else:   self.small_bwd += (1 if plen <= 100 else 0)

        # payload entropy (first K bytes per dir)
        raw = pkt.getlayer(Raw)
        if raw and hasattr(raw, "load"):
            data = bytes(raw.load)
            if fwd: self.ent_fwd.ingest(data)
            else:   self.ent_bwd.ingest(data)

        # TCP flags & SYN fingerprint
        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            if flags & 0x02: self.syn_cnt += 1
            if flags & 0x10: self.ack_cnt += 1
            if flags & 0x01: self.fin_cnt += 1
            if flags & 0x04: self.rst_cnt += 1
            if flags & 0x08: self.psh_cnt += 1
            if flags & 0x20: self.urg_cnt += 1
            if flags & 0x40: self.ece_cnt += 1
            if flags & 0x80: self.cwr_cnt += 1

            if (flags & 0x01) or (flags & 0x04):
                self.seen_fin_rst = True

            # first forward SYN (no ACK) → options
            if fwd and (flags & 0x02) and not (flags & 0x10) and self.tcp_syn_opt_len == 0:
                opts = pkt[TCP].options or []
                self.tcp_syn_opt_len = len(opts)
                for name, val in opts:
                    if name == "MSS": self.tcp_syn_mss = int(val)
                    elif name == "WScale": self.tcp_syn_wscale = int(val)
                    elif name == "SAckOK": self.tcp_syn_sack_perm = 1

        # multicast/broadcast hints
        self.saw_multicast |= is_multicast_ip(dst_ip)
        self.saw_broadcast_ip |= (dst_ip == "255.255.255.255")
        eth = pkt.getlayer(Ether)
        if eth and getattr(eth, "dst", "").lower() == "ff:ff:ff:ff:ff:ff":
            self.saw_broadcast_l2 = True

    def idle_timeout(self):
        return TCP_IDLE_S if self.proto == 6 else UDP_IDLE_S

    def should_evict(self, now_ts):
        if self.proto == 6 and self.seen_fin_rst:
            return True
        if (now_ts - self.last) >= self.idle_timeout():
            return True
        if (self.last - self.first) >= MAX_AGE_S:
            return True
        return False

def features_from_state(st: FlowState):
    dur = max(st.last - st.first, 0.0)
    pkts_tot = st.pkts_fwd + st.pkts_bwd
    bytes_tot = st.bytes_fwd + st.bytes_bwd
    pps = pkts_tot / max(dur, 1e-3)
    bps = bytes_tot / max(dur, 1e-3)
    down_up_pkt_ratio  = (st.pkts_bwd / max(st.pkts_fwd, 1)) if (st.pkts_fwd or st.pkts_bwd) else 0.0
    down_up_byte_ratio = (st.bytes_bwd / max(st.bytes_fwd, 1)) if (st.bytes_fwd or st.bytes_bwd) else 0.0

    pf_min, pf_mean, pf_max, pf_std = st.len_fwd.stats()
    pb_min, pb_mean, pb_max, pb_std = st.len_bwd.stats()
    if_min, if_mean, if_max, if_std = st.iat_fwd.stats()
    ib_min, ib_mean, ib_max, ib_std = st.iat_bwd.stats()
    it_min, it_mean, it_max, it_std = st.iat_tot.stats()

    small_pkt_ratio_fwd = (st.small_fwd / max(st.pkts_fwd, 1)) if st.pkts_fwd else 0.0
    small_pkt_ratio_bwd = (st.small_bwd / max(st.pkts_bwd, 1)) if st.pkts_bwd else 0.0
    iat_cv_fwd = (if_std / if_mean) if if_mean > 0 else 0.0
    iat_cv_bwd = (ib_std / ib_mean) if ib_mean > 0 else 0.0

    (ip_lo, port_lo), (ip_hi, port_hi), proto = st.a, st.b, st.proto
    f_src_ip   = ip_lo if st.forward_is_a else ip_hi
    f_src_port = port_lo if st.forward_is_a else port_hi
    f_dst_ip   = ip_hi if st.forward_is_a else ip_lo
    f_dst_port = port_hi if st.forward_is_a else port_lo

    internal_dst = 1 if is_internal(f_dst_ip) else 0
    iot_sid = iot_service_id(proto, port_lo, port_hi)
    disc_id = discovery_proto_id(proto, f_dst_port)
    top_dport_iot = iot_sid  # reuse mapping unless you build a dedicated top-N vocab

    row = {
        # identifiers (for joins/debug; drop from model inputs)
        "start_ts": st.first, "end_ts": st.last,
        "src_ip": f_src_ip, "dst_ip": f_dst_ip,
        "sport": f_src_port, "dport": f_dst_port,
        "proto": {6:"tcp", 17:"udp"}.get(proto, "other"),
        "internal_dst": internal_dst,

        # core numerics
        "dur": dur,
        "pkts_fwd": st.pkts_fwd, "pkts_bwd": st.pkts_bwd, "pkts_tot": pkts_tot,
        "bytes_fwd": st.bytes_fwd, "bytes_bwd": st.bytes_bwd, "bytes_tot": bytes_tot,
        "pktlen_fwd_min": pf_min, "pktlen_fwd_mean": pf_mean, "pktlen_fwd_max": pf_max, "pktlen_fwd_std": pf_std,
        "pktlen_bwd_min": pb_min, "pktlen_bwd_mean": pb_mean, "pktlen_bwd_max": pb_max, "pktlen_bwd_std": pb_std,
        "iat_fwd_min": if_min, "iat_fwd_mean": if_mean, "iat_fwd_max": if_max, "iat_fwd_std": if_std,
        "iat_bwd_min": ib_min, "iat_bwd_mean": ib_mean, "iat_bwd_max": ib_max, "iat_bwd_std": ib_std,
        "iat_tot_min": it_min, "iat_tot_mean": it_mean, "iat_tot_max": it_max, "iat_tot_std": it_std,
        "pps": pps, "bps": bps,
        "down_up_pkt_ratio": down_up_pkt_ratio, "down_up_byte_ratio": down_up_byte_ratio,

        # TCP flags
        "syn_cnt": st.syn_cnt, "ack_cnt": st.ack_cnt, "fin_cnt": st.fin_cnt, "rst_cnt": st.rst_cnt,
        "psh_cnt": st.psh_cnt, "urg_cnt": st.urg_cnt, "ece_cnt": st.ece_cnt, "cwr_cnt": st.cwr_cnt,

        # payload bytes stats
        "payload_entropy_fwd": st.ent_fwd.entropy(),
        "payload_entropy_bwd": st.ent_bwd.entropy(),
        "payload_nonzero_frac_fwd": st.ent_fwd.nonzero_frac(),
        "payload_nonzero_frac_bwd": st.ent_bwd.nonzero_frac(),

        # presence flags
        "has_fwd": st.has_fwd, "has_bwd": st.has_bwd,

        # port buckets
        "sport_bucket": port_bucket(f_src_port),
        "dport_bucket": port_bucket(f_dst_port),

        # IoT additions
        "is_multicast_dst": 1 if st.saw_multicast else 0,
        "is_broadcast_dst": 1 if (st.saw_broadcast_ip or st.saw_broadcast_l2) else 0,
        "iot_service_id": iot_sid,
        "discovery_proto_id": disc_id,
        "iat_cv_fwd": iat_cv_fwd, "iat_cv_bwd": iat_cv_bwd,
        "small_pkt_ratio_fwd": small_pkt_ratio_fwd, "small_pkt_ratio_bwd": small_pkt_ratio_bwd,
        "ttl_fwd_first": st.ttl_fwd_first,

        # TLS/QUIC hints (placeholders unless you parse)
        "tls_clienthello_ext_count": st.tls_ext_cnt,
        "tls_cipher_count": st.tls_cipher_cnt,
        "tls_sni_len": st.tls_sni_len,
        "ja3_id_hashed": "NA",

        # TCP SYN fingerprint
        "tcp_syn_mss": st.tcp_syn_mss,
        "tcp_syn_wscale": st.tcp_syn_wscale,
        "tcp_syn_sack_perm": st.tcp_syn_sack_perm,
        "tcp_syn_opt_len": st.tcp_syn_opt_len,

        # IoT port vocab (reuse iot_service_id here)
        "top_dport_iot": top_dport_iot,
    }
    return row

def run(input_pcap: str, output_csv: str): # Streaming
    flows: dict[tuple, FlowState] = {}
    buffer = []
    writer = None
    fieldnames = None
    pkt_count = 0
    header_needed = not (os.path.exists(output_csv) and os.path.getsize(output_csv) > 0)

    with PcapReader(input_pcap) as pcap, open(output_csv, "a", newline="") as fout:
        for pkt in pcap:
            pkt_count += 1
            ts = float(getattr(pkt, "time", 0.0))

            k = five_tuple(pkt)
            if not k:
                continue
            ck = canonize(k)

            st = flows.get(ck)
            if st is None:
                st = FlowState(ck, ts, first_src_ip=k[0])
                flows[ck] = st

            st.update(pkt)

            # Evict on FIN/RST (TCP) or idle/max-age
            if st.should_evict(ts):
                row = features_from_state(st)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    if header_needed:
                        writer.writeheader()
                        header_needed = False
                buffer.append(row)
                del flows[ck]

            # Periodic sweep + batch write
            if (pkt_count % SWEEP_EVERY_PKTS) == 0:
                stale = []
                for key, fs in flows.items():
                    if (ts - fs.last) >= fs.idle_timeout() or (fs.last - fs.first) >= MAX_AGE_S:
                        stale.append(key)
                for key in stale:
                    row = features_from_state(flows[key])
                    if writer is None:
                        fieldnames = list(row.keys())
                        writer = csv.DictWriter(fout, fieldnames=fieldnames)
                        if header_needed:
                            writer.writeheader()
                            header_needed = False
                    buffer.append(row)
                    del flows[key]
                if writer and len(buffer) >= BATCH_ROWS:
                    writer.writerows(buffer); buffer.clear()

        # EOF: finalize remaining flows
        for key, fs in list(flows.items()):
            row = features_from_state(fs)
            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                if header_needed:
                    writer.writeheader()
                    header_needed = False
            buffer.append(row)
            del flows[key]
        if writer and buffer:
            writer.writerows(buffer); buffer.clear()


def main():
    os.makedirs(PREPROCESSED_DATA_DIRECTORY)
    for device in os.listdir(RAW_DATA_DIRECTORY):
        data_dir = Path(f"{RAW_DATA_DIRECTORY}/{device}")
        for pcap_path in data_dir.rglob("*.pcap"): 
            run(str(pcap_path), f"{PREPROCESSED_DATA_DIRECTORY}/{device}.csv")

if __name__ == "__main__":
    print('running...')
    main()
