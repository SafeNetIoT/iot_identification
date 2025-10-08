import csv
import math
import os
from pathlib import Path
import pandas as pd
from typing import Optional
from config import RAW_DATA_DIRECTORY, PREPROCESSED_DATA_DIRECTORY, K_PAYLOAD_BYTES, TCP_IDLE_S, UDP_IDLE_S, MAX_AGE_S, SWEEP_EVERY_PKTS, BATCH_ROWS

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from src.utils import is_internal, is_multicast_ip, port_bucket, iot_service_id, discovery_proto_id, five_tuple, canonize, flatten
from identification.features.flow_structs import *

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

class FlowStateFactory:
    @staticmethod
    def create(ck, ts, first_src_ip):
        fs = FlowState(ck, ts, first_src_ip)

        FlowStateFactory._set_forward_direction(fs, first_src_ip)
        FlowStateFactory._init_counters(fs)
        FlowStateFactory._init_stats(fs)
        FlowStateFactory._init_flags(fs)
        FlowStateFactory._init_entropy(fs)
        FlowStateFactory._init_misc(fs)
        FlowStateFactory._init_tls(fs)

        return fs

    @staticmethod
    def _set_forward_direction(fs, first_src_ip):
        a_ip, b_ip = fs.a[0], fs.b[0]
        if is_internal(a_ip) ^ is_internal(b_ip):
            fs.forward_is_a = is_internal(a_ip)
        else:
            fs.forward_is_a = (first_src_ip == a_ip)

    @staticmethod
    def _init_counters(fs):
        fs.pkts_fwd = fs.pkts_bwd = 0
        fs.bytes_fwd = fs.bytes_bwd = 0
        fs.small_fwd = fs.small_bwd = 0
        fs.has_fwd = fs.has_bwd = 0

    @staticmethod
    def _init_stats(fs):
        fs.len_fwd, fs.len_bwd = OnlineStats(), OnlineStats()
        fs.iat_fwd, fs.iat_bwd, fs.iat_tot = OnlineStats(), OnlineStats(), OnlineStats()
        fs.last_ts_fwd = fs.last_ts_bwd = fs.last_ts_any = None

    @staticmethod
    def _init_flags(fs):
        fs.syn_cnt = fs.ack_cnt = fs.fin_cnt = fs.rst_cnt = 0
        fs.psh_cnt = fs.urg_cnt = fs.ece_cnt = fs.cwr_cnt = 0
        fs.seen_fin_rst = False

    @staticmethod
    def _init_entropy(fs):
        fs.ent_fwd, fs.ent_bwd = (
            EntropyCounter(K_PAYLOAD_BYTES),
            EntropyCounter(K_PAYLOAD_BYTES),
        )

    @staticmethod
    def _init_misc(fs):
        fs.ttl_fwd_first = 0
        fs.saw_multicast = fs.saw_broadcast_l2 = fs.saw_broadcast_ip = False

    @staticmethod
    def _init_tls(fs):
        fs.tls_ext_cnt = fs.tls_cipher_cnt = fs.tls_sni_len = 0
        fs.tcp_syn_mss = fs.tcp_syn_wscale = fs.tcp_syn_sack_perm = fs.tcp_syn_opt_len = 0

class FlowState:
    """
    Everything needed to compute features for a single conversation (flow).
    Key = ((ip_lo,port_lo),(ip_hi,port_hi),proto)
    """
    __slots__ = (
        "a","b","proto","first","last","first_src_ip","forward_is_a",
        "pkts_fwd","pkts_bwd","bytes_fwd","bytes_bwd",
        "len_fwd","len_bwd",
        "iat_fwd","iat_bwd","iat_tot","last_ts_fwd","last_ts_bwd","last_ts_any",
        "small_fwd","small_bwd",
        "syn_cnt","ack_cnt","fin_cnt","rst_cnt","psh_cnt","urg_cnt","ece_cnt","cwr_cnt",
        "ent_fwd","ent_bwd",
        "has_fwd","has_bwd",
        "ttl_fwd_first","saw_multicast","saw_broadcast_l2","saw_broadcast_ip",
        "tls_ext_cnt","tls_cipher_cnt","tls_sni_len",
        "tcp_syn_mss","tcp_syn_wscale","tcp_syn_sack_perm","tcp_syn_opt_len",
        "seen_fin_rst",
    )

    def __init__(self, ck, ts, first_src_ip):
        self.a, self.b, self.proto = ck
        self.first = self.last = ts
        self.first_src_ip = first_src_ip
        self.forward_is_a = None  # set later by factory

    def _is_fwd_ip(self, pkt_src_ip: str) -> bool:
        return (pkt_src_ip == self.a[0]) if self.forward_is_a else (pkt_src_ip == self.b[0])

    def _update_bounds(self, ts: float):
        self.first = min(self.first, ts)
        self.last = max(self.last, ts)

    def _update_iat(self, ts: float):
        if self.last_ts_any is not None and ts >= self.last_ts_any:
            self.iat_tot.update(ts - self.last_ts_any)
        self.last_ts_any = ts

    def _update_lengths(self, pkt, ts: float, ip, fwd: bool):
        plen = int(len(pkt))
        if fwd:
            self.has_fwd = 1
            self.pkts_fwd += 1
            self.bytes_fwd += plen
            self.len_fwd.update(plen)
            if self.last_ts_fwd is not None and ts >= self.last_ts_fwd:
                self.iat_fwd.update(ts - self.last_ts_fwd)
            self.last_ts_fwd = ts
            if self.ttl_fwd_first == 0:
                ttl = int(getattr(ip, "ttl", getattr(ip, "hlim", 0)) or 0)
                self.ttl_fwd_first = ttl
        else:
            self.has_bwd = 1
            self.pkts_bwd += 1
            self.bytes_bwd += plen
            self.len_bwd.update(plen)
            if self.last_ts_bwd is not None and ts >= self.last_ts_bwd:
                self.iat_bwd.update(ts - self.last_ts_bwd)
            self.last_ts_bwd = ts

    def _update_small_pkt(self, pkt, fwd: bool):
        plen = int(len(pkt))
        if plen <= 100:
            if fwd:
                self.small_fwd += 1
            else:
                self.small_bwd += 1

    def _update_entropy(self, pkt, fwd: bool):
        raw = pkt.getlayer(Raw)
        if raw and hasattr(raw, "load"):
            data = bytes(raw.load)
            if fwd:
                self.ent_fwd.ingest(data)
            else:
                self.ent_bwd.ingest(data)

    def _update_tcp(self, pkt, fwd: bool):
        if TCP not in pkt:
            return
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

    def _update_multicast_broadcast(self, dst_ip: str, pkt):
        self.saw_multicast |= is_multicast_ip(dst_ip)
        self.saw_broadcast_ip |= (dst_ip == "255.255.255.255")
        eth = pkt.getlayer(Ether)
        if eth and getattr(eth, "dst", "").lower() == "ff:ff:ff:ff:ff:ff":
            self.saw_broadcast_l2 = True

    def update(self, pkt):
        ts = float(getattr(pkt, "time", 0.0))
        self._update_bounds(ts)

        ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
        if ip is None:
            return

        src_ip, dst_ip = str(ip.src), str(ip.dst)
        fwd = self._is_fwd_ip(src_ip)

        self._update_iat(ts)
        self._update_lengths(pkt, ts, ip, fwd)
        self._update_small_pkt(pkt, fwd)
        self._update_entropy(pkt, fwd)
        self._update_tcp(pkt, fwd)
        self._update_multicast_broadcast(dst_ip, pkt)

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

class FeatureExtractor:
    def __init__(self, state: FlowState) -> None:
        self.state = state

    def get_byte_ratios(self):
        down_up_pkt_ratio  = (self.state.pkts_bwd / max(self.state.pkts_fwd, 1)) if (self.state.pkts_fwd or self.state.pkts_bwd) else 0.0
        down_up_byte_ratio = (self.state.bytes_bwd / max(self.state.bytes_fwd, 1)) if (self.state.bytes_fwd or self.state.bytes_bwd) else 0.0
        return down_up_pkt_ratio, down_up_byte_ratio

    def get_packet_length_stats(self):
        forward = self.state.len_fwd.stats()
        backward = self.state.len_bwd.stats()
        return forward, backward

    def get_iat_stats(self):
        forward = self.state.iat_fwd.stats()
        backwards = self.state.iat_bwd.stats()
        total = self.state.iat_tot.stats()
        return forward, backwards, total      

    def get_small_pkt_ratios(self):
        small_pkt_ratio_fwd = (self.state.small_fwd / max(self.state.pkts_fwd, 1)) if self.state.pkts_fwd else 0.0
        small_pkt_ratio_bwd = (self.state.small_bwd / max(self.state.pkts_bwd, 1)) if self.state.pkts_bwd else 0.0
        return small_pkt_ratio_fwd, small_pkt_ratio_bwd

    def get_endpoints(self):
        (ip_lo, port_lo), (ip_hi, port_hi), proto = self.state.a, self.state.b, self.state.proto
        f_src_ip   = ip_lo if self.state.forward_is_a else ip_hi
        f_src_port = port_lo if self.state.forward_is_a else port_hi
        f_dst_ip   = ip_hi if self.state.forward_is_a else ip_lo
        f_dst_port = port_hi if self.state.forward_is_a else port_lo
        return (port_lo, port_hi), (f_src_ip, f_src_port), (f_dst_ip, f_dst_port), proto

    def features_from_state(self):
        dur = max(self.state.last - self.state.first, 0.0)
        pkts_tot = self.state.pkts_fwd + self.state.pkts_bwd
        bytes_tot = self.state.bytes_fwd + self.state.bytes_bwd
        pps = pkts_tot / max(dur, 1e-3)
        bps = bytes_tot / max(dur, 1e-3)
        down_up_pkt_ratio, down_up_byte_ratio = self.get_byte_ratios()
        
        (pf_min, pf_mean, pf_max, pf_std), (pb_min, pb_mean, pb_max, pb_std) = self.get_packet_length_stats()
        (if_min, if_mean, if_max, if_std),(ib_min, ib_mean, ib_max, ib_std), (it_min, it_mean, it_max, it_std) = self.get_iat_stats()

        small_pkt_ratio_fwd, small_pkt_ratio_bwd = self.get_small_pkt_ratios()
        iat_cv_fwd = (if_std / if_mean) if if_mean > 0 else 0.0
        iat_cv_bwd = (ib_std / ib_mean) if ib_mean > 0 else 0.0

        (port_lo, port_hi), (f_src_ip, f_src_port), (f_dst_ip, f_dst_port), proto = self.get_endpoints()

        internal_dst = 1 if is_internal(f_dst_ip) else 0
        iot_sid = iot_service_id(proto, port_lo, port_hi)
        disc_id = discovery_proto_id(proto, f_dst_port)
        top_dport_iot = iot_sid  # reuse mapping unless you build a dedicated top-N vocab

        identifiers = Identifiers(
            start_ts=self.state.first,
            end_ts=self.state.last,
            src_ip=f_src_ip,
            dst_ip=f_dst_ip,
            sport=f_src_port,
            dport=f_dst_port,
            proto = {6:"tcp", 17:"udp"}.get(proto, "other"),
            internal_dst=internal_dst
        )
        
        packet_stats = PacketsStats(
            pkts_fwd=self.state.pkts_fwd,
            pkts_bwd=self.state.pkts_bwd,
            pkts_tot=pkts_tot
        )

        bytes_stats = BytesStats(
            bytes_fwd=self.state.bytes_fwd,
            bytes_bwd=self.state.bytes_bwd,
            bytes_tot=bytes_tot
        )

        packet_length_stats = PacketLengthStats(
            pktlen_fwd_min=pf_min,
            pktlen_fwd_mean=pf_mean,
            pktlen_fwd_max=pf_max,
            pktlen_fwd_std=pf_std,
            pktlen_bwd_min=pb_min,
            pktlen_bwd_mean=pb_mean,
            pktlen_bwd_max=pb_max,
            pktlen_bwd_std=pb_std
        )

        iat_stats = IATStats(
            iat_fwd_min=if_min,
            iat_fwd_mean=if_mean,
            iat_fwd_max=if_max,
            iat_fwd_std=if_std,
            iat_bwd_min=ib_min,
            iat_bwd_mean=ib_mean,
            iat_bwd_max=ib_max,
            iat_bwd_std=ib_std,
            iat_tot_min=it_min,
            iat_tot_mean=it_mean,
            iat_tot_max=it_max,
            iat_tot_std=it_std
        )

        ratios = Ratios(
            down_up_byte_ratio=down_up_byte_ratio,
            down_up_pkt_ratio=down_up_pkt_ratio
        )

        tcp_flags = TcpFlags(
            syn_cnt=self.state.syn_cnt,
            ack_cnt=self.state.ack_cnt,
            fin_cnt=self.state.fin_cnt,
            rst_cnt=self.state.rst_cnt,
            psh_cnt=self.state.psh_cnt,
            urg_cnt=self.state.urg_cnt,
            ece_cnt=self.state.ece_cnt,
            cwr_cnt=self.state.cwr_cnt,
        )

        payload_stats = PayloadStats(
            payload_entropy_fwd=self.state.ent_fwd.entropy(),
            payload_entropy_bwd=self.state.ent_bwd.entropy(),
            payload_nonzero_frac_fwd=self.state.ent_fwd.nonzero_frac(),
            payload_nonzero_frac_bwd=self.state.ent_bwd.nonzero_frac()
        )

        presence_flags = PresenceFlags(
            has_fwd=self.state.has_fwd,
            has_bwd=self.state.has_bwd
        )

        port_buckets = PortBuckets(
            sport_bucket=port_bucket(f_src_port),
            dport_bucket=port_bucket(f_dst_port)
        )

        iot_additions = IoTAdditions(
            is_multicast_dst = 1 if self.state.saw_multicast else 0,
            is_broadcast_dst = 1 if (self.state.saw_broadcast_ip or self.state.saw_broadcast_l2) else 0,
            iot_service_id=iot_sid,
            discovery_proto_id=disc_id,
            iat_cv_fwd=iat_cv_fwd,
            iat_cv_bwd=iat_cv_bwd,
            small_pkt_ratio_fwd=small_pkt_ratio_fwd,
            small_pkt_ratio_bwd=small_pkt_ratio_bwd,
            ttl_fwd_first=self.state.ttl_fwd_first
        )

        tls_hints = TLSHints(
            tls_clienthello_ext_count=self.state.tls_ext_cnt,
            tls_cipher_count=self.state.tls_cipher_cnt,
            tls_sni_len=self.state.tls_sni_len,
            ja3_id_hashed="NA"
        )

        iot_port_vocab = IoTPortVocab(
            tcp_syn_mss=self.state.tcp_syn_mss,
            tcp_syn_wscale=self.state.tcp_syn_wscale,
            tcp_syn_sack_perm=self.state.tcp_syn_sack_perm,
            tcp_syn_opt_len=self.state.tcp_syn_opt_len,
            top_dport_iot=top_dport_iot
        )

        features = Features(
            identifiers=identifiers,
            dur=dur,
            packets_stats=packet_stats,
            bytes_stats=bytes_stats,
            packet_length_stats=packet_length_stats,
            iat_stats=iat_stats,
            pps=pps,
            bps=bps,
            ratios=ratios,
            tcp_flags=tcp_flags,
            payload_stats=payload_stats,
            presence_flags=presence_flags,
            port_buckets=port_buckets,
            iot_additions=iot_additions,
            tls_hints=tls_hints,
            iot_port_vocab=iot_port_vocab
        )
        return features.to_flat_dict()

class FlowManager:
    def __init__(self):
        self.flows = {}

    def update_flow(self, pkt, ts):
        k = five_tuple(pkt)
        if not k:
            return None

        ck = canonize(k)
        st = self.flows.get(ck)

        if st is None:
            st = FlowStateFactory.create(ck, ts, first_src_ip=k[0])
            self.flows[ck] = st

        st.update(pkt)

        if st.should_evict(ts):
            return self.evict(ck)

        return None

    def evict(self, key):
        st = self.flows.pop(key, None)
        if st:
            return FeatureExtractor(st).features_from_state()
        return None

    def sweep(self, ts):
        stale_keys = [
            key for key, fs in self.flows.items()
            if (ts - fs.last) >= fs.idle_timeout()
            or (fs.last - fs.first) >= MAX_AGE_S
        ]
        rows = [self.evict(key) for key in stale_keys]
        return [r for r in rows if r]

class ExtractionPipeline:
    def __init__(self, time_interval: Optional[float] = None) -> None:
        self.time_interval = time_interval

    def extract_features(self, input_pcap: str) -> pd.DataFrame:
        manager = FlowManager()
        rows = []
        start_time = None
        cutoff = None

        with PcapReader(input_pcap) as pcap:
            for pkt_count, pkt in enumerate(pcap, 1):
                ts = float(getattr(pkt, "time", 0.0))

                if start_time is None:
                    start_time = ts
                    if self.time_interval is not None:
                        cutoff = start_time + self.time_interval

                if cutoff is not None and ts > cutoff:
                    break

                row = manager.update_flow(pkt, ts)
                if row:
                    rows.append(row)
                if pkt_count % SWEEP_EVERY_PKTS == 0:
                    rows.extend(manager.sweep(ts))
            rows.extend([manager.evict(k) for k in list(manager.flows.keys())])

        return pd.DataFrame(rows)

def main():
    os.makedirs(PREPROCESSED_DATA_DIRECTORY, exist_ok=True)
    extractor = ExtractionPipeline()
    for device in os.listdir(RAW_DATA_DIRECTORY):
        device_dfs = []
        data_dir = Path(f"{RAW_DATA_DIRECTORY}/{device}")
        for pcap_path in data_dir.rglob("*.pcap"): 
            device_df = extractor.extract_features(str(pcap_path))
            device_dfs.append(device_df)
        complete_device = pd.concat(device_dfs, ignore_index=True)
        complete_device.to_csv(f"{PREPROCESSED_DATA_DIRECTORY}/{device}.csv", index=False)

if __name__ == "__main__":
    print('running...')
    main()
