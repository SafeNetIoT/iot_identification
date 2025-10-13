from src.utils import is_internal, is_multicast_ip
from src.identification.features.stats import OnlineStats, EntropyCounter
from config import K_PAYLOAD_BYTES, MAX_AGE_S, TCP_IDLE_S, UDP_IDLE_S
from scapy.layers.inet import IP, TCP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from scapy.packet import Raw

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
