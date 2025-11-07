from dataclasses import dataclass, field, asdict, is_dataclass, fields
from src.features.stats import OnlineStats, EntropyCounter
from config import settings

@dataclass
class DirectionStats:
    pkts: int = 0
    bytes: int = 0
    length: OnlineStats = field(default_factory=OnlineStats)
    iat: OnlineStats = field(default_factory=OnlineStats)
    last_ts: float = None
    small_pkts: int = 0
    entropy: EntropyCounter = field(default_factory=lambda: EntropyCounter(settings.k_payload_bytes))
    has_data: bool = 0

@dataclass
class TcpFlags:
    syn_cnt: int
    ack_cnt: int
    fin_cnt: int
    rst_cnt: int
    psh_cnt: int
    urg_cnt: int
    ece_cnt: int
    cwr_cnt: int


@dataclass
class TlsInfo:
    ext_cnt: int = 0
    cipher_cnt: int = 0
    sni_len: int = 0


@dataclass
class Identifiers:
    start_ts: float
    end_ts: float
    src_ip: str
    dst_ip: str
    sport: int
    dport: int
    proto: str
    internal_dst: int

@dataclass
class PacketsStats:
    pkts_fwd: int
    pkts_bwd: int
    pkts_tot: int

@dataclass
class BytesStats:
    bytes_fwd: int
    bytes_bwd: int
    bytes_tot: int

@dataclass
class PacketLengthStats:
    pktlen_fwd_min: float
    pktlen_fwd_mean: float
    pktlen_fwd_max: float
    pktlen_fwd_std: float
    pktlen_bwd_min: float
    pktlen_bwd_mean: float
    pktlen_bwd_max: float
    pktlen_bwd_std: float

@dataclass
class IATStats:
    iat_fwd_min: float
    iat_fwd_mean: float
    iat_fwd_max: float
    iat_fwd_std: float
    iat_bwd_min: float
    iat_bwd_mean: float
    iat_bwd_max: float
    iat_bwd_std: float
    iat_tot_min: float
    iat_tot_mean: float
    iat_tot_max: float
    iat_tot_std: float

@dataclass
class Ratios:
    down_up_pkt_ratio: float
    down_up_byte_ratio: float

@dataclass
class PayloadStats:
    payload_entropy_fwd: float
    payload_entropy_bwd: float
    payload_nonzero_frac_fwd: float
    payload_nonzero_frac_bwd: float

@dataclass
class PresenceFlags:
    has_fwd: int
    has_bwd: int

@dataclass
class PortBuckets:
    sport_bucket: str
    dport_bucket: str

@dataclass
class IoTAdditions:
    is_multicast_dst: int
    is_broadcast_dst: int
    iot_service_id: str
    discovery_proto_id: str
    iat_cv_fwd: float
    iat_cv_bwd: float
    small_pkt_ratio_fwd: float
    small_pkt_ratio_bwd: float
    ttl_fwd_first: int

@dataclass
class TLSHints:
    tls_clienthello_ext_count: int
    tls_cipher_count: int
    tls_sni_len: int
    ja3_id_hashed: str

@dataclass
class IoTPortVocab:
    tcp_syn_mss: int
    tcp_syn_wscale: int
    tcp_syn_sack_perm: int
    tcp_syn_opt_len: int
    top_dport_iot: str

@dataclass
class Features:
    identifiers: Identifiers
    dur: float
    packets_stats: PacketsStats
    bytes_stats: BytesStats
    packet_length_stats: PacketLengthStats
    iat_stats: IATStats
    pps: float
    bps: float
    ratios: Ratios
    tcp_flags: TcpFlags
    payload_stats: PayloadStats
    presence_flags: PresenceFlags
    port_buckets: PortBuckets
    iot_additions: IoTAdditions
    tls_hints: TLSHints
    iot_port_vocab: IoTPortVocab

    def to_flat_dict(self) -> dict:
        return flatten(self)
            
def flatten(obj, parent_key="", sep="_"):
    """
    Flattens nested dataclasses and dicts into a single flat dict.
    Drops all parent prefixes (so columns are flat like start_ts, src_ip, etc.).
    """
    items = {}

    if is_dataclass(obj):
        # Iterate directly over dataclass fields (do NOT use asdict)
        obj = {f.name: getattr(obj, f.name) for f in fields(obj)}

    for k, v in obj.items():
        # no prefix at all — always flatten into top level
        if is_dataclass(v) or isinstance(v, dict):
            items.update(flatten(v, sep=sep))
        else:
            items[k] = v
    return items