from config import INTERNAL_NETS
from scapy.layers.inet import IP, TCP, UDP
import ipaddress
from scapy.layers.inet6 import IPv6

def is_internal(ip: str) -> bool:
    ip_obj = ipaddress.ip_address(ip) if not isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)) else ip
    return any(ip_obj in net for net in INTERNAL_NETS)

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