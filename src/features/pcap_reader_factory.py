from contextlib import contextmanager
from scapy.all import PcapReader

@contextmanager
def _iter_context(it):
    yield it

class PcapReaderFactory:
    def __new__(cls, data):
        if isinstance(data, str) and data.endswith(".pcap"):
            return PcapReader(data)
        elif hasattr(data, "__iter__"):
            return _iter_context(iter(data))
        else:
            raise ValueError("Invalid input type")