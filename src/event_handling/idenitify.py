from scapy.all import sniff
from src.ml.binary_model import BinaryModel
from config import settings

def identify(timeout = 180):
    packets = sniff(iface="eth0", timeout=timeout)
    model = BinaryModel(loading_dir=settings.model_under_test)
    device = model.predict(packets)
    if device is not None:
        return device
    



