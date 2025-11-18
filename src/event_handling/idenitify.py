from scapy.all import sniff
from src.ml.binary_model import BinaryModel
from config import settings
import os

def dump_packets():
    pass

def remove_old_models():
    pass

def retrain(model: BinaryModel, device_name, device_data, fast = True):
    if fast:
        model.add_device(device_name, device_data)
    else:
        dump_packets()
        model.slow_train(memory=False)
    if len(os.listdir(settings.models_directory)) > 3:
        remove_old_models()

def identify(timeout = 180):
    packets = sniff(iface="eth0", timeout=timeout)
    model = BinaryModel(loading_dir=settings.model_under_test)
    device = model.predict(packets)
    if device is not None:
        return device
    retrain(model)
    
    



