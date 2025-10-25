import json
from pathlib import Path
import ipaddress

def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as file:
        return json.load(file)

CONFIG = load_config()
FEATURE_MENU_PATH = CONFIG["feature_menu_path"]
RAW_DATA_DIRECTORY = CONFIG['raw_data_directory']
PREPROCESSED_DATA_DIRECTORY = CONFIG['preprocessed_data_directory']
FAST_EXTRACTION_DIRECTORY = CONFIG['fast_extraction_directory']
VALID_FEATURES_DIRECTORY = CONFIG['valid_features_directory']
MODELS_DIRECTORY = CONFIG['models_directory']
TIME_INTERVALS = CONFIG['time_intervals']
INTERNAL_NETS = [ipaddress.ip_network(ip) for ip in CONFIG['internal_nets']]
TCP_IDLE_S = CONFIG['tcp_idle_s']
UDP_IDLE_S = CONFIG['udp_idle_s']
MAX_AGE_S = CONFIG['max_age_s']
K_PAYLOAD_BYTES = CONFIG['k_payload_bytes']
BATCH_ROWS = CONFIG['batch_rows']
SWEEP_EVERY_PKTS = CONFIG['sweep_every_pkts']
MODEL_ARCHITECTURES = CONFIG['model_architectures']
RANDOM_STATE = CONFIG['random_state']
TEST_FRACTION = CONFIG['testing']['sample_fraction'] if CONFIG['testing']['fast_mode'] else 1
MODEL_UNDER_TEST = CONFIG['model_under_test']
MULTICLASS_MODEL_UNDER_TEST = CONFIG['multiclass_model_under_test']
SESSION_CACHE_PATH = CONFIG['session_cache_path']
DESIRED_ACCURACY = CONFIG['desired_accuracy']
UNSEEN_FRACTION = CONFIG['unseen_fraction']
