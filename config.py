import json
from pathlib import Path

def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as file:
        return json.load(file)

CONFIG = load_config()
FEATURE_MENU_PATH = CONFIG["feature_menu_path"]
RAW_DATA_DIRECTORY = CONFIG['raw_data_directory']
PREPROCESSED_DATA_DIRECTORY = CONFIG['preprocessed_data_directory']
VALID_FEATURES_DIRECTORY = CONFIG['valid_features_directory']
MODELS_DIRECTORY = CONFIG['models_directory']
TIME_INTERVALS = CONFIG['time_intervals']