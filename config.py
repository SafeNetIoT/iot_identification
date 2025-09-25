import json
from pathlib import Path

def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as file:
        return json.load(file)

CONFIG = load_config()
FEATURE_MENU_PATH = CONFIG["feature_menu_path"]