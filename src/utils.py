import json
from config import FEATURE_MENU_PATH

def unpack_feature_groups():
    feature_groups = []
    with open(FEATURE_MENU_PATH, "r") as f:
        for line in f:
            if line.strip().startswith("Feature:"):
                features_str = line.split("Feature:")[1].strip()
                features = [f.strip() for f in features_str.split(",")]
                feature_groups.append(features)
    return feature_groups