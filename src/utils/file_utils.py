import yaml
from config import VALID_FEATURES_DIRECTORY, FEATURE_MENU_PATH

def unpack_feature_groups():
    with open(FEATURE_MENU_PATH, "r") as f:
        data = yaml.safe_load(f)

    feature_groups = []
    for block in data:
        if "Feature" in block:
            feature_groups.append(block["Feature"])
    return feature_groups

def unpack_features():
    with open(VALID_FEATURES_DIRECTORY, 'r') as file:
        features = [line.strip() for line in file]
    return features