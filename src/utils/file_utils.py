import yaml
from config import settings

def unpack_feature_groups():
    with open(settings.feature_menu_path, "r") as f:
        data = yaml.safe_load(f)

    feature_groups = []
    for block in data:
        if "Feature" in block:
            feature_groups.append(block["Feature"])
    return feature_groups

def unpack_features():
    with open(settings.valid_features_directory, 'r') as file:
        features = [line.strip() for line in file]
    return features