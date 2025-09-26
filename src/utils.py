import json
from config import FEATURE_MENU_PATH
import yaml

def unpack_feature_groups():
    with open(FEATURE_MENU_PATH, "r") as f:
        data = yaml.safe_load(f)

    feature_groups = []
    for block in data:
        if "Feature" in block:
            feature_groups.append(block["Feature"])
    return feature_groups

if __name__ == "__main__":
    print(unpack_feature_groups())