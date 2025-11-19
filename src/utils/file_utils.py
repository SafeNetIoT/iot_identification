import yaml
from config import settings
import os

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

def print_file_tree(root="."):
    # for debugging
    print(f"File tree for: {os.path.abspath(root)}")
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent} {os.path.basename(dirpath)}/")
        subindent = " " * 4 * (level + 1)
        for f in filenames:
            print(f"{subindent}- {f}")
