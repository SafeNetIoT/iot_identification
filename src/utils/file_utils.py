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

def count_sessions():
    directory_path = settings.session_cache_path / "collection_times"
    session_counter = {}
    for collection_time in directory_path.iterdir():
        for device_name in collection_time.iterdir():
            for session_file in device_name.iterdir():
                session_id = int(session_file.stem.split("_")[1])
                if session_id > session_counter.get(device_name, -1):
                    session_counter[device_name] = session_id
    return session_counter