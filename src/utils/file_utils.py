import yaml
from config import settings
import os
from collections import defaultdict, Counter

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

# def count_sessions():
#     directory_path = settings.session_cache_path / "collection_times"
#     session_sets = defaultdict(set)
#     session_counter = Counter()
#     for collection_time in directory_path.iterdir():
#         for device in collection_time.iterdir():
#             for session_file in device.iterdir():
#                 session_id = int(session_file.stem.split("_")[1])
#                 device_name = device.name
#                 if session_id not in session_sets[device_name]:
#                     session_counter[device_name] += 1
#                     session_sets[device_name].add(session_id)
#     return session_counter

# def cnt_sess():
#     directory_path = settings.raw_data_directory
#     cnt = Counter()
#     for device in directory_path.iterdir():
#         device_name = device.name
#         for date in device.iterdir():
#             for session in date.iterdir():
#                 cnt[device_name] += 1
#     return cnt


if __name__ == "__main__":
    print(cnt_sess())