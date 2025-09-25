import json

def unpack_feature_groups(feature_menu_path):
    feature_groups = []
    with open(feature_menu_path, "r") as f:
        for line in f:
            if line.strip().startswith("Feature:"):
                features_str = line.split("Feature:")[1].strip()
                features = [f.strip() for f in features_str.split(",")]
                feature_groups.append(features)
    return feature_groups