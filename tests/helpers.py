import random
import os
import pandas as pd
from typing import Optional
from config import RANDOM_STATE, DESIRED_ACCURACY, TEST_FRACTION, MAC_ADDRESS_MAP_PATH
from pandas.errors import EmptyDataError
import json


def list_device_dirs(raw_dir: str) -> list[str]:
    """Return all .pcap files in the raw directory."""
    return [f for f in os.listdir(raw_dir)]

def sample_devices(device_dirs: list[str], frac: float = 0.1, seed: int = 42) -> list[str]:
    """Randomly select a subset of device files."""
    random.seed(seed)
    sample_size = max(1, int(frac * len(device_dirs)))
    return random.sample(device_dirs, sample_size)

def unpack_device(device_path):
    pcaps = []

    def dfs(current_path: str):
        if os.path.isfile(current_path):
            pcaps.append(current_path)
            return
        if os.path.isdir(current_path):
            for child in os.listdir(current_path):
                child_path = os.path.join(current_path, child)
                dfs(child_path)

    dfs(device_path)
    return pcaps

def validate_columns_consistent(df: pd.DataFrame, reference_cols: Optional[list[str]]) -> list[str]:
    """Check that DataFrame columns match reference; return reference columns if None."""
    if reference_cols is None:
        return list(df.columns)
    assert list(df.columns) == reference_cols, "Column mismatch detected"
    return reference_cols


def validate_row_count(df: pd.DataFrame, pcap_path: str):
    """Ensure number of rows in extracted features matches number of input conversations."""
    expected = count_input_conversations(pcap_path)
    assert len(df) == expected, f"Row count mismatch for {pcap_path}: expected {expected}, got {len(df)}"

def count_input_conversations(pcap_path: str) -> int:
    """Placeholder for conversation counting logic."""
    csv_path = pcap_path.replace(".pcap", ".csv")
    return len(pd.read_csv(csv_path))

def _run_unseen_evaluation(model, predict_func):
    random.seed(RANDOM_STATE)
    correct, total = 0, 0
    for device_name, pcap_list in model.unseen_sessions.items():
        if not pcap_list:
            continue
        print(f"Evaluating {device_name}: {len(pcap_list)} pcaps")
        n_samples = max(1, int(len(pcap_list) * TEST_FRACTION))
        sampled_pcaps = random.sample(pcap_list, n_samples)
        for pcap_path in sampled_pcaps:
            try:
                prediction = predict_func(str(pcap_path))
                print("device name:", device_name)
                print("prediction:", prediction)
            except EmptyDataError:
                continue
            if prediction == device_name:
                correct += 1
            total += 1
    acc = correct / total if total > 0 else 0
    print("Accuracy:", acc)
    assert acc >= DESIRED_ACCURACY, "Accuracy lower than desired"

def get_mac_address_map():
    with open(MAC_ADDRESS_MAP_PATH, 'r') as file:
        mac_address_map = json.load(file)
    return mac_address_map