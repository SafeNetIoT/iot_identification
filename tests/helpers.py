import random
import os
import pandas as pd
from typing import Optional
import joblib

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
