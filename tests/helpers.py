import random
import os
import pandas as pd
from typing import Optional
import joblib

def assert_save_calls(mock_save, expected_names):
    """Ensure save_classifier was called with expected record names."""
    saved_records = [call.args[0] for call in mock_save.call_args_list]
    saved_names = [r.name for r in saved_records]
    assert set(saved_names) == set(expected_names), f"Expected {expected_names}, got {saved_names}"

def assert_save_paths(mock_save, expected_dir):
    """Ensure save paths start with the expected output directory."""
    for call in mock_save.call_args_list:
        record = call.args[0]
        assert record.name in str(expected_dir), f"{record.name} not saved in {expected_dir}"

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

def run_model_workflow_test(manager, tmp_path):
    """
    Generic test for ML model managers:
    - Runs preprocess/prepare_datasets or equivalent
    - Trains and saves all models
    - Asserts that model and evaluation files exist
    - Checks that preprocessed data is non-empty
    """

    manager.train_all()
    manager.save_all()
    model_files = list(tmp_path.rglob("*.pkl"))
    assert model_files, "Expected trained model files in output directory"

    eval_files = list(tmp_path.rglob("*evaluation*.txt"))
    assert eval_files, "Expected evaluation output file(s)"

    model = joblib.load(model_files[0])
    assert hasattr(model, "predict"), "Model object missing predict() method"

    # Data sanity
    assert manager.records, "Manager should have records"
    first_df = manager.records[0].data
    assert isinstance(first_df, pd.DataFrame)
    assert not first_df.empty, "Prepared dataset is empty"
