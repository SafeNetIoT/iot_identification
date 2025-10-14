import random
import os
import pandas as pd
from typing import Optional
import joblib

def list_device_files(raw_dir: str) -> list[str]:
    """Return all .pcap files in the raw directory."""
    return [f for f in os.listdir(raw_dir) if f.endswith(".pcap")]


def sample_devices(device_files: list[str], frac: float = 0.1, seed: int = 42) -> list[str]:
    """Randomly select a subset of device files."""
    random.seed(seed)
    sample_size = max(1, int(frac * len(device_files)))
    return random.sample(device_files, sample_size)


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

    # --- Run workflow ---
    if hasattr(manager, "prepare_datasets"):
        manager.prepare_datasets()
    elif hasattr(manager, "preprocess"):
        manager.preprocess()
    else:
        raise AttributeError("Manager has no prepare_datasets or preprocess method")

    manager.train_all()
    manager.save_all(output_dir=tmp_path)

    # --- Assertions ---
    model_files = list(tmp_path.glob("*.pkl"))
    assert model_files, "Expected trained model files in output directory"

    # Optional: Load one model to ensure it’s valid
    model = joblib.load(model_files[0])
    assert hasattr(model, "predict"), "Model object missing predict() method"

    # Optional: Evaluation or metrics files
    eval_files = list(tmp_path.glob("*evaluation*.csv"))
    assert eval_files, "Expected evaluation output file(s)"

    # Data sanity
    assert manager.records, "Manager should have records"
    first_df = manager.records[0].data
    assert isinstance(first_df, pd.DataFrame)
    assert not first_df.empty, "Prepared dataset is empty"
