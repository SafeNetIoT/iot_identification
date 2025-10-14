import pytest
import os
from config import RAW_DATA_DIRECTORY
from tests.helpers import list_device_files, sample_devices, validate_columns_consistent,validate_row_count
from src.features.feature_extraction import ExtractionPipeline
import pandas as pd

@pytest.mark.integration
def test_extraction():
    """Integration test for feature extraction across a subset of devices."""
    device_files = list_device_files(RAW_DATA_DIRECTORY)
    assert device_files, "No device files found in raw data directory."

    sampled_devices = sample_devices(device_files, frac=0.1)
    print(f"Testing {len(sampled_devices)} of {len(device_files)} devices.")

    reference_cols = None

    extractor = ExtractionPipeline()
    for device_file in sampled_devices:
        pcap_path = os.path.join(RAW_DATA_DIRECTORY, device_file)
        df = extractor.extract_features(str(pcap_path))

        assert isinstance(df, pd.DataFrame), f"Output not DataFrame for {device_file}"
        assert not df.empty, f"No features extracted for {device_file}"

        reference_cols = validate_columns_consistent(df, reference_cols)
        validate_row_count(df, pcap_path)
