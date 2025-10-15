import pytest
import os
from config import RAW_DATA_DIRECTORY, TEST_FRACTION
from tests.helpers import list_device_dirs, sample_devices, validate_columns_consistent,validate_row_count, unpack_device
from src.features.feature_extraction import ExtractionPipeline
from src.features.fast_extraction import FastExtractionPipeline
import pandas as pd

@pytest.mark.parametrize("extractor_class", [ExtractionPipeline, FastExtractionPipeline])
def test_extraction(extractor_class):
    """Integration test for feature extraction across a subset of devices."""
    device_files = list_device_dirs(RAW_DATA_DIRECTORY)
    assert device_files, "No device files found in raw data directory."

    sampled_devices = sample_devices(device_files, frac=TEST_FRACTION)
    print(f"Testing {len(sampled_devices)} of {len(device_files)} devices.")

    reference_cols = None

    extractor = extractor_class()  # instantiate whichever pipeline we’re testing
    for device_file in sampled_devices:
        for pcap_path in unpack_device(device_file):
            df = extractor.extract_features(pcap_path)

            assert isinstance(df, pd.DataFrame), f"Output not DataFrame for {device_file}"
            assert not df.empty, f"No features extracted for {device_file}"

            reference_cols = validate_columns_consistent(df, reference_cols)
            validate_row_count(df, pcap_path)