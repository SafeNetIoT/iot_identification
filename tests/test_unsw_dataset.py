import pytest
from src.features.fast_extraction import FastExtractionPipeline
from src.ml.binary_model import BinaryModel
from tests.helpers import get_mac_address_map
import os
from pathlib import Path
from config import UNSW_DATASET_DIRECORY
from pandas.errors import EmptyDataError

@pytest.mark.integration
def test_unsw(binary_model_under_test):
    extractor = FastExtractionPipeline()
    for pcap_file in os.listdir(UNSW_DATASET_DIRECORY):
        try:
            prediction = binary_model_under_test.predict(f"{UNSW_DATASET_DIRECORY}/{pcap_file}")
            print(prediction)
        except EmptyDataError:
            continue

        
    