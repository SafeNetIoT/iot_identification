import pytest
from src.features.fast_extraction import FastExtractionPipeline
from src.ml.binary_model import BinaryModel
from tests.helpers import get_mac_address_map
import os
from pathlib import Path
from config import settings
from pandas.errors import EmptyDataError

@pytest.mark.integration
def test_unsw(binary_model_under_test):
    extractor = FastExtractionPipeline()
    for pcap_file in os.listdir(settings.unsw_dataset_path):
        try:
            prediction = binary_model_under_test.predict(f"{settings.unsw_dataset_path}/{pcap_file}")
            print(prediction)
        except EmptyDataError:
            continue

        
    