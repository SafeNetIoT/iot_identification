import pytest
import os
from config import settings
from pandas.errors import EmptyDataError

@pytest.mark.integration
@pytest.mark.skipif(settings.is_ci, reason="Skip on CI")
def test_unsw(binary_model_under_test):
    for pcap_file in os.listdir(settings.unsw_dataset_path):
        try:
            prediction = binary_model_under_test.predict(f"{settings.unsw_dataset_path}/{pcap_file}")
            print(prediction)
        except EmptyDataError:
            continue

        
    