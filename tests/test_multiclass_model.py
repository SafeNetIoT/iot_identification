import random
import pytest
from config import RANDOM_STATE, TEST_FRACTION, DESIRED_ACCURACY
from pandas.errors import EmptyDataError

@pytest.mark.integration
def test_multiclass_unseen(multiclass_model_under_test):
    random.seed(RANDOM_STATE)
    correct, total = 0, 0
    for device_name, pcap_list in multiclass_model_under_test.unseen_sessions.items():
        if not pcap_list:
            continue
        print("pcap_list length:", len(pcap_list))
        n_samples = max(1, int(len(pcap_list) * TEST_FRACTION))
        sampled_pcaps = random.sample(pcap_list, n_samples)
        for pcap_path in sampled_pcaps:
            try:
                prediction = multiclass_model_under_test.multi_predict(str(pcap_path))
                print("device name:", device_name)
                print("prediction:", prediction)
            except EmptyDataError:
                continue
            if prediction == device_name:
                correct += 1
            total += 1
    acc = correct / total
    print(acc)
    assert acc >= DESIRED_ACCURACY, "Accuracy lower than desired"