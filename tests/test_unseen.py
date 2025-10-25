from config import MODEL_UNDER_TEST, DESIRED_ACCURACY
from src.ml.binary_model import BinaryModel
from pandas.errors import EmptyDataError


def test_unseen():
    correct, total = 0, 0
    manager = BinaryModel(loading_dir=MODEL_UNDER_TEST)
    for device_name, pcap_list in manager.unseen_sessions.items():
        print(device_name)
        for pcap_path in pcap_list[:3]:
            try:
                prediction = manager.predict(str(pcap_path))
                print("prediction:", prediction)
            except EmptyDataError:
                continue
            if prediction == device_name:
                correct += 1
            total += 1
    acc = correct / total
    print(acc)
    assert acc >= DESIRED_ACCURACY




    