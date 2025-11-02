import random 
from config import TEST_FRACTION
from pandas.errors import EmptyDataError

def evaluate_on_fixed_unseen(unseen_dataset, predict_func): 
    total = 0
    correct = 0
    for device_name, pcap_list in unseen_dataset.items():
        if not pcap_list:
            continue
        # print(f"Evaluating {device_name}: {len(pcap_list)} pcaps")
        n_samples = max(1, int(len(pcap_list) * TEST_FRACTION))
        sampled_pcaps = random.sample(pcap_list, n_samples)
        for pcap_path in sampled_pcaps:
            try:
                prediction = predict_func(str(pcap_path))
                # print("device name:", device_name)
                # print("prediction:", prediction)
            except EmptyDataError:
                continue
            if prediction == device_name:
                correct += 1
            total += 1
    acc = correct / total if total > 0 else 0
    print("Accuracy:", acc)
    return acc