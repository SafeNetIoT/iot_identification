import random 
from config import settings
from pandas.errors import EmptyDataError
import json

def evaluate_on_fixed_unseen(model, verbose=True): 
    random.seed(settings.random_state)
    total = 0
    correct = 0
    accuracy_per_class = {}
    for device_name, pcap_list in model.unseen_sessions.items():
        if not pcap_list:
            continue
        if verbose:
            print(f"Evaluating {device_name}: {len(pcap_list)} pcaps")
        class_correct = 0
        n_samples = max(1, int(len(pcap_list) * settings.unseen_fraction))
        sampled_pcaps = random.sample(pcap_list, n_samples)
        for pcap_path in sampled_pcaps:
            try:
                prediction = model.predict(str(pcap_path))
                if verbose:
                    print("device name:", device_name)
                    print("prediction:", prediction)
            except EmptyDataError:
                continue
            if prediction == device_name:
                class_correct += 1
                correct += 1
            total += 1
        accuracy_per_class[device_name] = class_correct / len(pcap_list)
    acc = correct / total if total > 0 else 0
    print("Accuracy:", acc)
    model_id = str(model.loading_directory)
    if settings.is_ci:
        with open(settings.granular_test_results_path, 'w') as file:
            json.dump(accuracy_per_class, file, indent=2)
    else:
        redis = model.get_redis()
        redis.set_json(model_id, accuracy_per_class)
    return acc