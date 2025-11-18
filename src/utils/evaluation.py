import json
import random
from typing import Dict, Tuple
from config import settings
from src.utils.exceptions import EmptyDataError

def compute_unseen_accuracy(
    model,
    unseen_sessions: Dict[str, list],
    unseen_fraction: float,
    random_state: int,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float]]:
    random.seed(random_state)
    total, correct = 0, 0
    accuracy_per_class = {}

    for device_name, pcap_list in unseen_sessions.items():
        if not pcap_list:
            continue

        if verbose:
            print(f"Evaluating {device_name}: {len(pcap_list)} pcaps")

        class_correct = 0
        n_samples = max(1, int(len(pcap_list) * unseen_fraction))
        sampled_pcaps = random.sample(pcap_list, n_samples)

        for pcap_path in sampled_pcaps:
            try:
                prediction = model.predict(str(pcap_path))
                if verbose:
                    print(f"{device_name} → {prediction}")
            except EmptyDataError:
                continue

            if prediction == device_name:
                class_correct += 1
                correct += 1
            total += 1

        accuracy_per_class[device_name] = class_correct / len(pcap_list)

    overall_acc = correct / total if total > 0 else 0
    return overall_acc, accuracy_per_class

def persist_accuracy_results(model_id: str, accuracy_per_class: Dict[str, float], redis_client=None) -> None:
    if settings.is_ci:
        print("::notice:: Saving granular test results to CI file...")
        with open(settings.granular_test_results_path, "w") as file:
            json.dump(accuracy_per_class, file, indent=2)
    else:
        if redis_client is None:
            raise ValueError("Redis client must be provided for local persistence.")
        print("::notice:: Saving granular test results to Redis...")
        redis_client.set_json(model_id, accuracy_per_class)

def evaluate_on_fixed_unseen(model, verbose: bool = True) -> float:
    print("::notice:: Starting unseen evaluation...")
    acc, accuracy_per_class = compute_unseen_accuracy(
        model=model,
        unseen_sessions=model.unseen_sessions,
        unseen_fraction=settings.unseen_fraction,
        random_state=settings.random_state,
        verbose=verbose,
    )

    print(f"::notice:: Overall accuracy: {acc:.3f}")
    model_id = str(model.loading_directory)

    if settings.is_ci:
        persist_accuracy_results(model_id, accuracy_per_class)
    else:
        redis_client = model.get_redis()
        persist_accuracy_results(model_id, accuracy_per_class, redis_client)

    return acc
