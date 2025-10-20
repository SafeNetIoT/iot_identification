import os
import pandas as pd
from config import RAW_DATA_DIRECTORY
import pandas as pd
import os
from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord
from pathlib import Path
from collections import defaultdict
import random

class BinaryModel(Manager):
    """Trains one binary classifier per device (device vs all others)."""

    def __init__(self, architecture_name="standard_forest", manager_name="binary_model", output_directory=None):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name, output_directory=output_directory)
        self.data_directory = RAW_DATA_DIRECTORY
        self.device_sessions = defaultdict(list)
        random.seed(self.random_state)

    def prepare_sessions(self):
        data_directory = Path(self.data_directory)
        for device_pcap in data_directory.rglob("*.pcap"):
            device_name = device_pcap.parent.parent.name
            unlabeled_device_df = self.fast_extractor.extract_features(str(device_pcap))
            if unlabeled_device_df.empty:
                continue
            labeled_df = self.data_prep.label_device(unlabeled_device_df, 0)
            labeled_df.attrs['pcap_path'] = str(device_pcap)
            self.device_sessions[device_name].append(labeled_df)
            if not os.path.exists(f"extracted_features/{device_name}.vsv"):
                labeled_df.to_csv(f"extracted_features/{device_name}.csv", index=False)

    def sample_false_class(self, current_device_name, sessions_per_class):
        sampled_dfs = []
        for device_name, session_list in self.device_sessions.items():
            if device_name == current_device_name:
                continue
            sample_size = min(sessions_per_class, len(session_list))
            sampled = random.sample(session_list, sample_size)
            sampled_dfs.extend(sampled)
        return sampled_dfs
            
    def prepare_true_class(self, current_device_name):
        true_class = []
        for session in self.device_sessions[current_device_name]:
            labeled_df = self.data_prep.label_device(session, 1)
            true_class.append(labeled_df)
        return true_class

    def prepare_datasets(self):
        for device_name in self.device_sessions:
            true_class = self.prepare_true_class(device_name)
            true_class_num_sessions = len(self.device_sessions[device_name])
            records_per_session = max(1, true_class_num_sessions // max(1, len(self.device_sessions) - 1))
            false_class = self.sample_false_class(device_name, records_per_session)
            print("true class length:", len(true_class))
            print("false class length:", len(false_class))
            print()
            data = true_class + false_class
            record = ModelRecord(name=device_name, data=data)
            self.records.append(record)
        print(len(self.records))

    def add_device(self, device_name, device_directory):
        device_path = Path(device_directory)
        if not device_path.exists():
            raise FileNotFoundError(f"Device directory not found: {device_path}")

        pcap_files = list(device_path.rglob("*.pcap"))
        true_class = []
        for pcap_path in pcap_files:
            session = self.fast_extractor.extract_features(str(pcap_path))
            session = self.data_prep.label_device(session, 1)
            session = self.data_prep.clean_up(session)
            if not session.empty:
                true_class.append(session)

        false_class = self.sample_false_class(device_name, len(true_class))
        dataset = true_class + false_class
        record = ModelRecord(device_name, dataset)
        self.records.append(record)

        self.train_classifier(record, show_curve=True)
        self.save_classifier(record)

def main():
    manager = BinaryModel()
    # manager.add_device("alexa2", "data/raw/alexa_swan_kettle/2023-10-19/2023-10-19_00:02:55.402s.pcap")

    manager.prepare_sessions()
    for key, val in manager.device_sessions.items():
        print(key, len(val))
        print()
    manager.prepare_datasets()
    manager.train_all()
    manager.save_all()

    # manager = BinaryModel(output_directory="models/2025-10-20/binary_model")
    # res = manager.predict("data/raw/alexa_swan_kettle/2023-10-19/2023-10-19_00:02:55.402s.pcap")
    # print(res)


if __name__ == "__main__":
    main()
