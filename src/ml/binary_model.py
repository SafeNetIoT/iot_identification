import os
import pandas as pd
from config import RAW_DATA_DIRECTORY, SESSION_CACHE_PATH
import pandas as pd
import os
from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord
from pathlib import Path
from collections import defaultdict
import random
import zarr

class BinaryModel(Manager):
    """Trains one binary classifier per device (device vs all others)."""

    def __init__(self, architecture_name="standard_forest", manager_name="binary_model", output_directory=None, loading_dir=None):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name, output_directory=output_directory, loading_directory=loading_dir)
        self.data_path = Path(RAW_DATA_DIRECTORY)
        self.cache_path = Path(SESSION_CACHE_PATH)
        self.device_sessions = defaultdict(list)
        random.seed(self.random_state)
        self.prepare_sessions()

    def save_session(self):
        root = zarr.open(self.cache_path / "sessions.zarr", mode="w")
        for device, sessions in self.device_sessions.items():
            group = root.create_group(device)
            for i, df in enumerate(sessions):
                group.create_dataset(f"session_{i:05d}", data=df.to_records(index=False))

    def load_sessions(self):
        root = zarr.open(self.cache_path / "sessions.zarr", mode="r")
        sessions = {}
        for device in root.group_keys():
            device_group = root[device]
            dfs = [pd.DataFrame(ds[:]) for ds in device_group.values()]
            sessions[device] = dfs
        return sessions
    
    def prepare_sessions(self):
        self.cache_path = self.cache_path
        if self.cache_path.exists() and any(self.cache_path.iterdir()):
            self.device_sessions = self.load_sessions()
            return
            
        data_directory = Path(self.data_directory)
        for device_pcap in data_directory.rglob("*.pcap"):
            device_name = device_pcap.parent.parent.name
            unlabeled_device_df = self.fast_extractor.extract_features(str(device_pcap))
            if unlabeled_device_df.empty:
                continue
            labeled_df = self.data_prep.label_device(unlabeled_device_df, 0)
            labeled_df.attrs['pcap_path'] = str(device_pcap)
            self.device_sessions[device_name].append(labeled_df)
        self.save_session()

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

    def prepare_datasets(self, verbose=False):
        for device_name in self.device_sessions:
            true_class = self.prepare_true_class(device_name)
            true_class_num_sessions = len(self.device_sessions[device_name])
            records_per_session = max(1, true_class_num_sessions // max(1, len(self.device_sessions) - 1))
            false_class = self.sample_false_class(device_name, records_per_session)
            data = true_class + false_class
            if verbose:
                print(device_name, f"true class length: {len(true_class)}, false class length: {len(false_class)}")
            record = ModelRecord(name=device_name, data=data)
            self.records.append(record)

    def add_device(self, device_name, device_directory, verbose=False):
        device_path = Path(device_directory)
        if not device_path.exists():
            raise FileNotFoundError(f"Device directory not found: {device_path}")

        pcap_files = list(device_path.rglob("*.pcap"))
        true_class = []
        for pcap_path in pcap_files:
            session = self.fast_extractor.extract_features(str(pcap_path))
            if session.empty:
                continue
            session = self.data_prep.label_device(session, 1)
            true_class.append(session)

        false_class = self.sample_false_class(device_name, len(true_class))
        dataset = true_class + false_class
        if verbose:
            print(device_name, f"true class length: {len(true_class)}, false class length: {len(false_class)}")
        record = ModelRecord(device_name, dataset)
        self.records.append(record)
        self.train_classifier(record, show_curve=True)
        self.save_classifier(record)

def main():
    # manager = BinaryModel()
    # manager.add_device("alexa2", "data/raw/alexa_swan_kettle")

    manager = BinaryModel()
    manager.prepare_datasets()
    manager.train_all()
    manager.save_all()

    # manager = BinaryModel(output_directory="models/2025-10-21/binary_model2")
    # manager.predict("data/raw/alexa_swan_kettle/2023-10-19/2023-10-19_00:31:44.397s.pcap")


if __name__ == "__main__":
    main()
