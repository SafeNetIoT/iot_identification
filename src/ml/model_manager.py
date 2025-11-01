import os
from src.ml.base_model import BaseModel
from config import MODEL_ARCHITECTURES, RANDOM_STATE, MODELS_DIRECTORY, RAW_DATA_DIRECTORY, SESSION_CACHE_PATH, UNSEEN_FRACTION, TIME_INTERVALS
from src.ml.dataset_preparation import DatasetPreparation
from typing import List
from datetime import datetime
import joblib
from src.ml.model_record import ModelRecord
from src.features.fast_extraction import FastExtractionPipeline
import pandas as pd
import zarr
from pathlib import Path
from collections import defaultdict, Counter
import random
import json

class Manager:
    def __init__(self, architecture_name="standard_forest", manager_name="random_forest", output_directory=None, loading_directory=None):
        self.architecture = MODEL_ARCHITECTURES[architecture_name]
        self.data_prep = DatasetPreparation()
        self.records: List[ModelRecord] = []
        self.random_state = RANDOM_STATE            
        self.output_directory = output_directory if output_directory is not None else MODELS_DIRECTORY
        self.loading_directory = loading_directory
        self.total_train_acc, self.total_test_acc = 0, 0
        self.manager_name = manager_name
        self.fast_extractor = FastExtractionPipeline()
        self.registry = self.fast_extractor.registry
        self.registry.set_directory("session_cache")
        self.model_directory = None
        self.data_path = Path(RAW_DATA_DIRECTORY)
        self.cache_path = Path(SESSION_CACHE_PATH)
        self.device_sessions = defaultdict(list)
        self.unseen_sessions = defaultdict(list)
        self.unseen_fraction = UNSEEN_FRACTION
        self.session_counts = Counter()
        random.seed(self.random_state)
        self.collection_times = TIME_INTERVALS
        self.prepare_sessions()

    def save_session(self, cache, cache_name):
        root = zarr.open(self.cache_path / f"{cache_name}.zarr", mode="w")
        for device, sessions in cache.items():
            group = root.create_group(device)
            for i, item in enumerate(sessions):
                if isinstance(item, pd.DataFrame):
                    group.create_dataset(f"session_{i:05d}", data=item.to_records(index=False))
                elif isinstance(item, (str, Path)):
                    group.create_dataset(f"session_{i:05d}", data=str(item))
                else:
                    raise TypeError(f"Unsupported item type {type(item)} in cache for {device}")

    def load_sessions(self, cache_name):
        root = zarr.open(self.cache_path / f"{cache_name}.zarr", mode="r")
        sessions = {}
        for device in root.group_keys():
            device_group = root[device]
            loaded = []
            for ds in device_group.values():
                if ds.dtype.names:
                    loaded.append(pd.DataFrame(ds[:]))
                else:
                    val = ds[()]  # scalar read (not slicing)
                    if isinstance(val, bytes):
                        val = val.decode()
                    loaded.append(Path(val))
            sessions[device] = loaded
        return sessions

    def cache_sessions(self):
        for device_dir in self.data_path.iterdir():
            device_name = str(device_dir.name)
            print(device_name)
            time_to_session = defaultdict(list)
            session_id = 0
            for device_pcap in device_dir.rglob("*.pcap"):
                unlabeled_device_df = self.fast_extractor.extract_features(str(device_pcap))
                if unlabeled_device_df.empty:
                    continue
                time_arr = self.registry.get_metadata()
                if random.random() < self.unseen_fraction:
                    self.unseen_sessions[device_name].append(device_pcap)
                    continue
                labeled_df = self.data_prep.label_device(unlabeled_device_df, 0)
                window_start = 0
                for i, interval_end in enumerate(time_arr):
                    for collection_time in self.collection_times:
                        if interval_end < collection_time:
                            time_to_session[collection_time].append(
                                (labeled_df.iloc[window_start:i + 1], session_id)
                                )
                            window_start += 1
                            break
                session_id += 1
                self.session_counts[device_name] = session_id
            self._save_time_to_session(device_name, time_to_session)
        self.save_session_counts()

    def save_session_counts(self):
        output_directory = self.cache_path / "session_counts.json"
        with open(output_directory, 'w') as file:
            json.dump(self.session_counts, file, indent=2)

    def load_session_counts(self):
        output_directory = self.cache_path / "session_counts.json"
        if len(self.session_counts) != 0:
            return self.session_counts
        with open(output_directory, 'r') as file:
            self.session_counts = json.load(file)
            
    def _save_time_to_session(self, device_name, time_to_session):
        for collection_time in time_to_session:
            collection_dir = self.cache_path / "collection_times" / str(collection_time)
            print(collection_dir)
            collection_dir.mkdir(parents=True, exist_ok=True)
            for session, session_id in time_to_session[collection_time]:
                session_file = collection_dir / device_name / f"session_{session_id}.parquet"
                session_file.parent.mkdir(parents=True, exist_ok=True)
                session.to_parquet(session_file, index=False)

    def map_sessions(self):
        self.load_session_counts()
        self.device_sessions = {device_name:[None]*self.session_counts[device_name] for device_name in self.session_counts}
        seen_cache_dir = self.cache_path / "collection_times"
        for collection_time in seen_cache_dir.iterdir():
            for device_dir in collection_time.iterdir():
                device_name = device_dir.name
                for session_file in device_dir.iterdir():
                    session = pd.read_parquet(session_file)
                    session_index = int(session_file.stem.split("_")[1])
                    placeholder = self.device_sessions[device_name][session_index]
                    if placeholder is None:
                        self.device_sessions[device_name][session_index] = session
                    else:
                        self.device_sessions[device_name][session_index] = pd.concat([placeholder, session], ignore_index=True)

    def prepare_sessions(self):
        if not self.cache_path.exists() or not any(self.cache_path.iterdir()):
            self.cache_sessions()
        else:
            self.map_sessions()
            self.unseen_sessions = self.load_sessions("unseen_sessions")

    def train_classifier(self, record, show_curve = False):
        clf = BaseModel(self.architecture, record.data, record.name)
        clf.split()
        clf.scale()
        clf.train()
        clf.evaluate()
        record.model = clf
        record.evaluation = {
            "train_acc": clf.train_acc,
            "test_acc": clf.test_acc,
            "report": clf.report,
            "confusion_matrix": clf.confusion_matrix,
        }
        if show_curve:
            clf.plot_learning_curve()

    def train_all(self):
        for record in self.records:
            self.train_classifier(record)
        print("\n Training complete.")

    def create_model_directory(self):
        current_date = datetime.today().strftime('%Y-%m-%d')
        current_directory = f"{self.output_directory}/{current_date}"
        os.makedirs(current_directory, exist_ok=True)
        model_ref = str(len(os.listdir(current_directory)))
        if model_ref == '0': model_ref = ''
        current_model_dir = f"{current_directory}/{self.manager_name}{model_ref}"
        os.makedirs(current_model_dir)
        self.model_directory = current_model_dir

    def save_evaluation(self, record: ModelRecord):
        name = record.name
        train_acc, test_acc, report, conf_matrix = record.evaluation.values()
        with open(f"{self.model_directory}/z_evaluation.txt", 'a') as file:
            file.write(f"\n=== {name} ===\n")
            file.write(f"train accuracy: {train_acc}\n")
            file.write(f"test accuracy: {test_acc}\n")
            file.write(f"report:\n{report}\n")
            file.write(f"confusion_matrix:\n{conf_matrix}")
            file.write("\n")
        return train_acc, test_acc

    def save_average_accuracies(self):
        avg_train_acc = self.total_train_acc / len(self.records)
        avg_test_acc = self.total_test_acc / len(self.records)
        with open(f"{self.model_directory}/z_evaluation.txt", 'a') as file:
            file.write("\n=== Average Accuracies ===\n")
            file.write(f"Average Train Accuracy: {avg_train_acc:.4f}\n")
            file.write(f"Average Test Accuracy: {avg_test_acc:.4f}\n")

    def save_classifier(self, record, save_input_data = False):
        if self.model_directory is None:
            self.create_model_directory()
        model = record.model
        name = record.name
        joblib.dump(model, f"{self.model_directory}/{name}.pkl")
        train_acc, test_acc = self.save_evaluation(record)
        self.total_train_acc += train_acc
        self.total_test_acc += test_acc
        if save_input_data:
            model.X_test.to_csv(f"{self.model_directory}/input.csv")
            model.y_test.to_csv(f"{self.model_directory}/output.csv")
        if model.cv_results is not None:
            model.cv_results.to_csv(f"{self.model_directory}/cross_validation.csv")

    def save_all(self, save_input_data = False):
        if os.path.isfile(self.output_directory):
            raise ValueError("output_directory is a file")
        self.create_model_directory()
        for record in self.records:
            model = record.model
            name = record.name
            joblib.dump(model, f"{self.model_directory}/{name}.pkl")
            print(f"saved {model} to {self.model_directory}/{name}.pkl")
            train_acc, test_acc = self.save_evaluation(record)
            self.total_train_acc += train_acc
            self.total_test_acc += test_acc
            if save_input_data:
                model.X_test.to_csv(f"{self.model_directory}/input.csv")
                model.y_test.to_csv(f"{self.model_directory}/output.csv")
            if model.cv_results is not None:
                model.cv_results.to_csv(f"{self.model_directory}/cross_validation.csv")
        if len(self.records) > 1:
            self.save_average_accuracies()

    def load_model(self):
        if self.loading_directory is None: 
            raise ValueError("Loading directory has not been specified")
        if not os.path.exists(self.loading_directory):
            raise FileNotFoundError("Model has to be saved before it is loaded")
        return [joblib.load(f"{self.loading_directory}/{file}") for file in os.listdir(self.loading_directory) if file.endswith(".pkl")]

if __name__ == "__main__":
    manager = Manager()
        


