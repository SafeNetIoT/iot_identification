import os
from pandas.errors import EmptyDataError
from src.ml.base_model import BaseModel
from config import MODEL_ARCHITECTURES, RANDOM_STATE, MODELS_DIRECTORY, RAW_DATA_DIRECTORY, SESSION_CACHE_PATH, UNSEEN_FRACTION
from src.ml.dataset_preparation import DatasetPreparation
from typing import List
from datetime import datetime
import joblib
from src.ml.model_record import ModelRecord
from src.features.fast_extraction import FastExtractionPipeline
import pandas as pd
import zarr
from pathlib import Path
from collections import defaultdict
import random

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
        self.model_directory = None
        self.data_path = Path(RAW_DATA_DIRECTORY)
        self.cache_path = Path(SESSION_CACHE_PATH)
        self.device_sessions = defaultdict(list)
        self.unseen_sessions = defaultdict(list)
        self.unseen_fraction = UNSEEN_FRACTION
        random.seed(self.random_state)
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

    def map_sessions(self):
        for device_pcap in self.data_path.rglob("*.pcap"):
            device_name = device_pcap.parent.parent.name
            unlabeled_device_df = self.fast_extractor.extract_features(str(device_pcap))
            if unlabeled_device_df.empty:
                continue
            if random.random() < self.unseen_fraction:
                self.unseen_sessions[device_name].append(device_pcap)
                continue
            labeled_df = self.data_prep.label_device(unlabeled_device_df, 0)
            labeled_df.attrs['pcap_path'] = str(device_pcap)
            self.device_sessions[device_name].append(labeled_df)

    def prepare_sessions(self):
        if self.cache_path.exists() and any(self.cache_path.iterdir()):
            self.device_sessions = self.load_sessions("seen_sessions")
            self.unseen_sessions = self.load_sessions("unseen_sessions")
        else:
            self.map_sessions()
            self.save_session(self.device_sessions, "seen_sessions")
            self.save_session(self.unseen_sessions, "unseen_sessions")

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

    def predict(self, pcap_file):
        X = self.fast_extractor.extract_features(pcap_file)
        if X.empty:
            raise EmptyDataError("PCAP file is empty")
        model_arr = self.load_model()
        result_class, score = None, 0
        for model in model_arr:
            predicted_class, confidence = model.predict(X)
            if predicted_class == 0:
                continue
            if confidence > score:
                result_class, score = model.name, confidence
        return result_class

        


