import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from identification.ml.dataset_preparation import DatasetPreparation
from src.identification.ml.base_model import BaseModel
from config import PREPROCESSED_DATA_DIRECTORY, MODEL_ARCHITECTURES

class BinaryModel:
    def __init__(self, name, test_size=0.2) -> None:
        self.device_csvs = [
            f.replace(".csv", "")
            for f in os.listdir(PREPROCESSED_DATA_DIRECTORY)
            if f.endswith(".csv")
        ]
        self.name = name
        self.test_size = test_size
        self.num_classes = len(self.device_csvs)
        self.data_prep = DatasetPreparation()
        self.models = {}
        self.evaluations = {}
        self.architecture = MODEL_ARCHITECTURES['standard_forest']

    def train_classifier(self, input_data, name):
        """Train RandomForest on binary (device vs others) data."""
        clf = BaseModel(self.architecture, input_data, name)
        clf.train()
        clf.evaluate()
        self.evaluations[name] = (clf.train_acc, clf.test_acc, clf.report, clf.confusion_matrix)

    def sample_false_class(self, current_device_name, records_per_class):
        sampled_dfs = []
        for device_name in self.device_csvs:
            if device_name == current_device_name:
                continue
            device_path = f"{PREPROCESSED_DATA_DIRECTORY}/{device_name}.csv"
            device_df = pd.read_csv(device_path)
            pruned = self.data_prep.prune_features(device_df)
            labeled = self.data_prep.label_device(pruned, 0)
            sampled = labeled.sample(
                n=min(records_per_class, len(labeled)),
                random_state=self.data_prep.random_state
            )
            sampled_dfs.append(sampled)
        return pd.concat(sampled_dfs, ignore_index=True)

    def add_device(self, current_device_name):
        """Train binary classifier for one device vs all others."""
        current_device_path = f"{PREPROCESSED_DATA_DIRECTORY}/{current_device_name}.csv"
        device_df = pd.read_csv(current_device_path)
        pruned = self.data_prep.prune_features(device_df)
        labeled = self.data_prep.label_device(pruned, 1)

        num_records = len(labeled)
        records_per_class = max(1, num_records // max(1, self.num_classes - 1))

        false_class = self.sample_false_class(current_device_name, records_per_class)
        input_data = pd.concat([labeled, false_class], ignore_index=True)
        self.train_classifier(input_data, current_device_name)

    def train_all(self):
        """Train one model per device."""
        for device_name in self.device_csvs:
            print(f"\n Training model for device: {device_name}")
            self.add_device(device_name)

    def save(self):
        pass


def main():
    binary_model = BinaryModel()
    binary_model.train_all()
    binary_model.save()


if __name__ == "__main__":
    main()
