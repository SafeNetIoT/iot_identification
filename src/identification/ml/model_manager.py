import os
import pandas as pd
from typing import Dict
from src.identification.ml.base_model import BaseModel
from config import MODEL_ARCHITECTURES
from src.identification.ml.dataset_preparation import DatasetPreparation


class BaseManager:
    """Coordinates dataset preparation and BaseModel training/evaluation."""
    def __init__(self, architecture_name="standard_forest"):
        self.architecture = MODEL_ARCHITECTURES[architecture_name]
        self.data_prep = DatasetPreparation()
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, float] = {}

    def train_all(self):
        dataset_map = self.prepare_datasets()
        for name, df in dataset_map.items():
            print(f"\n Training model: {name}")
            model = BaseModel(self.architecture, df, name)
            model.split()
            model.scale()
            model.train()
            acc = model.evaluate()
            self.models[name] = model
            self.results[name] = acc
        print("\n Training complete.")

    def save_all(self):
        for model in self.models.values():
            model.save()

    def summary(self):
        return pd.DataFrame([
            {"name": n, "accuracy": a}
            for n, a in self.results.items()
        ]).sort_values("accuracy", ascending=False)

    def prepare_datasets(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement prepare_datasets().")
