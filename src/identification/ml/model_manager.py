import os
import pandas as pd
from typing import Dict
from src.identification.ml.base_model import BaseModel
from config import MODEL_ARCHITECTURES, RANDOM_STATE
from src.identification.ml.dataset_preparation import DatasetPreparation
from typing import List


class Manager:
    def __init__(self, input_data: List[pd.DataFrame], architecture_name="standard_forest", name = "random_forest"):
        self.architecture = MODEL_ARCHITECTURES[architecture_name]
        self.data_prep = DatasetPreparation()
        self.models = []
        self.results = []
        self.input_data = input_data
        self.random_state = RANDOM_STATE
        self.name = name

    def validate_input(self):
        if not isinstance(self.input_data, list):
            raise ValueError("Input data is not a list:" + str(type(input)))


    def train_all(self):
        for labeled_data in self.input_data:
            clf = BaseModel(self.architecture, labeled_data, self.name)
            clf.split()
            clf.scale()
            clf.train()
            clf.evaluate()
            self.models.append(clf)
            self.results.append((clf.train_acc, clf.test_acc, clf.report, clf.confusion_matrix))
        print("\n Training complete.")

    def save_all(self):
        for model in self.models:
            model.save()

    # def summary(self):
    #     return pd.DataFrame([
    #         {"name": n, "accuracy": a}
    #         for n, a in self.results.items()
    #     ]).sort_values("accuracy", ascending=False)
