from config import TIME_INTERVALS
import pandas as pd
from src.ml.dataset_preparation import DatasetPreparation as prep
from scipy import stats
from itertools import combinations
from pathlib import Path
from src.ml.model_record import ModelRecord
import random
from src.ml.cache import TimeBasedCache
from src.ml.binary_model import BinaryModel
from src.utils.evaluation import evaluate_on_fixed_unseen

class TestPipeline:
    def __init__(self, verbose=True) -> None:
        self.collection_times = TIME_INTERVALS
        self.verbose = verbose
        self.cache = TimeBasedCache()
        self.time_datasets, self.unseen_sessions = self.cache.build()
        self.manager = BinaryModel()

    def run_intervals(self):
        for dataset in self.time_datasets.values():
            self.manager.set_device_sessions(dataset)
            self.manager.prepare_datasets()
            try:
                self.manager.train_all()
                self.manager.save_all()
            except ValueError:
                print("not enough data to train all classes")
            self.manager.reset_training_attributes()
        
    def run_time_learning_curve(self):
        results = []
        for collection_time, dataset in self.time_datasets.items():
            self.manager.set_device_sessions(dataset)
            self.manager.prepare_datasets()
            num_records = len(self.manager.records)
            if num_records == 0:
                print("no records:", collection_time)
                continue
            try:
                self.manager.train_all()
                print("num models:", len(self.manager.model_arr))
                acc = evaluate_on_fixed_unseen(self.unseen_sessions, self.manager.predict)
                results.append((collection_time, acc))
            except ValueError:
                print(f"Skipping {collection_time}: not enough data")
            self.manager.reset_training_attributes()

        return pd.DataFrame(results, columns=["time", "accuracy"])

def main():
    import json
    pipeline = TestPipeline()
    # pipeline.run_intervals()
    print(pipeline.run_time_learning_curve())

if __name__ == "__main__":
    main()