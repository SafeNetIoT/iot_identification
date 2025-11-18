from config import settings
import pandas as pd
from src.services.cache import TimeBasedCache
from src.ml.binary_model import BinaryModel
from src.utils.evaluation import evaluate_on_fixed_unseen

class TestPipeline:
    def __init__(self, verbose=True) -> None:
        self.collection_times = settings.time_intervals
        self.verbose = verbose
        self.manager = BinaryModel()
        self.time_datasets, self.unseen_sessions = self.manager.set_cache(cache=TimeBasedCache())

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
                acc = evaluate_on_fixed_unseen(self.manager)
                results.append((collection_time, acc))
            except ValueError:
                print(f"Skipping {collection_time}: not enough data")
            self.manager.reset_training_attributes()

        return pd.DataFrame(results, columns=["time", "accuracy"])

def main():
    pipeline = TestPipeline()
    # pipeline.run_intervals()
    print(pipeline.run_time_learning_curve())

if __name__ == "__main__":
    main()