"""
Still WIP
"""

from src.features.feature_extraction import ExtractionPipeline
import os
from config import TIME_INTERVALS, RAW_DATA_DIRECTORY, PREPROCESSED_DATA_DIRECTORY
import pandas as pd
from src.ml.dataset_preparation import DatasetPreparation as prep
from src.ml.multi_class_model import MultiClassModel
from scipy import stats
from src.utils.file_utils import unpack_features
from itertools import combinations
from src.ml.binary_model import BinaryModel
from collections import defaultdict, Counter
from pathlib import Path
from src.ml.model_record import ModelRecord
import sys
from src.features.fast_extraction import FastExtractionPipeline

class TestPipeline:
    def __init__(self, verbose=True, manager = BinaryModel()) -> None:
        self.collection_times = TIME_INTERVALS
        self.verbose = verbose
        self.manager = manager
        self.fast_extractor = self.manager.fast_extractor
        self.registry = self.manager.registry
        self.time_datasets = defaultdict(lambda: defaultdict(list))
        self.data_path = Path(RAW_DATA_DIRECTORY)

    def generate_time_datasets(self):
        import os, psutil
        proc = psutil.Process(os.getpid())
        for device_pcap in self.data_path.rglob("*.pcap"):
            device_name = device_pcap.parent.parent.name
            print(f"Extracting from {device_pcap}", flush=True)
            device_df = self.fast_extractor.extract_features(str(device_pcap))
            print(f"Done extracting {device_pcap}", flush=True)
            if device_df.empty:
                continue
            labeled_df = prep.label_device(device_df, 0)
            time_arr = self.registry.get_metadata()
            for time_period in time_arr:
                for collection_time in self.collection_times:
                    if time_period <= collection_time:
                        self.time_datasets[collection_time][device_name].append(str(device_pcap))
                        print(collection_time, device_name, flush=True)
                        print(f"Memory: {proc.memory_info().rss / 1024**2:.2f} MB", flush=True)


    def train_model(self, device_dataset_map):
        for device_name in device_dataset_map:
            true_class = self.manager.prepare_true_class(device_name)
            true_class_num_sessions = len(device_dataset_map[device_name])
            records_per_session = max(1, true_class_num_sessions // max(1, len(device_dataset_map) - 1))
            false_class = self.manager.sample_false_class(device_name, records_per_session)
            data = true_class + false_class
            record = ModelRecord(name=device_name, data=data)
            self.manager.records.append(record)
        self.manager.train_all()
        self.manager.save_all()
        self.manager.records = []
        self.manager.total_train_acc, self.manager.total_test_acc = 0, 0

    def test_intervals(self):
        self.generate_time_datasets()
        for collection_time in self.time_datasets:
            self.train_model(self.time_datasets[collection_time])

    def compare_time_intervals(self, cache: dict, alpha: float = 0.05):
        intervals = sorted(cache.keys())
        if len(intervals) < 2:
            raise ValueError("Need at least two time intervals to compare.")
        cols = cache[intervals[0]].columns
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(cache[intervals[0]][col])]
        results = []
        for (t1, df1), (t2, df2) in combinations(cache.items(), 2):
            for col in numeric_cols:
                a = pd.to_numeric(df1[col], errors="coerce").dropna()
                b = pd.to_numeric(df2[col], errors="coerce").dropna()
                if len(a) < 2 or len(b) < 2:
                    continue
                stat, p = stats.ttest_ind(a, b, equal_var=False)
                results.append({
                    "interval_1": t1,
                    "interval_2": t2,
                    "column": col,
                    "statistic": stat,
                    "p_value": p,
                    "significant": p < alpha
                })
        results_df = pd.DataFrame(results)
        summary = (
            results_df.groupby("column")["significant"]
            .sum()
            .reset_index()
            .rename(columns={"significant": "significant_differences"})
            .sort_values("significant_differences", ascending=False)
        )
        print(f"Compared {len(results_df)} feature pairs across {len(intervals)} intervals.")
        print(f"Top differing features:\n{summary.head(10)}")
        return results_df, summary

    def test_windows(self):
        cache = {}
        for collection_time in self.collection_times:
            if collection_time not in cache:
                cache[collection_time] = self.combine_csvs(collection_time)
        self.compare_time_intervals(cache)

def main():
    import json
    pipeline = TestPipeline()
    pipeline.test_intervals()

if __name__ == "__main__":
    main()