"""
Still WIP
"""

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

class TestPipeline:
    def __init__(self, verbose=True) -> None:
        self.collection_times = TIME_INTERVALS
        self.verbose = verbose
        self.cache = TimeBasedCache()
        self.time_datasets = self.cache.build()
        self.manager = BinaryModel()

    def run_intervals(self):
        for collection_time, dataset in self.time_datasets.items():
            self.manager.set_device_sessions(dataset)
            print(dataset)
            self.manager.prepare_datasets()
            try:
                self.manager.train_all()
                self.manager.save_all()
            except ValueError:
                print("not enough data to train all classes")
            self.manager.reset_training_attributes()

    def generate_time_datasets(self):
        for device_pcap in self.data_path.rglob("*.pcap"):
            device_name = device_pcap.parent.parent.name
            device_df = self.fast_extractor.extract_features(str(device_pcap))
            if device_df.empty:
                continue
            device_df = prep.label_device(device_df, 0)
            time_arr = self.registry.get_metadata()
            for i, time_period in enumerate(time_arr):
                for collection_time in self.collection_times:
                    if time_period <= collection_time:
                        cache_dir = self.base_cache_dir /str(collection_time) / device_name
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        file_path = cache_dir / f"{device_pcap.stem}_{time_period}.parquet"
                        device_df.iloc[:i].to_parquet(file_path, index=False)
                        self.time_datasets[collection_time][device_name].append(str(file_path))

    def prepare_true_class(self, device_directory: Path):
        true_class = []
        for device_file in device_directory.iterdir():
            device_df = pd.read_parquet(device_file)
            true_class.append(prep.label_device(device_df, 1))
        return true_class
        
    def sample_false_class(self, current_device_name, collection_time_directory, sessions_per_class):
        sampled_dfs = []
        for device_name in collection_time_directory.iterdir():
            if device_name == current_device_name:
                continue
            device_directory = collection_time_directory / device_name
            all_files = [f for f in device_directory.iterdir() if f.is_file()]
            sample_size = min(sessions_per_class, len(all_files))
            sampled = random.sample(all_files, sample_size)
            sampled_dfs.extend([pd.read_parquet(file) for file in sampled])
        return sampled_dfs

    def train_model(self):
        current_dir = self.base_cache_dir
        for collection_time in self.base_cache_dir.iterdir():
            current_dir = current_dir / str(collection_time)
            for device_name in current_dir.iterdir():
                current_dir = current_dir / device_name
                true_class = self.prepare_true_class(current_dir)
                sessions_per_class = max(1, len(true_class) // max(1, len(self.base_cache_dir) - 1))
                false_class = self.sample_false_class(device_name, collection_time, sessions_per_class)
                data = true_class + false_class
                record = ModelRecord(name=device_name, data=data)
                self.manager.records.append(record)
        self.manager.train_all()
        self.manager.save_all()
        self.manager.records = []
        self.manager.total_train_acc, self.manager.total_test_acc = 0, 0

    def test_intervals(self):
        if not self.time_datasets:
            self.generate_time_datasets()
        for collection_time in self.time_datasets:
            self.train_model(self.time_datasets[collection_time])

    def compare_time_intervals(self, alpha: float = 0.05):
        intervals = sorted(self.time_datasets.keys())
        if len(intervals) < 2:
            raise ValueError("Need at least two time intervals to compare.")

        # --- Step 1: Load and combine all files for each interval ---
        combined = {}
        for interval in intervals:
            dfs = []
            for device, paths in self.time_datasets[interval].items():
                for path in paths:
                    path = Path(path)
                    if not path.exists():
                        print(f"⚠️  Missing file: {path}")
                        continue
                    try:
                        df = pd.read_csv(path)
                        dfs.append(df)
                    except Exception as e:
                        print(f"⚠️  Error reading {path}: {e}")
            if dfs:
                combined[interval] = pd.concat(dfs, ignore_index=True)
            else:
                combined[interval] = pd.DataFrame()
        
        # --- Step 2: Pick numeric columns ---
        first_nonempty = next((df for df in combined.values() if not df.empty), None)
        if first_nonempty is None:
            raise ValueError("All intervals have empty or missing data.")
        
        numeric_cols = [
            col for col in first_nonempty.columns
            if pd.api.types.is_numeric_dtype(first_nonempty[col])
        ]

        # --- Step 3: Pairwise comparisons across intervals ---
        results = []
        for (t1, df1), (t2, df2) in combinations(combined.items(), 2):
            if df1.empty or df2.empty:
                continue
            for col in numeric_cols:
                a = pd.to_numeric(df1[col], errors="coerce").dropna()
                b = pd.to_numeric(df2[col], errors="coerce").dropna()
                if len(a) < 2 or len(b) < 2:
                    continue
                stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                results.append({
                    "interval_1": t1,
                    "interval_2": t2,
                    "column": col,
                    "statistic": stat,
                    "p_value": p,
                    "significant": p < alpha
                })

        # --- Step 4: Summarize results ---
        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("⚠️  No comparable numeric data found across intervals.")
            return results_df, pd.DataFrame()

        summary = (
            results_df.groupby("column")["significant"]
            .sum()
            .reset_index()
            .rename(columns={"significant": "significant_differences"})
            .sort_values("significant_differences", ascending=False)
        )

        print(f"✅ Compared {len(results_df)} feature pairs across {len(intervals)} intervals.")
        print(f"Top differing features:\n{summary.head(10)}")

        return results_df, summary
    
    def test_windows(self):
        if not self.time_datasets:
            self.generate_time_datasets()
        results_df, summary = self.compare_time_intervals()
        return results_df, summary

def main():
    pipeline = TestPipeline()
    # pipeline.generate_time_datasets()
    # pipeline.test_intervals()
    # pipeline.test_windows()

    # for collection_time, dataset_map in pipeline.time_datasets.items():
    #     print(collection_time)
    #     for device_name, session_list in dataset_map.items():
    #         print(device_name, len(session_list))
    #     print()
    pipeline.run_intervals()

if __name__ == "__main__":
    main()