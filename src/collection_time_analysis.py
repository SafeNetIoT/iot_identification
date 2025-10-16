from src.features.feature_extraction import ExtractionPipeline
import os
from config import TIME_INTERVALS, RAW_DATA_DIRECTORY, PREPROCESSED_DATA_DIRECTORY
import pandas as pd
from src.ml.dataset_preparation import DatasetPreparation
from src.ml.multi_class_model import MultiClassModel
from scipy import stats
from src.utils import unpack_features
from itertools import combinations


class TestPipeline:
    def __init__(self, verbose=True) -> None:
        self.collection_times = TIME_INTERVALS
        self.prep = DatasetPreparation()
        self.raw_data_directory = RAW_DATA_DIRECTORY
        self.verbose = verbose
        self.preprocessed_data_dir = PREPROCESSED_DATA_DIRECTORY

    def combine_csvs(self, collection_time):
        extractor = ExtractionPipeline(collection_time)
        all_dfs = []
        for device in os.listdir(self.raw_data_directory):
            if self.verbose:
                print(device)
            for date in os.listdir(f"{self.raw_data_directory}/{device}"):
                for pcap_file in os.listdir(f"{self.raw_data_directory}/{device}/{date}"):
                    pcap_df = extractor.extract_features(input_pcap=f"{self.raw_data_directory}/{device}/{date}/{pcap_file}")
                    if pcap_df.empty:
                        continue
                    prepared_df = self.prep.prepare_df(pcap_df, device)
                    all_dfs.append(prepared_df)
        return pd.concat(all_dfs, ignore_index=True)

    def test_intervals(self):
        cache = {}
        for collection_time in self.collection_times:
            if collection_time not in cache:
                cache[collection_time] = self.combine_csvs(collection_time)
            input_data = cache[collection_time]
            manager = MultiClassModel()
            manager.add_device(input_data)
            manager.train_all()
            manager.save_all()

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
    pipeline = TestPipeline()
    # pipeline.test_intervals()
    pipeline.test_windows()

if __name__ == "__main__":
    main()