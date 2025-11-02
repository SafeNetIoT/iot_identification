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
            if all(len(v) == 0 for v in dataset.values()):
                print("empty dict:", collection_time)
                continue
            self.manager.set_device_sessions(dataset)
            self.manager.prepare_datasets()
            print("num records:", len(self.manager.records))
            try:
                self.manager.train_all()
                print("num models:", len(self.manager.model_arr))
                acc = evaluate_on_fixed_unseen(self.unseen_sessions, self.manager.predict)
                results.append((collection_time, acc))
            except ValueError:
                print(f"Skipping {collection_time}: not enough data")
            self.manager.reset_training_attributes()

        return pd.DataFrame(results, columns=["time", "accuracy"])

    # def compare_time_intervals(self, alpha: float = 0.05):
    #     intervals = sorted(self.time_datasets.keys())
    #     if len(intervals) < 2:
    #         raise ValueError("Need at least two time intervals to compare.")

    #     # --- Step 1: Load and combine all files for each interval ---
    #     combined = {}
    #     for interval in intervals:
    #         dfs = []
    #         for device, paths in self.time_datasets[interval].items():
    #             for path in paths:
    #                 path = Path(path)
    #                 if not path.exists():
    #                     print(f"⚠️  Missing file: {path}")
    #                     continue
    #                 try:
    #                     df = pd.read_csv(path)
    #                     dfs.append(df)
    #                 except Exception as e:
    #                     print(f"⚠️  Error reading {path}: {e}")
    #         if dfs:
    #             combined[interval] = pd.concat(dfs, ignore_index=True)
    #         else:
    #             combined[interval] = pd.DataFrame()
        
    #     # --- Step 2: Pick numeric columns ---
    #     first_nonempty = next((df for df in combined.values() if not df.empty), None)
    #     if first_nonempty is None:
    #         raise ValueError("All intervals have empty or missing data.")
        
    #     numeric_cols = [
    #         col for col in first_nonempty.columns
    #         if pd.api.types.is_numeric_dtype(first_nonempty[col])
    #     ]

    #     # --- Step 3: Pairwise comparisons across intervals ---
    #     results = []
    #     for (t1, df1), (t2, df2) in combinations(combined.items(), 2):
    #         if df1.empty or df2.empty:
    #             continue
    #         for col in numeric_cols:
    #             a = pd.to_numeric(df1[col], errors="coerce").dropna()
    #             b = pd.to_numeric(df2[col], errors="coerce").dropna()
    #             if len(a) < 2 or len(b) < 2:
    #                 continue
    #             stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    #             results.append({
    #                 "interval_1": t1,
    #                 "interval_2": t2,
    #                 "column": col,
    #                 "statistic": stat,
    #                 "p_value": p,
    #                 "significant": p < alpha
    #             })

    #     # --- Step 4: Summarize results ---
    #     results_df = pd.DataFrame(results)
    #     if results_df.empty:
    #         print("⚠️  No comparable numeric data found across intervals.")
    #         return results_df, pd.DataFrame()

    #     summary = (
    #         results_df.groupby("column")["significant"]
    #         .sum()
    #         .reset_index()
    #         .rename(columns={"significant": "significant_differences"})
    #         .sort_values("significant_differences", ascending=False)
    #     )

    #     print(f"✅ Compared {len(results_df)} feature pairs across {len(intervals)} intervals.")
    #     print(f"Top differing features:\n{summary.head(10)}")

    #     return results_df, summary
    
    # def test_windows(self):
    #     if not self.time_datasets:
    #         self.generate_time_datasets()
    #     results_df, summary = self.compare_time_intervals()
    #     return results_df, summary

def main():
    import json
    pipeline = TestPipeline()
    # pipeline.run_intervals()
    print(pipeline.run_time_learning_curve())

if __name__ == "__main__":
    main()