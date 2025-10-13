import pandas as pd
from config import PREPROCESSED_DATA_DIRECTORY, VALID_FEATURES_DIRECTORY


class DatasetPreparation:
    def __init__(self) -> None:
        self.preprocessed_data_directory = PREPROCESSED_DATA_DIRECTORY
        feature_dir = VALID_FEATURES_DIRECTORY
        with open(feature_dir, 'r') as file:
            self.features = [line.strip() for line in file]
        self.feature_set = set(self.features)
        self.output = pd.DataFrame(columns=self.features + ['label'])

    def prune_features(self, device_df):
        keep_cols = [col for col in device_df.columns if col in self.feature_set]
        device = device_df[keep_cols].copy()
        return device

    def label_device(self, device_df, device_label):
        device_df.loc[:, 'label'] = device_label
        return device_df

    def clean_up(self, df: pd.DataFrame): #optimize
        if df is None or df.empty:
            raise ValueError("Output DataFrame is empty. Run preprocess() first.")

        null_counts = df.isnull().sum()
        # if not null_counts.empty:
        #     print("Missing values per column:")
        #     print(null_counts[null_counts > 0])
        df = df.dropna()

        for col in self.features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        bad_vals = df.isnull().sum()
        if bad_vals.sum() > 0:
            print("Warning: Non-numeric values were found and converted to NaN.")
            print(bad_vals[bad_vals > 0])
            df = df.dropna()

        df = df.reset_index(drop=True)
        return df

    def prepare_df(self, device_df, label):
        pruned_df = self.prune_features(device_df)
        labeled_df = self.label_device(pruned_df, label)
        return self.clean_up(labeled_df)
