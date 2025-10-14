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

    def clean_up(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("Output DataFrame is empty. Run preprocess() first.")

        existing_cols = [col for col in self.features if col in df.columns]
        df[existing_cols] = df[existing_cols].apply(pd.to_numeric, errors="coerce")

        before_rows = len(df)
        df = df.dropna().reset_index(drop=True)

        if len(df) < before_rows:
            print(f"Warning: Dropped {before_rows - len(df)} rows with invalid or missing values.")
        return df

    def prepare_df(self, device_df, label):
        pruned_df = self.prune_features(device_df)
        labeled_df = self.label_device(pruned_df, label)
        return self.clean_up(labeled_df)
