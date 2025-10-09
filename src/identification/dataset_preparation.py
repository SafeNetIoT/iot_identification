import pandas as pd
from config import PREPROCESSED_DATA_DIRECTORY, VALID_FEATURES_DIRECTORY, MODELS_DIRECTORY
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os


class DatasetPreparation:
    def __init__(self) -> None:
        self.preprocessed_data_directory = PREPROCESSED_DATA_DIRECTORY
        feature_dir = VALID_FEATURES_DIRECTORY
        with open(feature_dir, 'r') as file:
            self.features = [line.strip() for line in file]
        self.feature_set = set(self.features)
        self.output = pd.DataFrame(columns=self.features + ['label'])
        self.device_map = {} # device_name: df
        self.output_directory = MODELS_DIRECTORY
        self.random_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.test_size = 0.2

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

        # Report missing values
        null_counts = df.isnull().sum()
        if not null_counts.empty:
            print("Missing values per column:")
            print(null_counts[null_counts > 0])

        # Drop rows with any missing values (optional: you can impute instead)
        df = df.dropna()

        # Ensure all feature columns are numeric
        for col in self.features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check again if conversion caused NaNs (bad values → NaN)
        bad_vals = df.isnull().sum()
        if bad_vals.sum() > 0:
            print("Warning: Non-numeric values were found and converted to NaN.")
            print(bad_vals[bad_vals > 0])
            # Optionally drop those rows too
            df = df.dropna()

        # Reset index after cleanup
        df = df.reset_index(drop=True)

        print("Cleanup complete. Shape:", df.shape)
        return df

    def balance_dataset(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run preprocess() before balancing.")

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(self.X_train, self.y_train)

        # print("Before balancing:", self.y_train.value_counts().to_dict())
        # print("After balancing:", pd.Series(y_resampled).value_counts().to_dict())

        self.X_train, self.y_train = X_resampled, y_resampled
        return self.X_train, self.y_train

    def combine_csvs(self):      
        all_devices = []
        for device_csv in os.listdir(self.preprocessed_data_directory):
            device_df = pd.read_csv(f"{self.preprocessed_data_directory}/{device_csv}")
            device_name = device_csv.split(".csv")[0]
            device_df = self.label_device(self.prune_features(device_df), device_name)
            device_df = self.clean_up(device_df)
            self.device_map[device_name] = device_df
            all_devices.append(device_df)
        self.output = pd.concat(all_devices, ignore_index=True)

    def scale(self, X_train, X_test, columns):
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=columns)
        return X_train, X_test

    def preprocess(self):
        self.combine_csvs()
        X = self.output.drop(columns=["label"])
        y = self.output["label"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self.X_train, self.X_test = self.scale(self.X_train, self.X_test, X.columns)

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
