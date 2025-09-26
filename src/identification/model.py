import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from config import PREPROCESSED_DATA_DIRECTORY, VALID_FEATURES_DIRECTORY, MODELS_DIRECTORY
import datetime

class DatasetPreparation:
    def __init__(self) -> None:
        self.preprocessed_data_directory = PREPROCESSED_DATA_DIRECTORY
        feature_dir = VALID_FEATURES_DIRECTORY
        with open(feature_dir, 'r') as file:
            self.features = [line.strip() for line in file]

        self.output = pd.DataFrame(columns=self.features + ['label'])
        self.output_directory = MODELS_DIRECTORY
        self.random_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.test_size = 0.2


    def prune_features(self, device_name, device_csv):
        feature_set = set(self.features)
        device = pd.read_csv(f"{self.preprocessed_data_directory}/{device_csv}")
        keep_cols = [col for col in device.columns if col in feature_set]
        device = device[keep_cols]
        device.loc[:, 'label'] = device_name
        return device

    def clean_up(self):
        if self.output is None or self.output.empty:
            raise ValueError("Output DataFrame is empty. Run preprocess() first.")

        # Report missing values
        null_counts = self.output.isnull().sum()
        if not null_counts.empty:
            print("Missing values per column:")
            print(null_counts[null_counts > 0])

        # Drop rows with any missing values (optional: you can impute instead)
        self.output = self.output.dropna()

        # Ensure all feature columns are numeric
        for col in self.features:
            if col in self.output.columns:
                self.output[col] = pd.to_numeric(self.output[col], errors="coerce")

        # Check again if conversion caused NaNs (bad values → NaN)
        bad_vals = self.output.isnull().sum()
        if bad_vals.sum() > 0:
            print("Warning: Non-numeric values were found and converted to NaN.")
            print(bad_vals[bad_vals > 0])
            # Optionally drop those rows too
            self.output = self.output.dropna()

        # Reset index after cleanup
        self.output = self.output.reset_index(drop=True)

        print("Cleanup complete. Shape:", self.output.shape)
        return self.output

    def balance_dataset(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run preprocess() before balancing.")

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(self.X_train, self.y_train)

        # print("Before balancing:", self.y_train.value_counts().to_dict())
        # print("After balancing:", pd.Series(y_resampled).value_counts().to_dict())

        self.X_train, self.y_train = X_resampled, y_resampled
        return self.X_train, self.y_train      

    def preprocess(self):
        all_devices = []
        for device_directory in os.listdir(self.preprocessed_data_directory):
            device_name = device_directory.split(".csv")[0]
            device_df = self.prune_features(device_name, device_directory)
            all_devices.append(device_df)
        self.output = pd.concat(all_devices, ignore_index=True)
        self.clean_up()
        X = self.output.drop(columns=["label"])
        y = self.output["label"]

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=X.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=X.columns)

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")

        os.makedirs(f"{self.output_directory}")
        self.X_test.to_csv(f"{self.output_directory}/independent.csv")
        self.y_test.to_csv(f"{self.output_directory}/dependent.csv")
        return self.X_train, self.X_test, self.y_train, self.y_test

class Model:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # use all cores
        )
        self.output_directory = MODELS_DIRECTORY

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return acc

    def predict(self, X):
        return self.model.predict(X)

    def save(self, output_path):
        joblib.dump(self.model, output_path)
        print(f"Model saved to {output_path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model

def main():
    prep = DatasetPreparation()
    X_train, X_test, y_train, y_test = prep.preprocess()
    X_train_bal, y_train_bal = prep.balance_dataset()
    clf = Model()
    clf.train(X_train_bal, y_train_bal)
    clf.evaluate(X_test, y_test)
    clf.save(f"{MODELS_DIRECTORY}/rf_model_{datetime.date()}.pkl")

if __name__ == "__main__":
    main()