import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from config import PREPROCESSED_DATA_DIRECTORY, VALID_FEATURES_DIRECTORY, MODELS_DIRECTORY
from datetime import datetime


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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=X.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=X.columns)

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1  # use all cores
        )
        self.output_directory = MODELS_DIRECTORY
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)

        self.acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {self.acc:.4f}")

        self.report = classification_report(self.y_test, y_pred)
        print("\nClassification Report:")
        print(self.report)

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(self.confusion_matrix)

    def save(self):
        current_date = datetime.today().strftime('%Y-%m-%d')
        current_directory = f"{self.output_directory}/{current_date}"
        os.makedirs(current_directory, exist_ok=True)
        model_ref = str(len(os.listdir(current_directory)))
        if model_ref == '0': 
            model_ref = ''
        current_model_dir = f"{current_directory}/random_forest{model_ref}"
        os.makedirs(current_model_dir)
        joblib.dump(self.model, f"{current_model_dir}/random_forest.pkl")
        self.X_test.to_csv(f"{current_model_dir}/input.csv")
        self.y_test.to_csv(f"{current_model_dir}/output.csv")
        with open(f"{current_model_dir}/evalutation.txt", 'w') as file:
            file.write(f"accuracy: {self.acc}\n")
            file.write(f"report:\n{self.report}\n")
            file.write(f"confusion_matrix:\n{self.confusion_matrix}")
        print(f"Model saved to {current_model_dir}")

def main():
    prep = DatasetPreparation()
    _, X_test, _, y_test = prep.preprocess()
    X_train_bal, y_train_bal = prep.balance_dataset()
    clf = Model(X_train=X_train_bal, y_train=y_train_bal, X_test=X_test, y_test=y_test)
    clf.train()
    clf.evaluate()
    clf.save()

if __name__ == "__main__":
    main()