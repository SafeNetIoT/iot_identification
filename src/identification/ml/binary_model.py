import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from identification.ml.dataset_preparation import DatasetPreparation
from config import PREPROCESSED_DATA_DIRECTORY, MODELS_DIRECTORY


class BinaryModel:
    def __init__(self) -> None:
        self.models = {}  # {device_name: model_object}
        self.device_csvs = [
            f.replace(".csv", "")
            for f in os.listdir(PREPROCESSED_DATA_DIRECTORY)
            if f.endswith(".csv")
        ]
        self.num_classes = len(self.device_csvs)
        self.data_prep = DatasetPreparation()
        self.data_prep.combine_csvs()
        self.evaluations = {}  # {device_name: (train_acc, test_acc, report, cm)}
        self.output_directory = MODELS_DIRECTORY

    def sample_false_class(self, current_device_name, records_per_class):
        sampled_dfs = []

        for device_name, device_df in self.data_prep.device_map.items():
            if device_name == current_device_name:
                continue
            pruned = self.data_prep.prune_features(device_df)
            labeled = self.data_prep.label_device(pruned, 0)
            sampled = labeled.sample(
                n=min(records_per_class, len(labeled)),
                random_state=self.data_prep.random_state
            )
            sampled_dfs.append(sampled)

        return pd.concat(sampled_dfs, ignore_index=True)

    def add_device(self, current_device_name):
        """Train binary classifier for one device vs all others."""
        current_device_path = f"{PREPROCESSED_DATA_DIRECTORY}/{current_device_name}.csv"
        device_df = pd.read_csv(current_device_path)
        device_df = self.data_prep.label_device(
            self.data_prep.prune_features(device_df), 1
        )

        num_records = len(device_df)
        records_per_class = max(1, num_records // max(1, self.num_classes - 1))

        false_class = self.sample_false_class(current_device_name, records_per_class)
        input_data = pd.concat([device_df, false_class], ignore_index=True)

        model, evaluation = self.train_classifier(input_data)
        self.models[current_device_name] = model
        self.evaluations[current_device_name] = evaluation

    def train_all(self):
        """Train one model per device."""
        for device_name in self.device_csvs:
            print(f"\n🔹 Training model for device: {device_name}")
            self.add_device(device_name)

    def train_classifier(self, input_data):
        """Train RandomForest on binary (device vs others) data."""
        X = input_data.drop(columns=["label"])
        y = input_data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.data_prep.test_size,
            random_state=self.data_prep.random_state,
            stratify=y,
        )

        # Scale features if needed
        X_train, X_test = self.data_prep.scale(X_train, X_test, X.columns)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        evaluation = self.evaluate(model, X_train, y_train, X_test, y_test)
        return model, evaluation

    def evaluate(self, model, X_train, y_train, X_test, y_test):
        """Compute metrics and print evaluation summary."""
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Train Accuracy: {train_acc:.4f}")

        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Validation/Test Accuracy: {test_acc:.4f}")

        report = classification_report(y_test, y_test_pred)
        print("\nClassification Report (Test):")
        print(report)

        cm = confusion_matrix(y_test, y_test_pred)
        print("Confusion Matrix (Test):")
        print(cm)

        return train_acc, test_acc, report, cm

    def save(self):
        current_date = datetime.today().strftime('%Y-%m-%d')
        current_directory = os.path.join(self.output_directory, current_date)
        os.makedirs(current_directory, exist_ok=True)

        model_ref = len(os.listdir(current_directory))
        current_model_dir = os.path.join(current_directory, f"binary_model_{model_ref}")
        os.makedirs(current_model_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = os.path.join(current_model_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)

        eval_path = os.path.join(current_model_dir, "z_evaluation.txt")
        train_acc_sum, test_acc_sum = 0, 0
        with open(eval_path, "w") as file:
            for model_name, (train_acc, test_acc, report, cm) in self.evaluations.items():
                train_acc_sum += train_acc
                test_acc_sum += test_acc
                file.write(f"\n=== {model_name} ===\n")
                file.write(f"Train accuracy: {train_acc:.4f}\n")
                file.write(f"Test accuracy: {test_acc:.4f}\n")
                file.write("Classification report:\n")
                file.write(report + "\n")
                file.write("Confusion matrix:\n")
                file.write(str(cm) + "\n")
            avg_train_acc = train_acc_sum / self.num_classes
            avg_test_acc = test_acc_sum / self.num_classes
            file.write("\n=== combined accuracy ===\n")
            file.write(f"Train accuracy: {avg_train_acc:.4f}")
            file.write(f"Test accuracy: {avg_test_acc:.4f}")
        print(f"\n Models and evaluations saved to: {current_model_dir}")


def main():
    binary_model = BinaryModel()
    binary_model.train_all()
    binary_model.save()


if __name__ == "__main__":
    main()
