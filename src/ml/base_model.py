from config import MODELS_DIRECTORY, RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import unpack_features
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from exceptions import DataLeakageError, PipelineStateError
from src.ml.tune_config import TuneConfig

class BaseModel:
    def __init__(self, architecture: Dict, input_data: List[pd.DataFrame], name: str, test_size: float = 0.2) -> None:
        self.name = name
        self.output_directory = MODELS_DIRECTORY
        self.model = RandomForestClassifier(**architecture)
        self.data = input_data
        self.train_acc, self.test_acc, self.report, self.confusion_matrix = None, None, None, None
        self.X, self.y = None, None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.cv_results = None
        self.test_size = test_size
        self.random_state = RANDOM_STATE
        self._verify_schema()

    def _verify_schema(self):
        if not isinstance(self.data, list):
            raise TypeError(f"`data` must be a list, not {type(self.data)}")

        if not all(isinstance(df, pd.DataFrame) for df in self.data):
            bad_types = [type(df) for df in self.data if not isinstance(df, pd.DataFrame)]
            raise TypeError(f"All elements in `data` must be pandas DataFrames, got {bad_types}")

        feature_columns = unpack_features()
        expected_columns = feature_columns + ["label"]
        for i, df in enumerate(self.data):
            if list(df.columns) != expected_columns:
                raise ValueError(
                    f"Schema mismatch in DataFrame {i}: "
                    f"expected columns {expected_columns}, got {list(df.columns)}"
                )

    def _split_sessions(self):
        session_labels = [df["label"].iloc[0] for df in self.data]
        train_sessions, test_sessions = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=session_labels,
        )
        return train_sessions, test_sessions

    def _prepare_train_test_data(self, train_sessions, test_sessions):
        train_merged = pd.concat(train_sessions, ignore_index=True)
        test_merged = pd.concat(test_sessions, ignore_index=True)

        all_data = pd.concat([train_merged, test_merged], ignore_index=True)
        self.X = all_data.drop(columns=["label"])
        self.y = all_data["label"]

        self.X_train = train_merged.drop(columns=["label"])
        self.y_train = train_merged["label"]
        self.X_test = test_merged.drop(columns=["label"])
        self.y_test = test_merged["label"]

    def _check_data_leakage(self, train_sessions, test_sessions):
        train_paths = {df.attrs.get("pcap_path") for df in train_sessions}
        test_paths = {df.attrs.get("pcap_path") for df in test_sessions}
        overlap = train_paths & test_paths
        overlap = {val for val in overlap if val != None}
        if overlap != set():
            raise DataLeakageError(f"Data leakage detected between training and test data: {overlap}")

    def split(self):
        train_sessions, test_sessions = self._split_sessions()
        self._prepare_train_test_data(train_sessions, test_sessions)
        self._check_data_leakage(train_sessions, test_sessions)

    def scale(self):
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)

    def balance_dataset(self, verbose = False):
        if self.X_train is None or self.y_train is None:
            raise PipelineStateError("Training data has not been initialized")

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(self.X_train, self.y_train)

        if verbose:
            print("Before balancing:", self.y_train.value_counts().to_dict())
            print("After balancing:", pd.Series(y_resampled).value_counts().to_dict())

        self.X_train, self.y_train = X_resampled, y_resampled

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def _build_param_distributions(self, estimator_range, max_depth_range, min_samples_split_range, min_samples_leaf_range, max_features):
        return {
            'n_estimators': randint(*estimator_range),
            'max_depth': [None] + list(range(*max_depth_range)),
            'min_samples_split': randint(*min_samples_split_range),
            'min_samples_leaf': randint(*min_samples_leaf_range),
            'max_features': list(max_features),
        }

    def _report_cv_results(self, verbose=False):
        if self.cv_results is None:
            return
        cols = [c for c in self.cv_results.columns if "split" in c and "test_score" in c]
        if verbose:
            print(self.cv_results[cols + ["mean_test_score", "std_test_score"]])
    
    def tune(self, tune_config: TuneConfig):
        param_dist = tune_config.param_distributions()
        search_kwargs = tune_config.random_search_kwargs()

        rand_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            **search_kwargs,
        )

        rand_search.fit(self.X_train, self.y_train)
        self.model = rand_search.best_estimator_
        self.cv_results = pd.DataFrame(rand_search.cv_results_)

    def evaluate(self, verbose = True):
        y_train_pred = self.model.predict(self.X_train)
        self.train_acc = accuracy_score(self.y_train, y_train_pred)

        y_test_pred = self.model.predict(self.X_test)
        self.test_acc = accuracy_score(self.y_test, y_test_pred)

        self.report = classification_report(self.y_test, y_test_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_test_pred)

        if verbose:
            print(f"Train Accuracy: {self.train_acc:.4f}")
            print(f"Validation/Test Accuracy: {self.test_acc:.4f}")
            print("\nClassification Report (Test):")
            print(self.report)
            print("Confusion Matrix (Test):")
            print(self.confusion_matrix)

    def plot_learning_curve(self):
        if self.X is None:
            raise PipelineStateError("Training data has not been initialized")
        if self.y is None:
            raise PipelineStateError("Training data has not been initialized")
        train_sizes, train_scores, test_scores = learning_curve(
        self.model, self.X, self.y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),  # 10% incr
        scoring="accuracy"
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        plt.plot(train_sizes, train_mean, label="Training score")
        plt.plot(train_sizes, test_mean, label="Validation score")
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def predict(self, X: pd.DataFrame) -> tuple[str, float]:
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        probas = self.model.predict_proba(X_scaled)[0]
        best_idx = probas.argmax()
        return self.model.classes_[best_idx], probas[best_idx]

if __name__ == "__main__":
    from config import MODEL_ARCHITECTURES
    df = pd.read_csv("src/identification/sample.csv")
    architecture = MODEL_ARCHITECTURES['standard_forest']
    model = BaseModel(architecture, df, "")
    print(1)