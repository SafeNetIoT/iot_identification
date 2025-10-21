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
        # self._verify_schema()

    def _verify_schema(self):
        # if not isinstance(self.data, list(pd.DataFrame)):
        #     raise ValueError("Input datatype is not a List[pd.DataFrame]:" + str(type(self.data)))
        feature_columns = unpack_features()
        feature_columns.append('label')
        if list(self.data.columns) != feature_columns:
            raise ValueError("Input data schema does not match the requirements")

    def split(self):
        if not isinstance(self.data, list):
            raise TypeError("self.data must be a list of session DataFrames (each containing a 'label' column).")

        session_labels = [df["label"].iloc[0] for df in self.data]
        train_sessions, test_sessions = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=session_labels,
        )

        train_merged = pd.concat(train_sessions, ignore_index=True)
        test_merged = pd.concat(test_sessions, ignore_index=True)

        all_data = pd.concat([train_merged, test_merged], ignore_index=True)
        self.X = all_data.drop(columns=["label"])
        self.y = all_data["label"]

        self.X_train = train_merged.drop(columns=["label"])
        self.y_train = train_merged["label"]
        self.X_test = test_merged.drop(columns=["label"])
        self.y_test = test_merged["label"]
        print("split training data")

        train_paths = {df.attrs.get("pcap_path") for df in train_sessions}
        test_paths  = {df.attrs.get("pcap_path") for df in test_sessions}
        overlap = train_paths & test_paths
        print(overlap)

    def scale(self):
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)

    def balance_dataset(self, verbose = False):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run preprocess() before balancing.")

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(self.X_train, self.y_train)

        if verbose:
            print("Before balancing:", self.y_train.value_counts().to_dict())
            print("After balancing:", pd.Series(y_resampled).value_counts().to_dict())

        self.X_train, self.y_train = X_resampled, y_resampled
        return self.X_train, self.y_train

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def tune(self, n_iter=5, cv=5, estimator_range=[100, 500], max_depth_range=[5, 31, 5], min_samples_split_range=[2, 20],
        min_samples_leaf_range=[1,10], max_features = ['sqrt', 'log2', None], scoring='accuracy', n_jobs=-1, verbose=2):
        param_dist = {
            'n_estimators': randint(estimator_range),
            'max_depth': [None] + list(range(max_depth_range)),
            'min_samples_split': randint(min_samples_split_range),
            'min_samples_leaf': randint(min_samples_leaf_range),
            'max_features': max_features
        }

        rand_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state
        )

        rand_search.fit(self.X_train, self.y_train)
        self.model = rand_search.best_estimator_

        self.cv_results = pd.DataFrame(rand_search.cv_results_)
        cols = [c for c in self.cv_results.columns if "split" in c and "test_score" in c]
        print(self.cv_results[cols + ["mean_test_score", "std_test_score"]])
        return self.model

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
            print("X is None")
            return 
        if self.y is None:
            print("y is None")
            return 
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