from config import MODELS_DIRECTORY, RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import unpack_features
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from typing import Dict


class BaseModel:
    def __init__(self, architecture: Dict, input_data: pd.DataFrame, name: str, test_size: float = 0.2) -> None:
        self.name = name
        self.output_directory = MODELS_DIRECTORY
        self.model = RandomForestClassifier(**architecture)
        self.data = input_data
        self.train_acc, self.test_acc, self.report, self.confusion_matrix = None, None, None, None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.cv_results = None
        self.test_size = test_size
        self.random_state = RANDOM_STATE
        self._verify_schema()

    def _verify_schema(self):
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Input datatype is not a pd.DataFrame:" + str(type(self.data)))
        
        feature_columns = unpack_features()
        feature_columns.append('label')
        if list(self.data.columns) != feature_columns:
            raise ValueError("Input data schema does not match the requirements")

    def split(self):
        X = self.data.drop(columns=["label"])
        y = self.data["label"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

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

if __name__ == "__main__":
    from config import MODEL_ARCHITECTURES
    df = pd.read_csv("src/identification/sample.csv")
    architecture = MODEL_ARCHITECTURES['standard_forest']
    model = BaseModel(architecture, df, "")
    print(1)