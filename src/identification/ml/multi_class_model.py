import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from config import PREPROCESSED_DATA_DIRECTORY, VALID_FEATURES_DIRECTORY, MODELS_DIRECTORY
from datetime import datetime
from src.identification.ml.dataset_preparation import DatasetPreparation

class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=2
        )
        self.output_directory = MODELS_DIRECTORY
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        self.cv_results = None

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self):
        # Training accuracy
        y_train_pred = self.model.predict(self.X_train)
        self.train_acc = accuracy_score(self.y_train, y_train_pred)
        print(f"Train Accuracy: {self.train_acc:.4f}")

        # Validation/Test accuracy 
        y_test_pred = self.model.predict(self.X_test)
        self.test_acc = accuracy_score(self.y_test, y_test_pred)
        print(f"Validation/Test Accuracy: {self.test_acc:.4f}")

        # report
        self.report = classification_report(self.y_test, y_test_pred)
        print("\nClassification Report (Test):")
        print(self.report)

        # confusion matrix
        self.confusion_matrix = confusion_matrix(self.y_test, y_test_pred)
        print("Confusion Matrix (Test):")
        print(self.confusion_matrix)

    def tune(self, n_iter=5, cv=5):
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None] + list(range(5, 31, 5)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }

        rand_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        rand_search.fit(self.X_train, self.y_train)
        self.model = rand_search.best_estimator_

        # save cross validation granular results
        self.cv_results = pd.DataFrame(rand_search.cv_results_)
        cols = [c for c in self.cv_results.columns if "split" in c and "test_score" in c]
        print(self.cv_results[cols + ["mean_test_score", "std_test_score"]])
        return self.model

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
            file.write(f"train accuracy: {self.train_acc}\n")
            file.write(f"test accuracy: {self.test_acc}\n")
            file.write(f"report:\n{self.report}\n")
            file.write(f"confusion_matrix:\n{self.confusion_matrix}")
        if self.cv_results is not None:
            self.cv_results.to_csv(f"{current_model_dir}/cross_validation.csv")
        print(f"Model saved to {current_model_dir}")

def main():
    prep = DatasetPreparation()
    _, X_test, _, y_test = prep.preprocess()
    X_train_bal, y_train_bal = prep.balance_dataset()
    clf = Model(X_train=X_train_bal, y_train=y_train_bal, X_test=X_test, y_test=y_test)
    # clf.tune(3, 3)
    clf.train()
    clf.evaluate()
    clf.save()

if __name__ == "__main__":
    main()