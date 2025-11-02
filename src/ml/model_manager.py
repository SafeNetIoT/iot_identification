import os
from src.ml.base_model import BaseModel
from config import MODEL_ARCHITECTURES, RANDOM_STATE, MODELS_DIRECTORY
from src.ml.dataset_preparation import DatasetPreparation
from typing import List
from datetime import datetime
import joblib
from src.ml.model_record import ModelRecord
from src.features.fast_extraction import FastExtractionPipeline
from src.ml.cache import Cache
import random

class Manager:
    def __init__(self, architecture_name="standard_forest", manager_name="random_forest", output_directory=None, loading_directory=None):
        self.architecture = MODEL_ARCHITECTURES[architecture_name]
        self.data_prep = DatasetPreparation()
        self.records: List[ModelRecord] = []
        self.random_state = RANDOM_STATE            
        self.output_directory = output_directory if output_directory is not None else MODELS_DIRECTORY
        self.loading_directory = loading_directory
        self.total_train_acc, self.total_test_acc = 0, 0
        self.manager_name = manager_name
        self.fast_extractor = FastExtractionPipeline()
        self.model_directory = None
        random.seed(self.random_state)
        self.cache = Cache()
        # self.device_sessions, self.unseen_sessions = self.cache.build()

    def set_cache(self, cache = None):
        if cache is not None:
            self.cache = cache
        self.device_sessions, self.unseen_sessions = self.cache.build()
        return self.device_sessions, self.unseen_sessions

    def set_device_sessions(self, sessions_map):
        self.device_sessions = sessions_map
        return
    
    def reset_training_attributes(self):
        self.total_train_acc, self.total_test_acc = 0, 0
        self.records = []
        return

    def train_classifier(self, record, show_curve = False):
        clf = BaseModel(self.architecture, record.data, record.name)
        clf.split()
        clf.scale()
        clf.train()
        clf.evaluate()
        record.model = clf
        record.evaluation = {
            "train_acc": clf.train_acc,
            "test_acc": clf.test_acc,
            "report": clf.report,
            "confusion_matrix": clf.confusion_matrix,
        }
        if show_curve:
            clf.plot_learning_curve()

    def train_all(self):
        for record in self.records:
            self.train_classifier(record)
        print("\n Training complete.")

    def create_model_directory(self):
        current_date = datetime.today().strftime('%Y-%m-%d')
        current_directory = f"{self.output_directory}/{current_date}"
        os.makedirs(current_directory, exist_ok=True)
        model_ref = str(len(os.listdir(current_directory)))
        if model_ref == '0': model_ref = ''
        current_model_dir = f"{current_directory}/{self.manager_name}{model_ref}"
        os.makedirs(current_model_dir)
        self.model_directory = current_model_dir

    def save_evaluation(self, record: ModelRecord):
        name = record.name
        train_acc, test_acc, report, conf_matrix = record.evaluation.values()
        with open(f"{self.model_directory}/z_evaluation.txt", 'a') as file:
            file.write(f"\n=== {name} ===\n")
            file.write(f"train accuracy: {train_acc}\n")
            file.write(f"test accuracy: {test_acc}\n")
            file.write(f"report:\n{report}\n")
            file.write(f"confusion_matrix:\n{conf_matrix}")
            file.write("\n")
        return train_acc, test_acc

    def save_average_accuracies(self):
        avg_train_acc = self.total_train_acc / len(self.records)
        avg_test_acc = self.total_test_acc / len(self.records)
        with open(f"{self.model_directory}/z_evaluation.txt", 'a') as file:
            file.write("\n=== Average Accuracies ===\n")
            file.write(f"Average Train Accuracy: {avg_train_acc:.4f}\n")
            file.write(f"Average Test Accuracy: {avg_test_acc:.4f}\n")

    def save_classifier(self, record, save_input_data = False):
        if self.model_directory is None:
            self.create_model_directory()
        model = record.model
        name = record.name
        joblib.dump(model, f"{self.model_directory}/{name}.pkl")
        train_acc, test_acc = self.save_evaluation(record)
        self.total_train_acc += train_acc
        self.total_test_acc += test_acc
        if save_input_data:
            model.X_test.to_csv(f"{self.model_directory}/input.csv")
            model.y_test.to_csv(f"{self.model_directory}/output.csv")
        if model.cv_results is not None:
            model.cv_results.to_csv(f"{self.model_directory}/cross_validation.csv")

    def save_all(self, save_input_data = False):
        if os.path.isfile(self.output_directory):
            raise ValueError("output_directory is a file")
        self.create_model_directory()
        for record in self.records:
            model = record.model
            name = record.name
            joblib.dump(model, f"{self.model_directory}/{name}.pkl")
            print(f"saved {model} to {self.model_directory}/{name}.pkl")
            train_acc, test_acc = self.save_evaluation(record)
            self.total_train_acc += train_acc
            self.total_test_acc += test_acc
            if save_input_data:
                model.X_test.to_csv(f"{self.model_directory}/input.csv")
                model.y_test.to_csv(f"{self.model_directory}/output.csv")
            if model.cv_results is not None:
                model.cv_results.to_csv(f"{self.model_directory}/cross_validation.csv")
        if len(self.records) > 1:
            self.save_average_accuracies()

    def load_model(self):
        if self.loading_directory is None: 
            raise ValueError("Loading directory has not been specified")
        if not os.path.exists(self.loading_directory):
            raise FileNotFoundError("Model has to be saved before it is loaded")
        return [joblib.load(f"{self.loading_directory}/{file}") for file in os.listdir(self.loading_directory) if file.endswith(".pkl")]

if __name__ == "__main__":
    manager = Manager()
        


