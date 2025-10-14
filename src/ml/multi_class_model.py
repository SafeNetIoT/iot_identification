from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord
import os
import pandas as pd


class MultiClassModel(Manager):
    """Trains a single multiclass model for all devices combined."""
    def __init__(self, architecture_name="standard_forest", manager_name="multiclass_model", output_directory=None):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name, output_directory=output_directory)

    def add_device(self, data):
        record = ModelRecord(self.manager_name, data)
        self.records.append(record)

    def preprocess(self):   
        all_devices = []
        for device_csv in os.listdir(self.data_prep.preprocessed_data_directory):
            device_df = pd.read_csv(f"{self.data_prep.preprocessed_data_directory}/{device_csv}")
            device_name = device_csv.split(".csv")[0]
            # device_df = self.data_prep.label_device(self.data_prep.prune_features(device_df), device_name)
            # device_df = self.data_prep.clean_up(device_df)
            device_df = self.data_prep.prepare_df(device_df, device_name)
            all_devices.append(device_df)
        data = pd.concat(all_devices, ignore_index=True)
        self.add_device(data)


def main():
    manager = MultiClassModel()
    manager.preprocess()
    manager.train_all()
    manager.save_all()

if __name__ == "__main__":
    main()
