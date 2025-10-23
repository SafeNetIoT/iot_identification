from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord
import os
import pandas as pd


class MultiClassModel(Manager):
    """Trains a single multiclass model for all devices combined."""
    def __init__(self, architecture_name="standard_forest", manager_name="multiclass_model", output_dir=None, loading_dir=None):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name, output_directory=output_dir, loading_directory=loading_dir)

    def add_device(self, data):
        record = ModelRecord(self.manager_name, data)
        self.records.append(record)

    def preprocess(self):   
        all_devices = []
        for device_csv in os.listdir(self.data_prep.preprocessed_data_directory):
            device_df = pd.read_csv(f"{self.data_prep.preprocessed_data_directory}/{device_csv}")
            device_name = device_csv.split(".csv")[0]
            device_df = self.data_prep.prepare_df(device_df, device_name)
            all_devices.append(device_df)
        data = pd.concat(all_devices, ignore_index=True)
        self.add_device(data)


def main():
    # manager = MultiClassModel()
    # manager.preprocess()
    # manager.train_all()
    # manager.save_all()

    manager = MultiClassModel(output_directory="models/2025-10-17/multiclass_model1")
    manager.predict("data/raw/alexa_swan_kettle/2023-10-19/2023-10-19_00:02:55.402s.pcap")

if __name__ == "__main__":
    main()
