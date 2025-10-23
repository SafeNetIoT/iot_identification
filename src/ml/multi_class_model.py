from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord
import os
import pandas as pd


class MultiClassModel(Manager):
    """Trains a single multiclass model for all devices combined. Still WIP. Gives okay results"""
    def __init__(self, architecture_name="standard_forest", manager_name="multiclass_model", output_dir=None, loading_dir=None):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name, output_directory=output_dir, loading_directory=loading_dir)

    def run(self):
        data = []
        for device_name, sessions in self.device_sessions.items():
            sessions = [self.data_prep.label_device(session, device_name) for session in sessions]
            data.extend(sessions)
        record = ModelRecord(name="multiclass_model", data=data)
        self.records.append(record)
        self.train_all()
        self.save_all(save_input_data=True)
        # self.train_classifier(record)
        # self.save_classifier(record, save_input_data=True)

    def multi_predict(self, pcap_file): # if more than 1
        model_arr = self.load_model()
        model = model_arr[0].model
        df = self.fast_extractor.extract_features(pcap_file)
        return model.predict(df)


def main(): # still shows incorrect results
    # manager = MultiClassModel()
    # manager.run()

    from pathlib import Path
    from config import RAW_DATA_DIRECTORY
    from pandas.errors import EmptyDataError
    manager = MultiClassModel(loading_dir="models/2025-10-23/multiclass_model8")
    prev = ""
    for subdir in Path(RAW_DATA_DIRECTORY).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.rglob("*"):
            if f.is_file():
                try:
                    device = str(f).split("/")[2]
                    if device == prev:
                        continue
                    print("file:", f)
                    res = manager.multi_predict(str(f))
                    print("prediction", res)
                    print()
                except EmptyDataError:
                    continue
                except ValueError:
                    continue
                prev = device

if __name__ == "__main__":
    main()
