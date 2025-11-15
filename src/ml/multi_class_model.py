from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord
import pandas as pd


class MultiClassModel(Manager):
    """Trains a single multiclass model for all devices combined. Still WIP. Gives okay results"""
    def __init__(self, architecture_name="standard_forest", manager_name="multiclass_model", output_dir=None, loading_dir=None):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name, output_directory=output_dir, loading_directory=loading_dir)

    def run(self):
        self.device_sessions, self.unseen_sessions = self.set_cache()
        data = []
        for device_name, sessions in self.device_sessions.items():
            sessions = [self.data_prep.label_device(session, device_name) for session in sessions]
            data.extend(sessions)
        record = ModelRecord(name="multiclass_model", data=data)
        self.records.append(record)
        self.train_all()
        self.save_all(save_input_data=True)

    def predict(self, pcap_file):
        self.load_model()
        model = self.model_arr[0]
        df = self.fast_extractor.extract_features(pcap_file)
        if df.empty:
            return None
        df_scaled = pd.DataFrame(model.scaler.transform(df), columns=df.columns)
        probas = model.model.predict_proba(df_scaled)
        mean_proba = probas.mean(axis=0)
        best_idx = mean_proba.argmax()
        predicted_class = model.model.classes_[best_idx]
        # confidence = mean_proba[best_idx]
        return predicted_class


def main(): # still shows incorrect results
    # manager = MultiClassModel()
    # manager.run()

    manager = MultiClassModel(loading_dir="models/2025-10-25/multiclass_model5/")
    res = manager.predict("data/raw/alexa_swan_kettle/2023-10-19/2023-10-19_00:02:55.402s.pcap")
    print("res:", res)

if __name__ == "__main__":
    main()
