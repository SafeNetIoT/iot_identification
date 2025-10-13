import os
import pandas as pd
from config import PREPROCESSED_DATA_DIRECTORY
import pandas as pd
import os
from src.ml.model_manager import Manager
from src.ml.model_record import ModelRecord


class BinaryModel(Manager):
    """Trains one binary classifier per device (device vs all others)."""

    def __init__(self, architecture_name="standard_forest", manager_name="binary_model"):
        super().__init__(architecture_name=architecture_name, manager_name=manager_name)
        self.device_csvs = [f.replace(".csv", "") for f in os.listdir(PREPROCESSED_DATA_DIRECTORY) if f.endswith(".csv")]
        self.num_classes = len(self.device_csvs)

    def sample_false_class(self, current_device_name, records_per_class):
        sampled_dfs = []
        for device_name in self.device_csvs:
            if device_name == current_device_name:
                continue
            device_path = f"{PREPROCESSED_DATA_DIRECTORY}/{device_name}.csv"
            device_df = pd.read_csv(device_path)
            pruned = self.data_prep.prune_features(device_df)
            labeled = self.data_prep.label_device(pruned, 0)
            sampled = labeled.sample(
                n=min(records_per_class, len(labeled)),
                random_state=self.random_state
            )
            sampled_dfs.append(sampled)
        return pd.concat(sampled_dfs, ignore_index=True)

    def prepare_true_class(self, current_device_name):
        current_device_path = f"{PREPROCESSED_DATA_DIRECTORY}/{current_device_name}.csv"
        device_df = pd.read_csv(current_device_path)
        pruned = self.data_prep.prune_features(device_df)
        labeled = self.data_prep.label_device(pruned, 1)
        return labeled

    def prepare_datasets(self):
        for device_name in self.device_csvs:
            pos_df = self.prepare_true_class(device_name)
            records_per_class = max(1, len(pos_df) // max(1, self.num_classes - 1))
            neg_df = self.sample_false_class(device_name, records_per_class)
            data = pd.concat([pos_df, neg_df], ignore_index=True)
            record = ModelRecord(name=device_name, data=data)
            self.records.append(record)

def main():
    manager = BinaryModel()
    manager.prepare_datasets()
    manager.train_all()
    manager.save_all()


if __name__ == "__main__":
    main()
