from src.identification.ml.model_manager import BaseManager


class MultiClassModel(BaseManager):
    """Trains a single multiclass model for all devices combined."""

    def prepare_datasets(self):
        input_data = self.data_prep.combine_csvs()
        return {"multi_class": input_data}


def main():
    manager = MultiClassModel()
    manager.train_all()
    print(manager.summary())
    manager.save_all()

if __name__ == "__main__":
    main()
