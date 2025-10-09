from src.identification.ml.model_manager import Manager
from src.identification.ml.dataset_preparation import DatasetPreparation


class MultiClassModel(Manager):
    """Trains a single multiclass model for all devices combined."""
    def __init__(self, input_data, architecture_name="standard_forest", name="multiclassForest"):
        super().__init__(input_data, architecture_name, name)


def main():
    data_prep = DatasetPreparation()
    input_data = [data_prep.combine_csvs()]
    print(type(input_data))
    manager = MultiClassModel(input_data)
    manager.train_all()
    # print(manager.summary())
    manager.save_all()

if __name__ == "__main__":
    main()
