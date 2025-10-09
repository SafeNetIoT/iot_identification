from config import MODEL_ARCHITECTURES
from src.identification.ml.dataset_preparation import DatasetPreparation
from src.identification.ml.base_model import BaseModel

class MultiClassModel(BaseModel):
    def __init__(self, architecture, input_data, name, test_size=0.2) -> None:
        super().__init__(architecture, input_data, name, test_size)

    def predict(self, X):
        return self.model.predict(X)

def main():
    prep = DatasetPreparation()
    input_data = prep.combine_csvs()
    architecure = MODEL_ARCHITECTURES['standard_forest']
    name = "MultiClass"
    clf = MultiClassModel(architecture=architecure, input_data=input_data, name=name)
    clf.train()
    clf.evaluate()
    clf.save()

if __name__ == "__main__":
    main()