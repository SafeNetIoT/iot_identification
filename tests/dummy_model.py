from sklearn.ensemble import RandomForestClassifier

class DummyModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.cv_results = None
        self.train_acc = 1.0
        self.test_acc = 1.0
        self.report = "OK"
        self.confusion_matrix = [[1]]
        self.X_test = None
        self.y_test = None