
class BaseModel:
    def __init__(self, name) -> None:
        self.name = name

    def train(self):
        pass

    def tune(self):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass