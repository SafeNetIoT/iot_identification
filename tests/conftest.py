import pytest
from unittest.mock import MagicMock, patch
from src.ml.binary_model import BinaryModel
from src.ml.multi_class_model import MultiClassModel
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_UNDER_TEST, MULTICLASS_MODEL_UNDER_TEST

@pytest.fixture
def fake_device_sessions():
    """Provides a minimal mock device_sessions structure."""
    return {
        "deviceA": ["df1", "df2"],
        "deviceB": ["df3"],
    }

@pytest.fixture
def fake_data_prep():
    """Mocks the data prep component with a predictable labeling function."""
    mock = MagicMock()
    mock.label_device.side_effect = lambda s, l: f"labeled_{s}"
    return mock

@pytest.fixture
def binary_model():
    """Creates a BinaryModel with mocked dependencies and a temp output dir."""
    model = BinaryModel()
    model.fast_extractor = MagicMock()
    model.data_prep = MagicMock()
    model.sample_false_class = MagicMock(return_value=["fake_false"])
    model.train_classifier = MagicMock()
    model.save_classifier = MagicMock()
    model.records = []
    return model

@pytest.fixture
def binary_model_under_test():
    """Creates a binary model instance with a loaded model specified in config"""
    model = BinaryModel(loading_dir=MODEL_UNDER_TEST)
    return model

@pytest.fixture
def multiclass_model_under_test():
    model = MultiClassModel(loading_dir=MULTICLASS_MODEL_UNDER_TEST)
    return model

@pytest.fixture
def mock_modelrecord():
    """Mocks ModelRecord to intercept dataset creation."""
    with patch("src.ml.binary_model.ModelRecord") as mock_record_cls:
        mock_record = MagicMock()
        mock_record_cls.return_value = mock_record
        yield mock_record_cls

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