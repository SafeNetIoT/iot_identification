import pytest
from unittest.mock import MagicMock, patch
from src.ml.binary_model import BinaryModel

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
def mock_modelrecord():
    """Mocks ModelRecord to intercept dataset creation."""
    with patch("src.ml.binary_model.ModelRecord") as mock_record_cls:
        mock_record = MagicMock()
        mock_record_cls.return_value = mock_record
        yield mock_record_cls