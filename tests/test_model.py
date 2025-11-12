import pytest
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
import joblib 
import random
from conftest import DummyModel
from datetime import datetime
from tests.helpers import _run_unseen_evaluation

@pytest.mark.integration
def test_slow_pipeline(binary_model, tmp_path):
    """
    Integration-style test verifying:
    - Directory creation pattern (date/manager_nameN/)
    - z_evaluation.txt is generated
    - Pickle files saved correctly
    """
    binary_model.output_directory = tmp_path
    binary_model.device_sessions = {
        "deviceA": ["sessionA1", "sessionA2"],
        "deviceB": ["sessionB1"],
        "deviceC": ["sessionC1"],
    }
    binary_model.data_prep = MagicMock()
    binary_model.data_prep.label_device.side_effect = lambda s, l: f"labeled_{s}"
    binary_model.sample_false_class = MagicMock(return_value=["fake_false"])
    binary_model.records = []

    def fake_train(record, show_curve=False):
        record.model = DummyModel()
        record.evaluation = {
            "train_acc": 1.0,
            "test_acc": 1.0,
            "report": "OK",
            "conf_matrix": [[1]],
        }

    binary_model.train_classifier = fake_train

    binary_model.prepare_datasets()
    binary_model.train_all()
    binary_model.save_all()

    # check directory structure
    today = datetime.today().strftime("%Y-%m-%d")
    expected_base = tmp_path / today
    assert expected_base.exists(), f"Base directory {expected_base} not created."

    # Should be a subfolder like "binary_model" or "binary_model1"
    subdirs = list(expected_base.iterdir())
    assert len(subdirs) == 1, f"Expected 1 subdir in {expected_base}, found {len(subdirs)}"
    model_dir = subdirs[0]
    assert model_dir.is_dir(), f"{model_dir} is not a directory"

    # Check for model files
    model_files = list(model_dir.glob("*.pkl"))
    expected_count = len(binary_model.device_sessions)
    assert len(model_files) == expected_count, (
        f"Expected {expected_count} model pickle files, found {len(model_files)}"
    )

    # check evaluation file
    eval_file = model_dir / "z_evaluation.txt"
    assert eval_file.exists(), f"{eval_file} not found"
    text = eval_file.read_text()
    for name in binary_model.device_sessions.keys():
        assert name in text, f"Device name {name} not found in evaluation file"

    # verify pickle contents
    for file in model_files:
        model = joblib.load(file)
        assert isinstance(model.model, RandomForestClassifier)

@pytest.mark.integration
def test_unseen(binary_model_under_test):
    _run_unseen_evaluation(binary_model_under_test, binary_model_under_test.predict)

def test_add_device_missing_directory(binary_model):
    with patch("src.ml.binary_model.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            binary_model.add_device("deviceX", "fake/path")

# @pytest.mark.integration
# def test_unseen_multiclass(multiclass_model_under_test):
#     _run_unseen_evaluation(multiclass_model_under_test, multiclass_model_under_test.multi_predict)
            