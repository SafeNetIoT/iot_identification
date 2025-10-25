from src.ml.binary_model import BinaryModel
from config import RAW_DATA_DIRECTORY, MODEL_UNDER_TEST, DESIRED_ACCURACY, TEST_FRACTION, RANDOM_STATE
import pytest
from pathlib import Path
from pandas.errors import EmptyDataError
from unittest.mock import patch, MagicMock
from tests.helpers import assert_save_calls, assert_save_paths
from sklearn.ensemble import RandomForestClassifier
import joblib 
import random
from tests.dummy_model import DummyModel
from datetime import datetime

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
def test_unseen():
    random.seed(RANDOM_STATE)
    correct, total = 0, 0
    manager = BinaryModel(loading_dir=MODEL_UNDER_TEST)
    for device_name, pcap_list in manager.unseen_sessions.items():
        if not pcap_list:
            continue
        print("pcap_list length:", len(pcap_list))
        n_samples = max(1, int(len(pcap_list) * TEST_FRACTION))
        sampled_pcaps = random.sample(pcap_list, n_samples)
        for pcap_path in sampled_pcaps:
            try:
                prediction = manager.predict(str(pcap_path))
                print("device name:", device_name)
                print("prediction:", prediction)
            except EmptyDataError:
                continue
            if prediction == device_name:
                correct += 1
            total += 1
    acc = correct / total
    print(acc)
    assert acc >= DESIRED_ACCURACY

def test_add_device_missing_directory(binary_model):
    with patch("src.ml.binary_model.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            binary_model.add_device("deviceX", "fake/path")

@patch("src.ml.binary_model.Path.exists", return_value=True)
@patch("src.ml.binary_model.Path.rglob")
@patch("src.ml.binary_model.ModelRecord")
def test_add_device_happy_path(mock_modelrecord, mock_rglob, _, binary_model):
    # Fake PCAP file paths
    fake_pcaps = [Path("file1.pcap"), Path("file2.pcap"), Path("file3.pcap")]
    mock_rglob.return_value = fake_pcaps

    # Mock feature extraction: return non-empty for first 2, empty for last
    non_empty_df = MagicMock()
    non_empty_df.empty = False
    empty_df = MagicMock()
    empty_df.empty = True

    binary_model.fast_extractor.extract_features.side_effect = [
        non_empty_df, non_empty_df, empty_df
    ]

    # Mock labeling
    binary_model.data_prep.label_device.side_effect = lambda df, label: f"labeled_{id(df)}"

    binary_model.add_device("deviceA", "fake/path", verbose=True)
    assert binary_model.fast_extractor.extract_features.call_count == len(fake_pcaps)
    assert binary_model.data_prep.label_device.call_count == 2

    binary_model.sample_false_class.assert_called_once_with("deviceA", 2)
    mock_modelrecord.assert_called_once()
    record_call_args = mock_modelrecord.call_args[0]
    assert record_call_args[0] == "deviceA"  # name
    assert "fake_false" in record_call_args[1]  # dataset includes false class

    record_instance = mock_modelrecord.return_value
    binary_model.train_classifier.assert_called_once_with(record_instance, show_curve=True)
    binary_model.save_classifier.assert_called_once_with(record_instance)
    assert record_instance in binary_model.records

