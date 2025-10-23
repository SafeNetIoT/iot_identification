from src.ml.binary_model import BinaryModel
from config import RAW_DATA_DIRECTORY, MODEL_UNDER_TEST, DESIRED_ACCURACY
import pytest
from pathlib import Path
from pandas.errors import EmptyDataError
from unittest.mock import patch, MagicMock
from tests.helpers import assert_save_calls, assert_save_paths
from sklearn.ensemble import RandomForestClassifier
import joblib 

@pytest.mark.integration
def test_slow_pipeline(binary_model, tmp_path):
    """
    Integration-style test verifying:
    1. Output directory is created.
    2. Pickle files + evaluation.txt are generated.
    3. Number of pickle files equals number of devices.
    """
    binary_model.output_directory = tmp_path
    binary_model.device_sessions = {
        "deviceA": ["sessionA1", "sessionA2"],
        "deviceB": ["sessionB1"],
        "deviceC": ["sessionC1"]
    }

    binary_model.data_prep = MagicMock()
    binary_model.data_prep.label_device.side_effect = lambda s, l: f"labeled_{s}"
    binary_model.sample_false_class = MagicMock(return_value=["fake_false"])
    binary_model.records = []

    def fake_train(record, show_curve=False):
        record.model = RandomForestClassifier()
    binary_model.train_classifier = fake_train

    # Fake saving (writes a pickle and a single evaluation.txt)
    def fake_save(record):
        model_path = tmp_path / f"{record.name}.pkl"
        joblib.dump(record.model, model_path)

        # Write an evaluation file once (idempotent)
        eval_path = tmp_path / "evaluation.txt"
        if not eval_path.exists():
            eval_path.write_text("accuracy=0.95")
            
    binary_model.save_classifier = fake_save

    binary_model.prepare_datasets()
    binary_model.train_all()
    binary_model.save_all()

    assert tmp_path.exists() and tmp_path.is_dir(), "Output directory not created"

    model_files = list(tmp_path.glob("*.pkl"))
    eval_file = tmp_path / "evaluation.txt"
    assert eval_file.exists(), "evaluation.txt not found"
    assert len(model_files) > 0, "No model pickle files found"

    expected_count = len(binary_model.device_sessions)
    assert len(model_files) == expected_count, (
        f"Expected {expected_count} model files, found {len(model_files)}"
    )
    for file in model_files:
        model = joblib.load(file)
        assert isinstance(model, RandomForestClassifier), f"{file.name} is not a RandomForestClassifier"

@pytest.mark.integration
def test_workflow():
    binary_model = BinaryModel(loading_dir=MODEL_UNDER_TEST)
    num_correct_predictions, total_predictions = 0, 0
    binary_model.prepare_datasets()
    binary_model.train_all()

    for subdir in Path(RAW_DATA_DIRECTORY).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.rglob("*"):
            if f.is_file():
                truth = str(f).split("/")[2]
                try:
                    prediction = binary_model.predict(str(f))
                except EmptyDataError:
                    continue
                print("Prediction:", prediction)
                print("Truth:", truth)
                if prediction == truth:
                    num_correct_predictions += 1
                total_predictions += 1
                break
    accuracy = num_correct_predictions / total_predictions
    assert accuracy > DESIRED_ACCURACY

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

