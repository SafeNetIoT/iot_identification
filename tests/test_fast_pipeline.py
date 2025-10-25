import pytest
import pandas as pd


def test_add_device_creates_record(binary_model, tmp_path):
    """Ensure add_device builds dataset, appends record, and trains/saves model."""
    device_name = "deviceA"
    device_dir = tmp_path / device_name
    device_dir.mkdir()

    # Create fake pcap files
    pcap1 = device_dir / "a.pcap"
    pcap2 = device_dir / "b.pcap"
    pcap1.touch()
    pcap2.touch()

    # Mock extract_features to return fake DataFrames
    df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})
    binary_model.fast_extractor.extract_features.side_effect = [df1, df2]

    # Mock label_device to add a 'label' column
    binary_model.data_prep.label_device.side_effect = lambda df, lbl: df.assign(label=lbl)

    binary_model.add_device(device_name, device_dir)

    assert binary_model.fast_extractor.extract_features.call_count == 2
    assert binary_model.data_prep.label_device.call_count == 2

    binary_model.sample_false_class.assert_called_once_with(device_name, 2)

    assert len(binary_model.records) == 1
    record = binary_model.records[0]
    assert record.name == device_name
    assert any(isinstance(x, str) and x == "fake_false" for x in record.data)

    # Should have trained and saved the classifier
    binary_model.train_classifier.assert_called_once_with(record, show_curve=True)
    binary_model.save_classifier.assert_called_once_with(record)

def test_add_device_missing_dir(binary_model):
    with pytest.raises(FileNotFoundError):
        binary_model.add_device("missing_device", "nonexistent/path")

def test_add_device_skips_empty_sessions(binary_model, tmp_path):
    device_dir = tmp_path / "deviceB"
    device_dir.mkdir()
    (device_dir / "x.pcap").touch()

    # extract_features returns empty DataFrame
    binary_model.fast_extractor.extract_features.return_value = pd.DataFrame()

    binary_model.add_device("deviceB", device_dir)
    assert len(binary_model.records) == 1  # Still creates record with false class only
