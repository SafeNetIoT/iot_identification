from src.ml.binary_model import BinaryModel
from src.ml.multi_class_model import MultiClassModel
import pandas as pd
from config import PREPROCESSED_DATA_DIRECTORY, RAW_DATA_DIRECTORY, TIME_INTERVALS
import pytest
from tests.helpers import run_model_workflow_test
from src.collection_time_analysis import TestPipeline

@pytest.mark.integration
def test_multiclass_model(tmp_path):
    """
    Integration test for MultiClassModel using real preprocessed data.
    Ensures the workflow runs end-to-end and produces valid artifacts.
    """
    manager = MultiClassModel(manager_name="test_multiclass")
    manager.output_directory = tmp_path
    manager.data_prep.preprocessed_data_directory = PREPROCESSED_DATA_DIRECTORY
    run_model_workflow_test(manager, tmp_path)

@pytest.mark.integration
def test_binary_model(tmp_path):
    """End-to-end test for BinaryModel workflow."""
    manager = BinaryModel(manager_name="test_binary")
    manager.output_directory = tmp_path
    manager.data_prep.preprocessed_data_directory = PREPROCESSED_DATA_DIRECTORY
    run_model_workflow_test(manager, tmp_path)

@pytest.mark.integration
def test_time_collection_analysis(tmp_path):
    """
    Integration test for the entire TestPipeline:
    - Runs the extraction + training workflow for one interval.
    - Checks that data is combined and models are saved correctly.
    """
    pipeline = TestPipeline(verbose=False)
    pipeline.raw_data_directory = RAW_DATA_DIRECTORY
    pipeline.collection_times = TIME_INTERVALS[:1]  # test only first interval for speed

    combined_df = pipeline.combine_csvs(pipeline.collection_times[0])
    assert isinstance(combined_df, pd.DataFrame)
    assert not combined_df.empty, "Combined DataFrame is empty after feature extraction"
    assert all(col in combined_df.columns for col in ["device", "label"]), \
        "Missing expected columns after preprocessing"

    manager = MultiClassModel(manager_name="test_pipeline_multiclass")
    manager.output_directory = tmp_path
    manager.add_device(combined_df)

    run_model_workflow_test(manager, tmp_path)
