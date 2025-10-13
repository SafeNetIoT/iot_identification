from identification.features.feature_extraction import ExtractionPipeline
import os
from config import TIME_INTERVALS, RAW_DATA_DIRECTORY
import pandas as pd
from src.identification.ml.dataset_preparation import DatasetPreparation
from src.identification.ml.multi_class_model import MultiClassModel


class TestPipeline:
    def __init__(self, verbose=True) -> None:
        self.collection_times = TIME_INTERVALS
        self.prep = DatasetPreparation()
        self.raw_data_directory = RAW_DATA_DIRECTORY
        self.verbose = verbose

    def combine_csvs(self, collection_time):
        extractor = ExtractionPipeline(collection_time)
        all_dfs = []
        for device in os.listdir(self.raw_data_directory):
            if self.verbose:
                print(device)
            for date in os.listdir(f"{self.raw_data_directory}/{device}"):
                for pcap_file in os.listdir(f"{self.raw_data_directory}/{device}/{date}"):
                    pcap_df = extractor.extract_features(input_pcap=f"{self.raw_data_directory}/{device}/{date}/{pcap_file}")
                    if pcap_df.empty:
                        continue
                    prepared_df = self.prep.prepare_df(pcap_df, device)
                    all_dfs.append(prepared_df)
        return pd.concat(all_dfs, ignore_index=True)

    def test_intervals(self):
        for collection_time in self.collection_times:
            input_data = self.combine_csvs(collection_time)
            manager = MultiClassModel()
            manager.add_device(input_data)
            manager.train_all()
            manager.save_all()
            

def main():
    pipeline = TestPipeline()
    pipeline.test_intervals()

if __name__ == "__main__":
    main()