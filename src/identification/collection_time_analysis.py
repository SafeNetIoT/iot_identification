from identification.features.feature_extraction import ExtractionPipeline
import os
from config import TIME_INTERVALS, RAW_DATA_DIRECTORY
import pandas as pd
from identification.multi_class_model import Model, DatasetPreparation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TestPipeline:
    def __init__(self) -> None:
        self.collection_times = TIME_INTERVALS
        self.prep = DatasetPreparation()

    def split_and_scale(self):
        X = self.prep.output.drop(columns=["label"])
        y = self.prep.output["label"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.prep.test_size, random_state=self.prep.random_state, stratify=y
        )

        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=X.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=X.columns)

        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def test_intervals(self):
        for collection_time in self.collection_times:
            extractor = ExtractionPipeline(collection_time)
            all_dfs = []
            for device in os.listdir(RAW_DATA_DIRECTORY):
                print(f"{RAW_DATA_DIRECTORY}/{device}")
                for date in os.listdir(f"{RAW_DATA_DIRECTORY}/{device}"):
                    for pcap_file in os.listdir(f"{RAW_DATA_DIRECTORY}/{device}/{date}"):
                        pcap_df = extractor.extract_features(input_pcap=f"{RAW_DATA_DIRECTORY}/{device}/{date}/{pcap_file}")
                        if pcap_df.empty:
                            continue
                        keep_cols = [col for col in pcap_df.columns if col in self.prep.feature_set]
                        pcap_df = pcap_df[keep_cols].copy()
                        pcap_df.loc[:, 'label'] = device 
                        all_dfs.append(pcap_df)
                        if pcap_df.empty:
                            print("EMPTY")
            self.prep.output = pd.concat(all_dfs, ignore_index=True)
            x_train, y_train, x_test, y_test = self.split_and_scale()
            clf = Model(x_train, y_train, x_test, y_test)
            clf.train()
            clf.evaluate()
            clf.save()

def main():
    pipeline = TestPipeline()
    pipeline.test_intervals()

if __name__ == "__main__":
    main()