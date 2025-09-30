from src.identification.feature_extraction import FlowState, PcapReader, features_from_state, SWEEP_EVERY_PKTS, BATCH_ROWS, MAX_AGE_S, five_tuple, canonize
import os
from config import TIME_INTERVALS, RAW_DATA_DIRECTORY
import pandas as pd
from src.identification.model import Model, DatasetPreparation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TestPipeline:
    def __init__(self) -> None:
        self.collection_times = TIME_INTERVALS
        self.prep = DatasetPreparation()

    def extract(self, input_pcap: str, time_interval: float = None) -> pd.DataFrame:
        flows: dict[tuple, FlowState] = {}
        buffer = []
        pkt_count = 0

        start_time = None
        cutoff = None

        with PcapReader(input_pcap) as pcap:
            for pkt in pcap:
                pkt_count += 1
                ts = float(getattr(pkt, "time", 0.0))

                # establish cutoff time based on first packet
                if start_time is None:
                    start_time = ts
                    if time_interval is not None:
                        cutoff = start_time + time_interval

                # stop if beyond cutoff 
                if cutoff is not None and ts > cutoff:
                    break

                k = five_tuple(pkt)
                if not k:
                    continue
                ck = canonize(k)

                st = flows.get(ck)
                if st is None:
                    st = FlowState(ck, ts, first_src_ip=k[0])
                    flows[ck] = st

                st.update(pkt)

                # Evict on FIN/RST (TCP) or idle/max-age
                if st.should_evict(ts):
                    buffer.append(features_from_state(st))
                    del flows[ck]

                # Periodic sweep
                if (pkt_count % SWEEP_EVERY_PKTS) == 0:
                    stale = []
                    for key, fs in flows.items():
                        if (ts - fs.last) >= fs.idle_timeout() or (fs.last - fs.first) >= MAX_AGE_S:
                            stale.append(key)
                    for key in stale:
                        buffer.append(features_from_state(flows[key]))
                        del flows[key]

            # EOF: finalize remaining flows
            for key, fs in list(flows.items()):
                buffer.append(features_from_state(fs))
                del flows[key]

        # Convert all rows to a single DataFrame
        df = pd.DataFrame(buffer)
        return df

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
            all_dfs = []
            for device in os.listdir(RAW_DATA_DIRECTORY):
                print(f"{RAW_DATA_DIRECTORY}/{device}")
                for date in os.listdir(f"{RAW_DATA_DIRECTORY}/{device}"):
                    for pcap_file in os.listdir(f"{RAW_DATA_DIRECTORY}/{device}/{date}"):
                        pcap_df = self.extract(f"{RAW_DATA_DIRECTORY}/{device}/{date}/{pcap_file}", collection_time)
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
    # pipeline = TestPipeline()
    # print(pipeline.extract("data/raw/alexa_swan_kettle/2023-10-19/2023-10-19_00:02:55.402s.pcap", 30))
    main()