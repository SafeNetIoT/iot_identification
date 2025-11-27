import os
from pathlib import Path
import pandas as pd
from typing import Optional
from config import settings

from scapy.all import PcapReader
from src.utils.network_utils import is_internal, port_bucket, iot_service_id, discovery_proto_id, five_tuple, canonize
from src.features.extraction_manager import FlowManager

class ExtractionPipeline:
    def __init__(self, time_interval: Optional[float] = None) -> None:
        self.time_interval = time_interval

    def extract_features(self, input_pcap: str) -> pd.DataFrame:
        manager = FlowManager()
        rows = []
        start_time = None
        cutoff = None

        with PcapReader(input_pcap) as pcap:
            for pkt_count, pkt in enumerate(pcap, 1):
                ts = float(getattr(pkt, "time", 0.0))

                if start_time is None:
                    start_time = ts
                    if self.time_interval is not None:
                        cutoff = start_time + self.time_interval

                if cutoff is not None and ts > cutoff:
                    break

                row = manager.update_flow(pkt, ts)
                if row:
                    rows.append(row)
                if pkt_count % settings.sweep_every_pkts == 0:
                    rows.extend(manager.sweep(ts))
            rows.extend([manager.evict(k) for k in list(manager.flows.keys())])

        return pd.DataFrame(rows)

def main(): 
    os.makedirs(settings.preprocessed_data_directory, exist_ok=True)
    extractor = ExtractionPipeline()
    for device in os.listdir(settings.raw_data_directory):
        device_dfs = []
        data_dir = Path(f"{settings.raw_data_directory}/{device}")
        for pcap_path in data_dir.rglob("*.pcap"): 
            device_df = extractor.extract_features(str(pcap_path))
            device_dfs.append(device_df)
        complete_device = pd.concat(device_dfs, ignore_index=True)
        complete_device.to_csv(f"{settings.preprocessed_data_directory}/{device}.csv", index=False)

if __name__ == "__main__":
    print('running...')
    main()
