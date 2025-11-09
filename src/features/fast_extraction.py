from dataclasses import dataclass, asdict
from src.features.feature_extraction import FlowManager
from typing import Optional
from scapy.all import PcapReader
import pandas as pd
from config import settings
import os
from pathlib import Path
from src.ml.dataset_preparation import DatasetPreparation as Prep
from src.features.session_registry import SessionRegistry
from src.features.pcap_reader_factory import PcapReaderFactory

@dataclass
class ValidFeatures:
    sport:int
    dur:float
    pkts_fwd:int
    pkts_bwd:int
    bytes_fwd:int
    bytes_bwd:int
    pktlen_fwd_mean:float
    pktlen_fwd_std:float
    pktlen_bwd_mean:float
    pktlen_bwd_std:float
    iat_fwd_mean:float
    iat_fwd_std:float
    pps:float
    payload_entropy_fwd:float
    iat_cv_fwd:float
    small_pkt_ratio_fwd:float
    small_pkt_ratio_bwd:float
    ttl_fwd_first:int

class FastFeatureExtractor:
    def __init__(self, state) -> None:
        self.state = state

    def features_from_state(self) -> ValidFeatures:
        s = self.state

        dur = max(s.last - s.first, 0.0)
        pkts_fwd, pkts_bwd = s.pkts_fwd, s.pkts_bwd
        bytes_fwd, bytes_bwd = s.bytes_fwd, s.bytes_bwd

        # only compute the forward/backward packet length and IAT stats
        _, pf_mean, _, pf_std = s.len_fwd.stats()
        _, pb_mean, _, pb_std = s.len_bwd.stats()

        _, if_mean, _, if_std = s.iat_fwd.stats()

        small_pkt_ratio_fwd = (s.small_fwd / max(pkts_fwd, 1)) if pkts_fwd else 0.0
        small_pkt_ratio_bwd = (s.small_bwd / max(pkts_bwd, 1)) if pkts_bwd else 0.0
        iat_cv_fwd = (if_std / if_mean) if if_mean > 0 else 0.0

        pps = (pkts_fwd + pkts_bwd) / max(dur, 1e-3)
        payload_entropy_fwd = s.ent_fwd.entropy()

        valid_features = ValidFeatures(
            sport=s.a[1] if s.forward_is_a else s.b[1],
            dur=dur,
            pkts_fwd=pkts_fwd,
            pkts_bwd=pkts_bwd,
            bytes_fwd=bytes_fwd,
            bytes_bwd=bytes_bwd,
            pktlen_fwd_mean=pf_mean,
            pktlen_fwd_std=pf_std,
            pktlen_bwd_mean=pb_mean,
            pktlen_bwd_std=pb_std,
            iat_fwd_mean=if_mean,
            iat_fwd_std=if_std,
            pps=pps,
            payload_entropy_fwd=payload_entropy_fwd,
            iat_cv_fwd=iat_cv_fwd,
            small_pkt_ratio_fwd=small_pkt_ratio_fwd,
            small_pkt_ratio_bwd=small_pkt_ratio_bwd,
            ttl_fwd_first=s.ttl_fwd_first,
        )
        return asdict(valid_features)

class FastExtractionPipeline:
    def __init__(self, time_interval: Optional[float] = None) -> None:
        self.time_interval = time_interval
        self.registry = SessionRegistry()

    def extract_features(self, input_data):
        manager = FlowManager(extractor = FastFeatureExtractor, registry=self.registry)
        rows = []

        with PcapReaderFactory(input_data) as pcap:
            try:
                first_pkt = next(pcap)
                self.registry.set_first_pkt(first_pkt)
                first_ts = float(getattr(first_pkt, "time", 0.0))
                row = manager.update_flow(first_pkt, first_ts)
                if row:
                    rows.append(row)
            except StopIteration:
                return pd.DataFrame()
            
            for pkt_count, pkt in enumerate(pcap, 1):
                ts = float(getattr(pkt, "time", 0.0))
                row = manager.update_flow(pkt, ts)
                if row:
                    rows.append(row)
                if pkt_count % settings.sweep_every_pkts == 0:
                    rows.extend(manager.sweep(ts))
            rows.extend([manager.evict(k) for k in list(manager.flows.keys())])

        return pd.DataFrame(rows)

def main():
    os.makedirs(settings.fast_extraction_directory, exist_ok=True)
    extractor = FastExtractionPipeline()
    for device in os.listdir(settings.raw_data_directory):
        device_dfs = []
        data_dir = Path(f"{settings.raw_data_directory}/{device}")
        for pcap_path in data_dir.rglob("*.pcap"): 
            device_df = extractor.extract_features(str(pcap_path))
            device_dfs.append(device_df)
        complete_device = pd.concat(device_dfs, ignore_index=True)
        complete_device = Prep.label_device(complete_device, device)
        complete_device.to_csv(f"{settings.fast_extraction_directory}/{device}.csv", index=False)

if __name__ == "__main__":
    main()