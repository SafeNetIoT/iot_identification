import pandas as pd
from config import settings
import random
from collections import defaultdict, Counter
import json
from src.features.fast_extraction import FastExtractionPipeline
from src.ml.dataset_preparation import DatasetPreparation as prep
from src.utils.data_utils import label_device, clean_up
from copy import deepcopy
from src.services.data_store import DataStoreFactory
from src.services.redis_cache import RedisCache
import joblib
from pathlib import Path


class Cache:
    def __init__(self):
        self.data_store = DataStoreFactory.create(settings.raw_data_directory)
        self.fast_extractor = FastExtractionPipeline()
        self.registry = self.fast_extractor.registry
        self.session_counts = Counter()
        self.unseen_fraction = settings.unseen_fraction
        self.collection_times = settings.time_intervals
        self.redis = RedisCache()
        self.unseen_sessions = defaultdict(list)
        self.local = not settings.is_ci
        self.session_counts_path = settings.session_counts_path
        self.unseen_path = settings.unseen_path

    def save_unseen(self):
        if self.local:
            self.redis.set("unseen_sessions", self.unseen_sessions)
        else:
            print("::notice::saved unseen sessions to pickle")
            joblib.dump(self.unseen_sessions, self.unseen_path)

    def load_unseen(self):
        if self.local:
            unseen_sessions = self.redis.get("unseen_sessions")
        else:
            print("::notice::loaded unseen sessions from pickle")
            unseen_sessions = joblib.load(self.unseen_path)
        return unseen_sessions
    
    def _split_by_collection_time(self, labeled_df, time_arr, time_to_session, session_id):
        """Split labeled data into sessions by time intervals."""
        window_start = 0
        for i, interval_end in enumerate(time_arr):
            for collection_time in self.collection_times:
                if interval_end < collection_time:
                    time_to_session[collection_time].append(
                        (labeled_df.iloc[window_start:i + 1], session_id)
                    )
                    window_start += 1
                    break
    
    def _process_pcap(self, device_pcap: Path, device_name: str, time_to_session: dict, session_id: int) -> int:
        """Extract features, label data, and split into sessions for one pcap file."""
        unlabeled_device_df = self.fast_extractor.extract_features(str(device_pcap))
        if unlabeled_device_df.empty:
            return session_id

        # skip unseen fraction
        if random.random() < self.unseen_fraction:
            self.unseen_sessions[device_name].append(device_pcap)
            return session_id

        labeled_df = label_device(unlabeled_device_df, 0)
        clean_up(labeled_df)
        time_arr = self.registry.get_metadata()
        self._split_by_collection_time(labeled_df, time_arr, time_to_session, session_id)
        return session_id + 1
    
    def _process_device(self, device_dir: Path, device_name: str) -> dict:
        time_to_session = defaultdict(list)
        session_id = 0

        for device_pcap in self.data_store.list_pcap_files(device_dir):
            session_id = self._process_pcap(device_pcap, device_name, time_to_session, session_id)

        self.session_counts[device_name] = session_id
        return time_to_session

    def cache_sessions(self):
        for device_dir in self.data_store.list_dirs():
            device_name = str(device_dir.name)
            time_to_session = self._process_device(device_dir, device_name)
            self.data_store.save_time_to_session(device_name, time_to_session)
        self.save_session_counts()
        self.save_unseen()

    def save_session_counts(self):
        if self.local:
            self.redis.set("session_counts", self.session_counts)
        else:
            print("::notice::saving session counts to a json file")
            with open(self.session_counts_path, 'w') as file:
                json.dump(self.session_counts, file, indent=2)

    def load_session_counts(self):
        if self.local:
            self.session_counts = self.redis.get("session_counts")
        else:
            print("::notice::loading session counts from json file")
            with open(self.session_counts_path, 'r') as file:
                self.session_counts = json.load(file)

    def _initialize_device_sessions(self):
        """Prepare an empty session list for each device based on session counts."""
        self.device_sessions = {
            device_name: [None] * count
            for device_name, count in self.session_counts.items()
        }

    def _process_collection_time(self, collection_time_dir: Path):
        """Iterate over all device directories within one collection time."""
        for device_dir in collection_time_dir.iterdir():
            device_name = device_dir.name
            self._process_device_sessions(device_dir, device_name)

    def _process_device_sessions(self, device_dir: Path, device_name: str):
        """Load and merge session files for one device."""
        for session_file in device_dir.iterdir():
            self._merge_session_file(session_file, device_name)

    def _merge_session_file(self, session_file: Path, device_name: str):
        """Read a session parquet file and merge it into device_sessions."""
        session = pd.read_parquet(session_file)
        session_index = int(session_file.stem.split("_")[1])
        existing = self.device_sessions[device_name][session_index]
        if existing is None:
            self.device_sessions[device_name][session_index] = session
        else:
            self.device_sessions[device_name][session_index] = pd.concat(
                [existing, session],
                ignore_index=True
            )

    def map_sessions(self):
        self.load_session_counts()
        self._initialize_device_sessions()
        for collection_time in self.data_store.list_collection_times():
            self._process_collection_time(collection_time)
        return self.device_sessions

    def build(self):
        if not self.data_store.cache_exists():
            self.cache_sessions()
        self.map_sessions()
        self.unseen_sessions = self.load_unseen()
        return self.device_sessions, self.unseen_sessions

class TimeBasedCache(Cache):
    def __init__(self):
        super().__init__()

    def _initialize_session_map(self):
        """Initialize the top-level session map structure."""
        return defaultdict(lambda: defaultdict(list))

    def _initialize_time_snapshot(self, session_map, index, collection_time):
        """Copy previous snapshot or initialize an empty one."""
        if index > 0:
            prev_time = self.collection_times[index - 1]
            session_map[collection_time] = deepcopy(session_map[prev_time])
        else:
            session_map[collection_time] = defaultdict(list)

    def _process_collection_dir(self, collection_time_dir: Path, collection_time: int, session_map, session_ptr):
        """Process all device directories within a single collection time."""
        for device_dir in collection_time_dir.iterdir():
            device_name = device_dir.name
            self._process_device_sessions(device_dir, device_name, session_map, session_ptr, collection_time)

    def _process_device_sessions(self, device_dir: Path, device_name: str, session_map, session_ptr, collection_time):
        """Process all session files for a single device."""
        for session_file in device_dir.iterdir():
            self._merge_time_session(session_file, device_name, session_map, session_ptr, collection_time)

    def _merge_time_session(self, session_file: Path, device_name: str, session_map, session_ptr, collection_time: int):
        """Merge or append a single session file into the correct collection time."""
        session_id = int(session_file.stem.split("_")[1])
        session_df = pd.read_parquet(str(session_file))
        if session_id not in session_ptr[device_name]:
            session_ptr[device_name][session_id] = len(session_map[collection_time][device_name])
            session_map[collection_time][device_name].append(session_df)
        else:
            index = session_ptr[device_name][session_id]
            existing = session_map[collection_time][device_name][index]
            session_map[collection_time][device_name][index] = pd.concat(
                [existing, session_df], ignore_index=True
            )

    def map_sessions(self):
        """Map sessions over time intervals for all devices."""
        self.device_sessions = self._initialize_session_map()
        session_ptr = defaultdict(dict) # device_name: session_id: index
        collection_dirs = self.data_store.list_collection_times(sorted_times=True)
        for i, collection_time_dir in enumerate(collection_dirs):
            collection_time = int(collection_time_dir.name)
            self._initialize_time_snapshot(self.device_sessions, i, collection_time)
            self._process_collection_dir(collection_time_dir, collection_time, self.device_sessions, session_ptr)

    def build(self):
        return super().build()
    
if __name__ == "__main__":
    time_based_cache = TimeBasedCache()
    time_based_cache.build()
    
