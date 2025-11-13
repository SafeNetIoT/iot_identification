import zarr
import pandas as pd
from pathlib import Path
from config import settings
import random
from collections import defaultdict, Counter
import json
from src.features.fast_extraction import FastExtractionPipeline
from src.ml.dataset_preparation import DatasetPreparation as prep
from copy import deepcopy
from src.services.data_store import DataStoreFactory
from src.services.redis_cache import RedisCache
import os
from src.utils.file_utils import print_file_tree


class Cache:
    def __init__(self):
        self.data_store = DataStoreFactory.create(settings.raw_data_directory)
        self.fast_extractor = FastExtractionPipeline()
        self.registry = self.fast_extractor.registry # maybe a method
        self.session_counts = Counter()
        self.unseen_fraction = settings.unseen_fraction
        self.collection_times = settings.time_intervals
        self.redis = RedisCache()
        self.unseen_sessions = defaultdict(list)
        self.local = os.getenv("GITHUB_ACTIONS", "").lower() != "true"
        if not self.local:
            self.session_counts_path = settings.session_cache_path / "session_counts.json"

    def save_unseen(self):
        self.redis.set("unseen_sessions", self.unseen_sessions)

    def load_unseen(self):
        return self.redis.get("unseen_sessions")

    def save_session(self, cache, cache_name):
        root = zarr.open(self.cache_path / f"{cache_name}.zarr", mode="w")
        for device, sessions in cache.items():
            group = root.create_group(device)
            for i, item in enumerate(sessions):
                if isinstance(item, pd.DataFrame):
                    group.create_dataset(f"session_{i:05d}", data=item.to_records(index=False))
                elif isinstance(item, (str, Path)):
                    group.create_dataset(f"session_{i:05d}", data=str(item))
                else:
                    raise TypeError(f"Unsupported item type {type(item)} in cache for {device}")

    def load_sessions(self, cache_name):
        root = zarr.open(self.cache_path / f"{cache_name}.zarr", mode="r")
        sessions = {}
        for device in root.group_keys():
            device_group = root[device]
            loaded = []
            for ds in device_group.values():
                if ds.dtype.names:
                    loaded.append(pd.DataFrame(ds[:]))
                else:
                    val = ds[()]  # scalar read (not slicing)
                    if isinstance(val, bytes):
                        val = val.decode()
                    loaded.append(Path(val))
            sessions[device] = loaded
        return sessions

    def cache_sessions(self):
        print("::notice::num devices:", len(list(self.data_store.list_dirs())))
        for device_dir in self.data_store.list_dirs():
            device_name = str(device_dir.name)
            time_to_session = defaultdict(list)
            session_id = 0
            for device_pcap in self.data_store.list_pcap_files(device_dir):
                unlabeled_device_df = self.fast_extractor.extract_features(str(device_pcap))
                if unlabeled_device_df.empty:
                    continue
                time_arr = self.registry.get_metadata()
                if random.random() < self.unseen_fraction:
                    self.unseen_sessions[device_name].append(device_pcap)
                    continue
                labeled_df = prep.label_device(unlabeled_device_df, 0)
                window_start = 0
                for i, interval_end in enumerate(time_arr):
                    for collection_time in self.collection_times:
                        if interval_end < collection_time:
                            time_to_session[collection_time].append(
                                (labeled_df.iloc[window_start:i + 1], session_id)
                                )
                            window_start += 1
                            break
                session_id += 1
                self.session_counts[device_name] = session_id
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

    def map_sessions(self):
        self.load_session_counts()
        print("::notice:: session counts", self.session_counts)
        self.device_sessions = {device_name:[None]*self.session_counts[device_name] for device_name in self.session_counts}
        for collection_time in self.data_store.list_collection_times():
            for device_dir in collection_time.iterdir():
                device_name = device_dir.name
                for session_file in device_dir.iterdir():
                    session = pd.read_parquet(session_file)
                    session_index = int(session_file.stem.split("_")[1])
                    placeholder = self.device_sessions[device_name][session_index]
                    if placeholder is None:
                        self.device_sessions[device_name][session_index] = session
                    else:
                        self.device_sessions[device_name][session_index] = pd.concat([placeholder, session], ignore_index=True)

    def build(self):
        if not self.data_store.cache_exists():
            self.cache_sessions()
        # print_file_tree()
        self.map_sessions()
        self.unseen_sessions = self.load_unseen()
        return self.device_sessions, self.unseen_sessions

class TimeBasedCache(Cache):
    def __init__(self):
        super().__init__()

    def map_sessions(self):
        time_collection_cache = self.cache_path / "collection_times"
        session_map = defaultdict(lambda: defaultdict(list))
        session_ptr = defaultdict(dict) # device_name: session_id: index
        collection_dirs = sorted(time_collection_cache.iterdir(), key=lambda p: int(p.name))
        for i, collection_time_dir in enumerate(collection_dirs):
            collection_time = int(collection_time_dir.name)
            if i > 0:
                prev_time = self.collection_times[i - 1]
                session_map[collection_time] = deepcopy(session_map[prev_time])
            else:
                session_map[collection_time] = defaultdict(list)
            for device_dir in collection_time_dir.iterdir():
                device_name = device_dir.name
                for session_file in device_dir.iterdir():
                    session_id = int(session_file.stem.split("_")[1])
                    session_df = pd.read_parquet(str(session_file))
                    if session_id not in session_ptr:
                        session_ptr[device_name][session_id] = len(session_map[collection_time][device_name])
                        session_map[collection_time][device_name].append(session_df)
                    else:
                        index = session_ptr[device_name][session_id]
                        placeholder = session_map[collection_time_dir][device_name][index]
                        session_map[collection_time][device_name][index] = pd.concat([placeholder, session_df], ignore_index=True)
        return session_map

    def build(self):
        if not self.data_store.cache_exists():
            self.cache_sessions()
        session_map = self.map_sessions()
        # unseen_session = self.load_sessions("unseen_sessions")
        unseen_sessions = self.load_unseen()
        return session_map, unseen_sessions

if __name__ == "__main__":
    cache = Cache()
    cache.cache_sessions()
    print(cache.session_counts)
