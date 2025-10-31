from collections import defaultdict
import joblib


class SessionRegistry:
    def __init__(self):
        self.first_pkt = None
        self._map = defaultdict(list)
        self.time_arr = []
        self.directory = None

    def set_directory(self, directory):
        self.directory = directory

    def clear_time_arr(self):
        self.time_arr = []

    def set_first_pkt(self, pkt):
        self.first_pkt = float(getattr(pkt, "time", 0.0))

    def add_collection_time(self, collection_time):
        self.time_arr.append(collection_time - self.first_pkt)

    def add_timing_metadata(self, key: str):
        self._map[key].extend(self.time_arr)
        self.clear_time_arr()

    def get_metadata(self):
        data = self.time_arr
        self.clear_time_arr()
        return data

    def get(self, key):
        return self._map.get(key, [])

    def save_registry(self, file_name):
        joblib.dump(self._map, f"{self.directory}/{file_name}")

    def load_registry(self, file_name):
        self.session_times = joblib.load(f"{self.directory}/{file_name}")