class SessionRegistry:
    def __init__(self):
        self.first_pkt = None
        self.time_arr = []
        self.directory = None

    def clear_time_arr(self):
        self.time_arr = []

    def set_first_pkt(self, pkt):
        self.first_pkt = float(getattr(pkt, "time", 0.0))

    def add_collection_time(self, collection_time):
        self.time_arr.append(collection_time - self.first_pkt)

    def get_metadata(self):
        data = self.time_arr
        self.clear_time_arr()
        return data