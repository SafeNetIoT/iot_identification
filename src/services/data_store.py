from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from ftplib import FTP
from io import BytesIO
from config import settings, PROJECT_ROOT

class DataStore(ABC):
    """Abstract interface for data access layer."""

    @abstractmethod
    def list_dirs(self):
        pass

    @abstractmethod
    def list_pcap_files(self, directory_name):
        pass

    @abstractmethod
    def save_time_to_session(self, device_name, time_to_session):
        pass

    @abstractmethod
    def list_collection_times(self):
        pass

    @abstractmethod
    def cache_exists(self):
        pass

    # @abstractmethod
    # def save_dataframe(self, df, path):
    #     pass

    # @abstractmethod
    # def load_dataframe(self, path):
    #     pass

class LocalStore(DataStore):
    def __init__(self, data_path):
        self.base_path = Path(PROJECT_ROOT) / data_path
        self.cache_path = Path(PROJECT_ROOT) / settings.session_cache_path

    def list_dirs(self):
        return [d for d in self.base_path.iterdir() if d.is_dir()]

    def list_pcap_files(self, directory_name):
        return list(directory_name.rglob("*.pcap"))
    
    def save_time_to_session(self, device_name, time_to_session):
        for collection_time in time_to_session:
            collection_dir = self.cache_path / "collection_times" / str(collection_time)
            collection_dir.mkdir(parents=True, exist_ok=True)
            for session, session_id in time_to_session[collection_time]:
                session_file = collection_dir / device_name / f"session_{session_id}.parquet"
                session_file.parent.mkdir(parents=True, exist_ok=True)
                session.to_parquet(session_file, index=False)

    def list_collection_times(self):
        collection_dirs = self.cache_path / "collection_times"
        return collection_dirs.iterdir()
    
    def cache_exists(self):
        collection_path = self.cache_path / "collection_times"
        if not collection_path.exists():
            return False
        try:
            next(collection_path.iterdir())
            return True
        except StopIteration:
            return False

    # def save_dataframe(self, df, rel_path):
    #     full_path = self.base_path / rel_path
    #     full_path.parent.mkdir(parents=True, exist_ok=True)
    #     df.to_parquet(full_path, index=False)
    #     return str(full_path)

    # def load_dataframe(self, rel_path):
    #     return pd.read_parquet(self.base_path / rel_path)

class FTPStore(DataStore):
    def __init__(self, host, user, password, base_dir="/"):
        self.host = host
        self.user = user
        self.password = password
        self.base_dir = Path(base_dir)
        self.ftp = self._connect()

    def _connect(self):
        ftp = FTP(self.host)
        ftp.login(self.user, self.password)
        ftp.cwd(self.base_dir)
        return ftp

    def list_dirs(self): # might need to be changed
        dirs = []
        self.ftp.retrlines('LIST', dirs.append)
        return dirs

    def list_pcap_files(self, directory_name):
        self.ftp.cwd(f"{self.base_dir}/{directory_name}")
        filenames = self.ftp.nlst()
        pcap_files = [f for f in filenames if f.endswith(".pcap")]
        return [Path(directory_name) / Path(name) for name in pcap_files]
    
    def save_time_to_session(self, device_name, time_to_session):
        pass

    def list_collection_times(self):
        pass

    def cache_exists(self):
        pass

    # def save_dataframe(self, df, rel_path):
    #     buffer = BytesIO()
    #     df.to_parquet(buffer, index=False)
    #     buffer.seek(0)
    #     self.ftp.storbinary(f"STOR {rel_path}", buffer)
    #     return f"ftp://{self.host}/{rel_path}"

    # def load_dataframe(self, rel_path):
    #     buffer = BytesIO()
    #     self.ftp.retrbinary(f"RETR {rel_path}", buffer.write)
    #     buffer.seek(0)
    #     return pd.read_parquet(buffer)

class DataStoreFactory:

    @staticmethod
    def create(data_path):
        store_type = settings.default_store.lower()
        if store_type == "local":
            return LocalStore(data_path)
        elif store_type == "ftp":
            return FTPStore(
                host=settings.ftp_settings.FTP_HOST,
                user=settings.ftp_settings.FTP_USER,
                password=settings.ftp_settings.FTP_PASS,
                base_dir=settings.ftp_settings.FTP_BASE_DIR,
            )
        else:
            raise ValueError(f"Unsupported data store type: {store_type}")
