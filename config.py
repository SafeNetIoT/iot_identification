import ipaddress
from typing import List, Optional, Dict
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class ModelArchitecture(BaseModel):
    n_estimators: int
    max_depth: Optional[int]
    random_state: int
    n_jobs: int

class TestingConfig(BaseModel):
    fast_mode: int
    sample_fraction: float

# class FTPSettings(BaseSettings):
#     FTP_HOST: str
#     FTP_USER: str
#     FTP_PASS: str
#     FTP_BASE_DIR: str

#     class Config:
#         env_file = ".env"

class RedisSettings(BaseSettings):
    host:str = "localhost"
    port:int = 6379
    db: int = 0

class Settings(BaseSettings):
    preprocessed_data_directory: str = "data/preprocessed"
    fast_extraction_directory: str = "data/fast_extraction"
    raw_data_directory: str = "data/raw"
    valid_features_directory: str = "src/features/features.txt"
    feature_menu_path: str = "src/features/feature_menu.yml"
    models_directory: str = "models"
    session_cache_path: str = "session_cache"
    unsw_dataset_path: str = "data/unsw_unzipped"
    time_intervals: List[int] = [10, 20, 30, 45, 60, 75, 90, 105]
    internal_nets: List[ipaddress.IPv4Network] = [ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"), ipaddress.ip_network("192.168.0.0/16")]
    tcp_idle_s: int = 300
    udp_idle_s: int = 60
    max_age_s: int = 3600
    k_payload_bytes: int = 512
    batch_rows: int = 5000
    sweep_every_pkts: int = 2000
    model_architectures: Dict[str, ModelArchitecture] = {
        "standard_forest": ModelArchitecture(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=2,
        )
    }
    random_state: int = 42
    unseen_fraction: float = 0.1
    testing: TestingConfig = TestingConfig(fast_mode=1, sample_fraction=0.1)
    model_under_test: str = "models/2025-10-25/binary_model1"
    multiclass_model_under_test: str = "models/2025-10-25/multiclass_model5"
    desired_accuracy: float = 0.85
    mac_address_map_path: str = "mac_address_map.json"
    redis_settings:RedisSettings = RedisSettings()
    # ftp_settings:FTPSettings = FTPSettings()
    default_store: str = "local"

settings = Settings()
