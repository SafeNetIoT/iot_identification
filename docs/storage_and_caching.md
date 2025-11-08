# Storage and Caching
This doc explains how storage and caching works for in this project. Storage can be local or facilitated using an FTP server.

As the project grows the memory requirements increase. Factors that mainly influence the memory requirements is the raw data and the intermediate data.

## Storage Config
The storage has two purposes - storing the raw data and the intermediate data.

Specify the storage type ("local" or "ftp") in config.py 

For local storage:
```python
class Settings(BaseSettings):
    default_store: str = "local"
```

For FTP based storage:
```python
class Settings(BaseSettings):
    default_store: str = "local"
```

FTP based storage additionally requires the specification of arguments in FTPSettings and the the specification of the password in .env.

## Caching 
The purpose of the cache is to store the sample of the raw data that will used for tests and to store metadata about the datatset.
Currently the metadata is simply a key-value that maps each device to the number of sessions of such device in the dataset.
Currently Redis is used for caching due to its low latency.

### Config
The config.py file already has specified values in RedisSettings, which can be changed if necessary.

