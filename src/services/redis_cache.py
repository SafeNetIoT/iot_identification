from redis import Redis
from config import RedisSettings
import json
import pickle

class RedisCache:
    """Wrapper around redis-py for caching Python objects."""

    def __init__(self):
        self.redis = Redis(
            host=RedisSettings().host,
            port=RedisSettings().port,
            db=RedisSettings().db,
            decode_responses=False,  # store bytes for pickled data
        )

    def set(self, key: str, value, expire: int | None = None):
        """Set a key to a pickled value with optional expiration (in seconds)."""
        data = pickle.dumps(value)
        self.redis.set(key, data, ex=expire)

    def get(self, key: str):
        """Retrieve and unpickle a value by key."""
        data = self.redis.get(key)
        if data is None:
            return None
        return pickle.loads(data)

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self.redis.exists(key) == 1

    def delete(self, key: str):
        """Delete a key."""
        self.redis.delete(key)

    def clear(self, prefix: str | None = None):
        """Delete all keys, or only those matching a prefix."""
        pattern = f"{prefix}*" if prefix else "*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)

    def set_json(self, key: str, value: dict, expire: int | None = None):
        """Set a JSON-serializable dict."""
        data = json.dumps(value)
        self.redis.set(key, data, ex=expire)

    def get_json(self, key: str):
        """Get and decode a JSON object."""
        data = self.redis.get(key)
        if not data:
            return None
        return json.loads(data)
