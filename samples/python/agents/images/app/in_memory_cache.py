"""In-memory cache implementation for storing image data across sessions."""

from typing import Any, Dict, Optional


class InMemoryCache:
    """Simple singleton in-memory cache for storing session data."""
    
    _instance: Optional['InMemoryCache'] = None
    _cache: Dict[str, Any] = {}
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(InMemoryCache, cls).__new__(cls)
            cls._cache = {}
        return cls._instance
    
    def get(self, key: str) -> Any:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        self._cache[key] = value
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed and was deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self._cache.keys())
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
