"""
Storage manager for orchestrating distributed storage operations.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..models.base import BaseDocument
from ..utils.logging import get_logger
from .qdrant_client import QdrantClient
from .cassandra_client import CassandraClient
from .elasticsearch_client import ElasticsearchClient
from .minio_client import MinioClient


class StorageConfig(BaseModel):
    """Storage manager configuration."""
    
    enable_qdrant: bool = Field(default=True, description="Enable Qdrant vector storage")
    enable_cassandra: bool = Field(default=False, description="Enable Cassandra storage")
    enable_elasticsearch: bool = Field(default=False, description="Enable Elasticsearch storage")
    enable_minio: bool = Field(default=False, description="Enable MinIO object storage")
    replication_factor: int = Field(default=3, description="Replication factor")
    
    class Config:
        extra = "allow"


class StorageManager:
    """Manages distributed storage operations across multiple storage systems."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.logger = get_logger(__name__)
        
        # Initialize storage clients
        self.qdrant_client = QdrantClient() if self.config.enable_qdrant else None
        self.cassandra_client = CassandraClient() if self.config.enable_cassandra else None
        self.elasticsearch_client = ElasticsearchClient() if self.config.enable_elasticsearch else None
        self.minio_client = MinioClient() if self.config.enable_minio else None
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all storage systems."""
        try:
            self.logger.info("Initializing storage manager...")
            
            # Initialize Cassandra
            if self.cassandra_client:
                await self.cassandra_client.connect()
            
            # Initialize Elasticsearch
            if self.elasticsearch_client:
                await self.elasticsearch_client.connect()
            
            # Initialize MinIO
            if self.minio_client:
                await self.minio_client.connect()
            
            self._initialized = True
            self.logger.info("Storage manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage manager: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all storage connections."""
        try:
            self.logger.info("Shutting down storage manager...")
            
            # Disconnect from storage systems
            if self.cassandra_client:
                await self.cassandra_client.disconnect()
            
            if self.elasticsearch_client:
                await self.elasticsearch_client.disconnect()
            
            if self.minio_client:
                await self.minio_client.disconnect()
            
            self._initialized = False
            self.logger.info("Storage manager shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error shutting down storage manager: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics for all storage systems."""
        stats = {
            "initialized": self._initialized,
            "storage_systems": {}
        }
        
        if self.qdrant_client:
            stats["storage_systems"]["qdrant"] = {
                "enabled": True,
                "connected": True  # Qdrant client is always connected
            }
        
        if self.cassandra_client:
            stats["storage_systems"]["cassandra"] = {
                "enabled": True,
                "connected": self.cassandra_client.is_connected()
            }
        
        if self.elasticsearch_client:
            stats["storage_systems"]["elasticsearch"] = {
                "enabled": True,
                "connected": self.elasticsearch_client.is_connected()
            }
        
        if self.minio_client:
            stats["storage_systems"]["minio"] = {
                "enabled": True,
                "connected": self.minio_client.is_connected()
            }
        
        return stats 