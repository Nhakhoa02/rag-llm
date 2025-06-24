"""
Storage manager for orchestrating distributed storage operations.
"""

import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..models.base import BaseDocument
from ..models.document import Document
from ..utils.logging import get_logger
from .distributed_storage_manager import DistributedStorageManager, create_distributed_storage_manager
from .distributed_vector_store import VectorNode
from .cassandra_client import CassandraClient
from .elasticsearch_client import ElasticsearchClient
from .minio_client import MinioClient


class StorageConfig(BaseModel):
    """Storage manager configuration."""
    
    enable_distributed_vector: bool = Field(default=True, description="Enable distributed vector storage")
    enable_cassandra: bool = Field(default=False, description="Enable Cassandra storage")
    enable_elasticsearch: bool = Field(default=False, description="Enable Elasticsearch storage")
    enable_minio: bool = Field(default=False, description="Enable MinIO object storage")
    replication_factor: int = Field(default=2, description="Replication factor for distributed storage")
    consistency_level: str = Field(default="quorum", description="Consistency level for distributed storage")
    shard_count: int = Field(default=8, description="Number of shards for distributed storage")
    vector_size: int = Field(default=384, description="Vector size for embeddings")
    
    class Config:
        extra = "allow"


class StorageManager:
    """Manages distributed storage operations across multiple storage systems."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.logger = get_logger(__name__)
        
        # Initialize distributed vector storage
        if self.config.enable_distributed_vector:
            # Default nodes for development
            nodes = [
                VectorNode("node1", "localhost", 8001),
                VectorNode("node2", "localhost", 8002),
                VectorNode("node3", "localhost", 8003)
            ]
            
            self.distributed_storage = create_distributed_storage_manager(
                nodes=nodes,
                replication_factor=self.config.replication_factor,
                consistency_level=self.config.consistency_level,
                shard_count=self.config.shard_count,
                vector_size=self.config.vector_size
            )
        else:
            self.distributed_storage = None
        
        # Initialize other storage clients
        self.cassandra_client = CassandraClient() if self.config.enable_cassandra else None
        self.elasticsearch_client = ElasticsearchClient() if self.config.enable_elasticsearch else None
        self.minio_client = MinioClient() if self.config.enable_minio else None
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all storage systems."""
        try:
            self.logger.info("Initializing storage manager...")
            
            # Initialize distributed vector storage
            if self.distributed_storage:
                # Check if distributed system is healthy
                is_healthy = await self.distributed_storage.health_check()
                if not is_healthy:
                    self.logger.warning("Distributed vector storage is not healthy")
            
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
    
    # Distributed vector storage methods
    async def create_index(self, index_name: str, vector_size: Optional[int] = None) -> bool:
        """Create a new index in the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return False
        
        return await self.distributed_storage.create_index(index_name, vector_size)
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete an index from the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return False
        
        return await self.distributed_storage.delete_index(index_name)
    
    async def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes in the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return []
        
        return await self.distributed_storage.list_indexes()
    
    async def upsert_documents(self, index_name: str, documents: List[Document]) -> bool:
        """Upsert documents to the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return False
        
        return await self.distributed_storage.upsert_documents(index_name, documents)
    
    async def search_documents(self, 
                             index_name: str, 
                             query: str, 
                             limit: int = 10, 
                             score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search documents in the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return []
        
        return await self.distributed_storage.search_documents(index_name, query, limit, score_threshold)
    
    async def get_document_by_id(self, index_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID from the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return None
        
        return await self.distributed_storage.get_document_by_id(index_name, document_id)
    
    async def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific index."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return None
        
        return await self.distributed_storage.get_index_stats(index_name)
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the distributed cluster."""
        if not self.distributed_storage:
            return {
                "error": "Distributed vector storage is not enabled",
                "total_nodes": 0,
                "healthy_nodes": 0,
                "total_shards": 0,
                "total_collections": 0,
                "total_vectors": 0
            }
        
        return await self.distributed_storage.get_cluster_status()
    
    async def add_node(self, node: VectorNode) -> bool:
        """Add a new node to the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return False
        
        return await self.distributed_storage.add_node(node)
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the distributed system."""
        if not self.distributed_storage:
            self.logger.error("Distributed vector storage is not enabled")
            return False
        
        return await self.distributed_storage.remove_node(node_id)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics for all storage systems."""
        stats = {
            "initialized": self._initialized,
            "storage_systems": {}
        }
        
        if self.distributed_storage:
            stats["storage_systems"]["distributed_vector"] = {
                "enabled": True,
                "connected": True  # Always connected
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