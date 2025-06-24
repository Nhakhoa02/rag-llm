"""
Distributed Storage Manager

Replaces Qdrant client with our own distributed vector storage system.
Provides the same interface as the original storage manager but uses
multiple distributed nodes for fault tolerance and scalability.
"""

import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from .distributed_vector_store import DistributedVectorStore, VectorNode, ConsistencyLevel
from ..utils.logging import get_logger
from ..models.base import BaseDocument
from ..models.document import Document

logger = get_logger(__name__)

class DistributedStorageManager:
    """
    Distributed storage manager that replaces Qdrant with our own distributed system.
    """
    
    def __init__(self, 
                 nodes: Optional[List[VectorNode]] = None,
                 replication_factor: int = 2,
                 consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM,
                 shard_count: int = 8,
                 vector_size: int = 384):
        
        self.vector_size = vector_size
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize distributed vector store
        if nodes is None:
            # Default nodes for development
            nodes = [
                VectorNode("node1", "localhost", 8001),
                VectorNode("node2", "localhost", 8002),
                VectorNode("node3", "localhost", 8003)
            ]
        
        self.distributed_store = DistributedVectorStore(
            nodes=nodes,
            replication_factor=replication_factor,
            consistency_level=consistency_level,
            shard_count=shard_count
        )
        
        logger.info(f"Initialized distributed storage with {len(nodes)} nodes")
    
    async def create_index(self, index_name: str, vector_size: Optional[int] = None) -> bool:
        """Create a new index (collection) in the distributed system."""
        try:
            if vector_size is None:
                vector_size = self.vector_size
            
            success = await self.distributed_store.create_collection(index_name, vector_size)
            if success:
                logger.info(f"Created distributed index: {index_name}")
            else:
                logger.error(f"Failed to create distributed index: {index_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating distributed index {index_name}: {e}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete an index from the distributed system."""
        try:
            success = await self.distributed_store.delete_collection(index_name)
            if success:
                logger.info(f"Deleted distributed index: {index_name}")
            else:
                logger.error(f"Failed to delete distributed index: {index_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting distributed index {index_name}: {e}")
            return False
    
    async def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes in the distributed system."""
        try:
            collections = await self.distributed_store.list_collections()
            
            # Convert to expected format
            indexes = []
            for collection in collections:
                indexes.append({
                    "name": collection["name"],
                    "vectors_count": collection["vectors_count"],
                    "shard_count": collection["shard_count"]
                })
            
            return indexes
            
        except Exception as e:
            logger.error(f"Error listing distributed indexes: {e}")
            return []
    
    async def upsert_documents(self, index_name: str, documents: List[Document]) -> bool:
        """Upsert documents to the distributed system."""
        try:
            if not documents:
                return True
            
            # Generate embeddings
            texts = [doc.content for doc in documents if doc.content]
            if not texts:
                logger.warning("No valid content found in documents")
                return False
                
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists - embeddings is already numpy array
            vectors = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
            # Convert documents to BaseDocument format
            base_documents = []
            for doc in documents:
                base_doc = BaseDocument(
                    id=doc.id,
                    content=doc.content or "",
                    metadata=doc.metadata,
                    type=doc.type
                )
                base_documents.append(base_doc)
            
            # Upsert to distributed system
            success = await self.distributed_store.upsert_vectors(index_name, vectors, base_documents)
            
            if success:
                logger.info(f"Upserted {len(documents)} documents to distributed index: {index_name}")
            else:
                logger.error(f"Failed to upsert documents to distributed index: {index_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error upserting documents to distributed index {index_name}: {e}")
            return False
    
    async def search_documents(self, 
                             index_name: str, 
                             query: str, 
                             limit: int = 10, 
                             score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search documents in the distributed system."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_vector = query_embedding[0].tolist() if hasattr(query_embedding[0], 'tolist') else query_embedding[0]
            
            # Search in distributed system
            results = await self.distributed_store.search_vectors(
                index_name, 
                query_vector, 
                limit, 
                score_threshold
            )
            
            logger.info(f"Found {len(results)} results in distributed index: {index_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching distributed index {index_name}: {e}")
            return []
    
    async def get_document_by_id(self, index_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID from the distributed system."""
        try:
            # This would require implementing a direct lookup in the distributed system
            # For now, we'll search with a very specific query
            results = await self.search_documents(index_name, document_id, limit=1)
            if results:
                return results[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {document_id} from distributed index {index_name}: {e}")
            return None
    
    async def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific index."""
        try:
            collections = await self.distributed_store.list_collections()
            
            for collection in collections:
                if collection["name"] == index_name:
                    return {
                        "name": collection["name"],
                        "vectors_count": collection["vectors_count"],
                        "shard_count": collection["shard_count"]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting stats for distributed index {index_name}: {e}")
            return None
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of the distributed cluster."""
        try:
            return self.distributed_store.get_cluster_status()
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return {
                "error": str(e),
                "total_nodes": 0,
                "healthy_nodes": 0,
                "total_shards": 0,
                "total_collections": 0,
                "total_vectors": 0
            }
    
    async def health_check(self) -> bool:
        """Check if the distributed system is healthy."""
        try:
            status = self.distributed_store.get_cluster_status()
            healthy_nodes = status.get("healthy_nodes", 0)
            total_nodes = status.get("total_nodes", 0)
            
            # System is healthy if at least 50% of nodes are healthy
            return healthy_nodes > 0 and (healthy_nodes / total_nodes) >= 0.5
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def add_node(self, node: VectorNode) -> bool:
        """Add a new node to the distributed system."""
        try:
            self.distributed_store.add_node(node)
            logger.info(f"Added node {node.id} to distributed system")
            return True
        except Exception as e:
            logger.error(f"Error adding node {node.id}: {e}")
            return False
    
    async def remove_node(self, node_id: str) -> bool:
        """Remove a node from the distributed system."""
        try:
            self.distributed_store.remove_node(node_id)
            logger.info(f"Removed node {node_id} from distributed system")
            return True
        except Exception as e:
            logger.error(f"Error removing node {node_id}: {e}")
            return False

# Factory function for easy initialization
def create_distributed_storage_manager(
    nodes: Optional[List[VectorNode]] = None,
    replication_factor: int = 2,
    consistency_level: str = "quorum",
    shard_count: int = 8,
    vector_size: int = 384
) -> DistributedStorageManager:
    """Create a distributed storage manager with the specified configuration."""
    
    # Convert consistency level string to enum
    if consistency_level == "one":
        consistency_enum = ConsistencyLevel.ONE
    elif consistency_level == "all":
        consistency_enum = ConsistencyLevel.ALL
    else:
        consistency_enum = ConsistencyLevel.QUORUM
    
    return DistributedStorageManager(
        nodes=nodes,
        replication_factor=replication_factor,
        consistency_level=consistency_enum,
        shard_count=shard_count,
        vector_size=vector_size
    ) 