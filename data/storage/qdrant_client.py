"""
Qdrant vector database client for distributed vector storage and search.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from qdrant_client import QdrantClient as QdrantBaseClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid

from core.models.base import BaseDocument, DataType
from core.utils.logging import LoggerMixin
from core.utils.metrics import monitor_function
from config.config import settings


class QdrantClient(LoggerMixin):
    """Distributed Qdrant client for vector storage and search."""
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant host
            port: Qdrant port
        """
        super().__init__()
        
        self.host = host or settings.database.qdrant_host
        self.port = port or settings.database.qdrant_port
        
        # Initialize client
        self.client = QdrantBaseClient(host=self.host, port=self.port)
        
        # Default collection settings
        self.default_vector_size = settings.processing.vector_dimension
        self.default_distance = Distance.COSINE
        
        self.logger.info("Qdrant client initialized", host=self.host, port=self.port)
    
    @monitor_function("qdrant_client", "create_collection", "collection")
    async def create_collection(self, collection_name: str, 
                               vector_size: int = None,
                               distance: Distance = None,
                               shard_number: int = 1,
                               replication_factor: int = 1) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of vector dimensions
            distance: Distance metric for similarity
            shard_number: Number of shards for distribution
            replication_factor: Number of replicas for fault tolerance
            
        Returns:
            True if collection created successfully
        """
        try:
            vector_size = vector_size or self.default_vector_size
            distance = distance or self.default_distance
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                    on_disk=True  # Store vectors on disk for large collections
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=10000  # Use memory mapping for large collections
                ),
                replication_factor=replication_factor,
                shard_number=shard_number
            )
            
            self.logger.info("Collection created successfully", 
                           collection_name=collection_name,
                           vector_size=vector_size,
                           shards=shard_number,
                           replicas=replication_factor)
            
            return True
            
        except Exception as e:
            self.logger.error("Collection creation failed", 
                            error=str(e), 
                            collection_name=collection_name)
            raise
    
    @monitor_function("qdrant_client", "delete_collection", "collection")
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            self.logger.info("Collection deleted", collection_name=collection_name)
            return True
        except Exception as e:
            self.logger.error("Collection deletion failed", 
                            error=str(e), 
                            collection_name=collection_name)
            raise
    
    @monitor_function("qdrant_client", "upsert_vectors", "vectors")
    async def upsert_vectors(self, collection_name: str, 
                            vectors: List[List[float]],
                            documents: List[BaseDocument],
                            batch_size: int = 100) -> bool:
        """
        Upsert vectors and metadata into collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vector embeddings
            documents: List of documents with metadata
            batch_size: Batch size for insertion
            
        Returns:
            True if vectors inserted successfully
        """
        try:
            if len(vectors) != len(documents):
                raise ValueError("Number of vectors must match number of documents")
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                
                # Create points
                points = []
                for j, (vector, doc) in enumerate(zip(batch_vectors, batch_documents)):
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "document_id": doc.id,
                            "content": doc.content,
                            "type": doc.type.value,
                            "metadata": doc.metadata,
                            "created_at": doc.created_at.isoformat(),
                            "updated_at": doc.updated_at.isoformat()
                        }
                    )
                    points.append(point)
                
                # Upsert batch
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                self.logger.debug("Batch upserted", 
                                collection_name=collection_name,
                                batch_size=len(points),
                                batch_index=i // batch_size)
            
            self.logger.info("Vectors upserted successfully", 
                           collection_name=collection_name,
                           total_vectors=len(vectors))
            
            return True
            
        except Exception as e:
            self.logger.error("Vector upsert failed", 
                            error=str(e), 
                            collection_name=collection_name)
            raise
    
    @monitor_function("qdrant_client", "search_vectors", "search")
    async def search_vectors(self, collection_name: str,
                           query_vector: List[float],
                           limit: int = 10,
                           score_threshold: float = 0.0,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Filter conditions for metadata
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter
            filter_query = None
            if filter_conditions:
                filter_query = self._build_filter(filter_conditions)
            
            # Search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_query,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for point in search_result:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "document_id": point.payload.get("document_id"),
                    "content": point.payload.get("content"),
                    "type": point.payload.get("type"),
                    "metadata": point.payload.get("metadata", {}),
                    "created_at": point.payload.get("created_at"),
                    "updated_at": point.payload.get("updated_at")
                }
                results.append(result)
            
            self.logger.debug("Vector search completed", 
                            collection_name=collection_name,
                            query_length=len(query_vector),
                            results_count=len(results))
            
            return results
            
        except Exception as e:
            self.logger.error("Vector search failed", 
                            error=str(e), 
                            collection_name=collection_name)
            raise
    
    @monitor_function("qdrant_client", "delete_vectors", "vectors")
    async def delete_vectors(self, collection_name: str, 
                           document_ids: List[str]) -> bool:
        """
        Delete vectors by document IDs.
        
        Args:
            collection_name: Name of the collection
            document_ids: List of document IDs to delete
            
        Returns:
            True if vectors deleted successfully
        """
        try:
            # Build filter for document IDs
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids)
                    )
                ]
            )
            
            # Delete vectors
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=filter_query)
            )
            
            self.logger.info("Vectors deleted successfully", 
                           collection_name=collection_name,
                           deleted_count=len(document_ids))
            
            return True
            
        except Exception as e:
            self.logger.error("Vector deletion failed", 
                            error=str(e), 
                            collection_name=collection_name)
            raise
    
    @monitor_function("qdrant_client", "get_collection_info", "info")
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information and statistics."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            
            # Get collection statistics
            stats = self.client.get_collection(collection_name=collection_name)
            
            return {
                "name": collection_name,
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance,
                    "on_disk": info.config.params.vectors.on_disk
                },
                "optimizer_status": info.optimizer_status,
                "payload_schema": info.payload_schema
            }
            
        except Exception as e:
            self.logger.error("Failed to get collection info", 
                            error=str(e), 
                            collection_name=collection_name)
            raise
    
    @monitor_function("qdrant_client", "list_collections", "collections")
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their information."""
        try:
            collections = self.client.get_collections()
            
            collection_info = []
            for collection in collections.collections:
                info = await self.get_collection_info(collection.name)
                collection_info.append(info)
            
            return collection_info
            
        except Exception as e:
            self.logger.error("Failed to list collections", error=str(e))
            raise
    
    def _build_filter(self, conditions: Dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from conditions."""
        must_conditions = []
        should_conditions = []
        must_not_conditions = []
        
        for key, value in conditions.items():
            if isinstance(value, dict):
                # Handle complex conditions
                if "range" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.DatetimeRange(
                                gte=value["range"].get("gte"),
                                lte=value["range"].get("lte")
                            )
                        )
                    )
                elif "match" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value["match"])
                        )
                    )
            else:
                # Simple equality
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(
            must=must_conditions,
            should=should_conditions,
            must_not=must_not_conditions
        )
    
    @monitor_function("qdrant_client", "create_index", "index")
    async def create_index(self, collection_name: str, 
                          field_name: str,
                          field_schema: Dict[str, Any]) -> bool:
        """
        Create an index on a field for faster filtering.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field to index
            field_schema: Schema of the field
            
        Returns:
            True if index created successfully
        """
        try:
            # Create payload index
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            
            self.logger.info("Index created successfully", 
                           collection_name=collection_name,
                           field_name=field_name)
            
            return True
            
        except Exception as e:
            self.logger.error("Index creation failed", 
                            error=str(e), 
                            collection_name=collection_name,
                            field_name=field_name)
            raise
    
    @monitor_function("qdrant_client", "update_collection", "collection")
    async def update_collection(self, collection_name: str,
                               updates: Dict[str, Any]) -> bool:
        """
        Update collection configuration.
        
        Args:
            collection_name: Name of the collection
            updates: Dictionary of updates to apply
            
        Returns:
            True if collection updated successfully
        """
        try:
            # Apply updates
            if "optimizers_config" in updates:
                self.client.update_collection(
                    collection_name=collection_name,
                    optimizers_config=updates["optimizers_config"]
                )
            
            if "vectors_config" in updates:
                self.client.update_collection(
                    collection_name=collection_name,
                    vectors_config=updates["vectors_config"]
                )
            
            self.logger.info("Collection updated successfully", 
                           collection_name=collection_name)
            
            return True
            
        except Exception as e:
            self.logger.error("Collection update failed", 
                            error=str(e), 
                            collection_name=collection_name)
            raise 