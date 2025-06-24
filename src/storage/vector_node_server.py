"""
Vector Storage Node Server

Individual node in the distributed vector storage system.
Each node handles:
- Vector storage and retrieval
- Health monitoring
- Load reporting
- Collection management
"""

import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..utils.logging import get_logger
from ..models.base import BaseDocument

logger = get_logger(__name__)

@dataclass
class VectorData:
    """Represents a vector with metadata."""
    id: str
    vector: List[float]
    document: Dict[str, Any]
    shard_id: str
    collection_name: str
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

class CollectionInfo(BaseModel):
    name: str
    vector_size: int
    vector_count: int
    created_at: float

class VectorNodeServer:
    """Individual vector storage node."""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 8001, data_dir: str = "./node_data"):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # In-memory storage (in production, use persistent storage)
        self.collections: Dict[str, CollectionInfo] = {}
        self.vectors: Dict[str, VectorData] = {}  # vector_id -> VectorData
        self.shards: Dict[str, List[str]] = defaultdict(list)  # shard_id -> vector_ids
        
        # Load balancing metrics
        self.start_time = time.time()
        self.request_count = 0
        self.last_request_time = time.time()
        
        # Create FastAPI app
        self.app = FastAPI(title=f"Vector Node {node_id}", version="1.0.0")
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "node_id": self.node_id,
                "uptime": time.time() - self.start_time,
                "load": self._calculate_load(),
                "vector_count": len(self.vectors),
                "collection_count": len(self.collections),
                "request_count": self.request_count
            }
        
        @self.app.post("/collections")
        async def create_collection(request: Dict[str, Any]):
            """Create a new collection."""
            try:
                collection_name = request["collection_name"]
                vector_size = request["vector_size"]
                
                if collection_name in self.collections:
                    return {"status": "exists", "message": "Collection already exists"}
                
                collection_info = CollectionInfo(
                    name=collection_name,
                    vector_size=vector_size,
                    vector_count=0,
                    created_at=time.time()
                )
                
                self.collections[collection_name] = collection_info
                logger.info(f"Created collection: {collection_name}")
                
                return {"status": "created", "collection": collection_info.dict()}
                
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/collections/{collection_name}")
        async def delete_collection(collection_name: str):
            """Delete a collection."""
            try:
                if collection_name not in self.collections:
                    return {"status": "not_found", "message": "Collection not found"}
                
                # Remove all vectors in this collection
                vectors_to_remove = []
                for vector_id, vector_data in self.vectors.items():
                    if vector_data.collection_name == collection_name:
                        vectors_to_remove.append(vector_id)
                
                for vector_id in vectors_to_remove:
                    del self.vectors[vector_id]
                
                # Remove from shards
                for shard_id, vector_ids in self.shards.items():
                    self.shards[shard_id] = [vid for vid in vector_ids if vid not in vectors_to_remove]
                
                # Remove collection
                del self.collections[collection_name]
                
                logger.info(f"Deleted collection: {collection_name}")
                return {"status": "deleted", "message": f"Collection {collection_name} deleted"}
                
            except Exception as e:
                logger.error(f"Failed to delete collection: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/vectors")
        async def upsert_vectors(request: Dict[str, Any]):
            """Upsert vectors to a shard."""
            try:
                shard_id = request["shard_id"]
                collection_name = request.get("collection_name", "default")
                vectors_data = request["vectors"]
                documents_data = request["documents"]
                
                # Initialize shard if it doesn't exist
                if shard_id not in self.shards:
                    self.shards[shard_id] = []
                    logger.info(f"Initialized shard: {shard_id}")
                
                upserted_count = 0
                
                for i, (vector, document) in enumerate(zip(vectors_data, documents_data)):
                    # Create unique vector ID using timestamp and random component
                    unique_id = str(uuid.uuid4())
                    vector_id = f"{shard_id}_{document['id']}_{unique_id}_{i}"
                    
                    vector_data = VectorData(
                        id=vector_id,
                        vector=vector,
                        document=document,
                        shard_id=shard_id,
                        collection_name=collection_name
                    )
                    
                    self.vectors[vector_id] = vector_data
                    self.shards[shard_id].append(vector_id)
                    upserted_count += 1
                
                # Update collection counts properly
                collection_counts = {}
                for vector_data in self.vectors.values():
                    coll_name = vector_data.collection_name
                    collection_counts[coll_name] = collection_counts.get(coll_name, 0) + 1
                
                # Update collection objects
                for coll_name, count in collection_counts.items():
                    if coll_name in self.collections:
                        self.collections[coll_name].vector_count = count
                
                self.request_count += 1
                self.last_request_time = time.time()
                
                logger.info(f"Upserted {upserted_count} vectors to shard {shard_id}")
                return {"status": "success", "upserted_count": upserted_count}
                
            except Exception as e:
                logger.error(f"Failed to upsert vectors: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/search")
        async def search_vectors(request: Dict[str, Any]):
            """Search vectors in a shard."""
            try:
                shard_id = request["shard_id"]
                query_vector = request["query_vector"]
                limit = request.get("limit", 10)
                score_threshold = request.get("score_threshold", 0.0)
                
                # Get vectors in this shard
                shard_vector_ids = self.shards.get(shard_id, [])
                shard_vectors = [self.vectors[vid] for vid in shard_vector_ids if vid in self.vectors]
                
                if not shard_vectors:
                    return {"results": []}
                
                # Calculate similarities
                results = []
                query_np = np.array(query_vector)
                
                for vector_data in shard_vectors:
                    vector_np = np.array(vector_data.vector)
                    
                    # Cosine similarity
                    similarity = np.dot(query_np, vector_np) / (np.linalg.norm(query_np) * np.linalg.norm(vector_np))
                    
                    if similarity >= score_threshold:
                        results.append({
                            "document_id": vector_data.document["id"],
                            "content": vector_data.document.get("content", ""),
                            "score": float(similarity),
                            "metadata": vector_data.document.get("metadata", {}),
                            "source_index": vector_data.collection_name
                        })
                
                # Sort by score and limit
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:limit]
                
                self.request_count += 1
                self.last_request_time = time.time()
                
                return {"results": results}
                
            except Exception as e:
                logger.error(f"Failed to search vectors: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/collections")
        async def list_collections():
            """List all collections on this node."""
            return {
                "collections": [collection.dict() for collection in self.collections.values()],
                "total_vectors": len(self.vectors)
            }
        
        @self.app.get("/shards")
        async def list_shards():
            """List all shards on this node."""
            shard_info = {}
            for shard_id, vector_ids in self.shards.items():
                shard_info[shard_id] = {
                    "vector_count": len(vector_ids),
                    "collections": list(set(self.vectors[vid].collection_name for vid in vector_ids if vid in self.vectors))
                }
            return {"shards": shard_info}
        
        @self.app.get("/stats")
        async def get_stats():
            """Get node statistics."""
            return {
                "node_id": self.node_id,
                "uptime": time.time() - self.start_time,
                "load": self._calculate_load(),
                "vector_count": len(self.vectors),
                "collection_count": len(self.collections),
                "shard_count": len(self.shards),
                "request_count": self.request_count,
                "last_request_time": self.last_request_time
            }
    
    def _calculate_load(self) -> float:
        """Calculate current load (0.0 to 1.0)."""
        # Simple load calculation based on request rate and vector count
        time_since_start = time.time() - self.start_time
        if time_since_start == 0:
            return 0.0
        
        request_rate = self.request_count / time_since_start
        vector_load = min(len(self.vectors) / 10000, 1.0)  # Normalize to 10k vectors
        
        # Combine factors
        load = (request_rate * 0.3) + (vector_load * 0.7)
        return min(load, 1.0)
    
    def run(self):
        """Run the node server."""
        logger.info(f"Starting vector node {self.node_id} on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

def create_node_server(node_id: str, host: str = "localhost", port: int = 8001, data_dir: str = "./node_data"):
    """Factory function to create a vector node server."""
    return VectorNodeServer(node_id, host, port, data_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python vector_node_server.py <node_id> <host> <port>")
        sys.exit(1)
    
    node_id = sys.argv[1]
    host = sys.argv[2]
    port = int(sys.argv[3])
    
    server = create_node_server(node_id, host, port)
    server.run() 