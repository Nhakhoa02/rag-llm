"""
Index models for the distributed data indexing system.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, validator

from .base import BaseIndex, ProcessingStatus


class IndexEntry(BaseModel):
    """Index entry model for storing indexed data."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="Document ID")
    index_id: str = Field(..., description="Index ID")
    vector: Optional[List[float]] = Field(None, description="Document vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Index metadata")
    score: Optional[float] = Field(None, description="Indexing score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('vector')
    def validate_vector(cls, v):
        """Validate vector dimensions."""
        if v is not None and len(v) == 0:
            raise ValueError("Vector cannot be empty")
        return v
    
    @validator('score')
    def validate_score(cls, v):
        """Validate score range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "index_id": self.index_id,
            "vector": self.vector,
            "metadata": self.metadata,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexEntry":
        """Create entry from dictionary."""
        return cls(**data)


class VectorIndex(BaseIndex):
    """Vector index model for similarity search."""
    
    dimension: int = Field(..., description="Vector dimension")
    distance_metric: str = Field(default="cosine", description="Distance metric")
    max_vectors: Optional[int] = Field(None, description="Maximum vectors")
    
    class Config:
        extra = "allow"
    
    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "vector"
        super().__init__(**data)
    
    @validator('dimension')
    def validate_dimension(cls, v):
        """Validate vector dimension."""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v
    
    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        """Validate distance metric."""
        valid_metrics = ["cosine", "euclidean", "manhattan", "dot"]
        if v not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v
    
    def add_document(self, document: Any) -> bool:
        """Add document to vector index."""
        # Implementation would go here
        return True
    
    def remove_document(self, document_id: str) -> bool:
        """Remove document from vector index."""
        # Implementation would go here
        return True
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents in vector index."""
        # Implementation would go here
        return []
    
    def update_document(self, document: Any) -> bool:
        """Update document in vector index."""
        # Implementation would go here
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vector index to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "max_vectors": self.max_vectors
        })
        return base_dict


class MetadataIndex(BaseIndex):
    """Metadata index model for structured search."""
    
    fields: List[str] = Field(default_factory=list, description="Indexed fields")
    index_type: str = Field(default="btree", description="Index type")
    
    class Config:
        extra = "allow"
    
    def __init__(self, **data):
        if "type" not in data:
            data["type"] = "metadata"
        super().__init__(**data)
    
    @validator('index_type')
    def validate_index_type(cls, v):
        """Validate index type."""
        valid_types = ["btree", "hash", "fulltext"]
        if v not in valid_types:
            raise ValueError(f"Index type must be one of: {valid_types}")
        return v
    
    def add_document(self, document: Any) -> bool:
        """Add document to metadata index."""
        # Implementation would go here
        return True
    
    def remove_document(self, document_id: str) -> bool:
        """Remove document from metadata index."""
        # Implementation would go here
        return True
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents in metadata index."""
        # Implementation would go here
        return []
    
    def update_document(self, document: Any) -> bool:
        """Update document in metadata index."""
        # Implementation would go here
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata index to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "fields": self.fields,
            "index_type": self.index_type
        })
        return base_dict 