"""
Query models for the distributed data indexing system.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, validator

from .base import ProcessingStatus


class QueryRequest(BaseModel):
    """Query request model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = Field(..., description="Search query")
    query_type: str = Field(default="semantic", description="Query type")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    limit: int = Field(default=10, description="Maximum results")
    offset: int = Field(default=0, description="Result offset")
    include_vectors: bool = Field(default=False, description="Include vectors in response")
    include_metadata: bool = Field(default=True, description="Include metadata in response")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('query_type')
    def validate_query_type(cls, v):
        """Validate query type."""
        valid_types = ["semantic", "keyword", "hybrid", "metadata"]
        if v not in valid_types:
            raise ValueError(f"Query type must be one of: {valid_types}")
        return v
    
    @validator('limit')
    def validate_limit(cls, v):
        """Validate result limit."""
        if v <= 0:
            raise ValueError("Limit must be positive")
        if v > 1000:
            raise ValueError("Limit cannot exceed 1000")
        return v
    
    @validator('offset')
    def validate_offset(cls, v):
        """Validate result offset."""
        if v < 0:
            raise ValueError("Offset must be non-negative")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "query_type": self.query_type,
            "filters": self.filters,
            "limit": self.limit,
            "offset": self.offset,
            "include_vectors": self.include_vectors,
            "include_metadata": self.include_metadata,
            "created_at": self.created_at.isoformat()
        }


class SearchResult(BaseModel):
    """Search result model."""
    
    document_id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Relevance score")
    content: Optional[str] = Field(None, description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    vector: Optional[List[float]] = Field(None, description="Document vector")
    highlight: Optional[str] = Field(None, description="Highlighted text")
    
    class Config:
        extra = "allow"
    
    @validator('score')
    def validate_score(cls, v):
        """Validate relevance score."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "document_id": self.document_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "vector": self.vector,
            "highlight": self.highlight
        }


class QueryResponse(BaseModel):
    """Query response model."""
    
    request_id: str = Field(..., description="Original request ID")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_count: int = Field(default=0, description="Total matching documents")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    status: ProcessingStatus = Field(default=ProcessingStatus.COMPLETED)
    error: Optional[str] = Field(None, description="Error message")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('total_count')
    def validate_total_count(cls, v):
        """Validate total count."""
        if v < 0:
            raise ValueError("Total count cannot be negative")
        return v
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        """Validate processing time."""
        if v < 0.0:
            raise ValueError("Processing time cannot be negative")
        return v
    
    def add_result(self, result: SearchResult) -> None:
        """Add search result."""
        self.results.append(result)
        self.total_count = len(self.results)
    
    def get_top_results(self, limit: int = 10) -> List[SearchResult]:
        """Get top results by score."""
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)
        return sorted_results[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "request_id": self.request_id,
            "results": [result.to_dict() for result in self.results],
            "total_count": self.total_count,
            "processing_time": self.processing_time,
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at.isoformat()
        } 