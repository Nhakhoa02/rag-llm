"""
Document models for the distributed data indexing system.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, validator

from .base import BaseDocument, BaseMetadata, DataType, ProcessingStatus


class Document(BaseDocument):
    """Document model for text-based content."""
    
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    language: Optional[str] = Field(default="en", description="Document language")
    file_path: Optional[str] = Field(None, description="Original file path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type")
    
    class Config:
        extra = "allow"
    
    def __init__(self, **data):
        if "type" not in data:
            data["type"] = DataType.DOCUMENT
        super().__init__(**data)
    
    @validator('title')
    def validate_title(cls, v):
        """Validate document title."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code."""
        if v is None:
            return "en"
        return v.lower()[:2]  # Use first two characters as language code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type
        })
        return base_dict


class DocumentChunk(BaseModel):
    """Document chunk model for text segmentation."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk position in document")
    start_position: int = Field(..., description="Start position in original text")
    end_position: int = Field(..., description="End position in original text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('content')
    def validate_content(cls, v):
        """Validate chunk content."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Chunk content cannot be empty")
        return v.strip()
    
    @validator('chunk_index')
    def validate_chunk_index(cls, v):
        """Validate chunk index."""
        if v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v
    
    @validator('end_position')
    def validate_positions(cls, v, values):
        """Validate position consistency."""
        if 'start_position' in values and v <= values['start_position']:
            raise ValueError("End position must be greater than start position")
        return v
    
    def get_length(self) -> int:
        """Get chunk length."""
        return len(self.content)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update chunk metadata."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create chunk from dictionary."""
        return cls(**data)


class DocumentMetadata(BaseMetadata):
    """Document-specific metadata model."""
    
    document_type: Optional[str] = Field(None, description="Type of document metadata")
    confidence: Optional[float] = Field(None, description="Confidence score")
    source: Optional[str] = Field(None, description="Metadata source")
    
    class Config:
        extra = "allow"
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "document_type": self.document_type,
            "confidence": self.confidence,
            "source": self.source
        })
        return base_dict 