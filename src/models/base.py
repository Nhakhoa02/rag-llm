"""
Base models and abstract classes for the distributed data indexing system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..utils.logging import LoggerMixin


class DataType(str, Enum):
    """Supported data types."""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TABULAR = "tabular"
    STRUCTURED = "structured"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConsistencyLevel(str, Enum):
    """Consistency levels for distributed operations."""
    ONE = "one"
    QUORUM = "quorum"
    ALL = "all"
    EVENTUAL = "eventual"


class BaseDocument(BaseModel, LoggerMixin):
    """Base document model for all data types."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: DataType = Field(..., description="Type of document")
    content: Optional[str] = Field(None, description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    version: int = Field(default=1, description="Document version")
    
    @validator('id')
    def validate_id(cls, v):
        """Validate document ID."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Document ID cannot be empty")
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata structure."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        return v
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata field."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field."""
        return self.metadata.get(key, default)
    
    def has_metadata(self, key: str) -> bool:
        """Check if metadata field exists."""
        return key in self.metadata
    
    def remove_metadata(self, key: str) -> None:
        """Remove metadata field."""
        if key in self.metadata:
            del self.metadata[key]
            self.updated_at = datetime.utcnow()
            self.version += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDocument":
        """Create document from dictionary."""
        return cls(**data)


class BaseMetadata(BaseModel, LoggerMixin):
    """Base metadata model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="Associated document ID")
    key: str = Field(..., description="Metadata key")
    value: Any = Field(..., description="Metadata value")
    data_type: str = Field(..., description="Data type of the value")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1)
    
    @validator('key')
    def validate_key(cls, v):
        """Validate metadata key."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Metadata key cannot be empty")
        return v.strip()
    
    def update_value(self, value: Any) -> None:
        """Update metadata value."""
        self.value = value
        self.data_type = type(value).__name__
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "key": self.key,
            "value": self.value,
            "data_type": self.data_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


class BaseIndex(BaseModel, ABC, LoggerMixin):
    """Abstract base class for indexing operations."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Index name")
    type: str = Field(..., description="Index type")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @abstractmethod
    def add_document(self, document: BaseDocument) -> bool:
        """Add document to index."""
        pass
    
    @abstractmethod
    def remove_document(self, document_id: str) -> bool:
        """Remove document from index."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents in index."""
        pass
    
    @abstractmethod
    def update_document(self, document: BaseDocument) -> bool:
        """Update document in index."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert index to dictionary."""
        return self.get_stats()


class BaseStorage(BaseModel, ABC, LoggerMixin):
    """Abstract base class for storage operations."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Storage name")
    type: str = Field(..., description="Storage type")
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    
    @abstractmethod
    def store_document(self, document: BaseDocument) -> str:
        """Store document and return storage ID."""
        pass
    
    @abstractmethod
    def retrieve_document(self, document_id: str) -> Optional[BaseDocument]:
        """Retrieve document by ID."""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        pass
    
    @abstractmethod
    def update_document(self, document: BaseDocument) -> bool:
        """Update document in storage."""
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get storage configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set storage configuration value."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert storage to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value
        }


class BaseProcessor(BaseModel, ABC, LoggerMixin):
    """Abstract base class for data processing operations."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Processor name")
    type: str = Field(..., description="Processor type")
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    
    @abstractmethod
    def process_document(self, document: BaseDocument) -> BaseDocument:
        """Process document and return processed version."""
        pass
    
    @abstractmethod
    def can_process(self, document: BaseDocument) -> bool:
        """Check if processor can handle the document type."""
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get processor configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set processor configuration value."""
        self.config[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert processor to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value
        } 