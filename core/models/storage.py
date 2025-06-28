"""
Storage models for the distributed data indexing system.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4
from pydantic import BaseModel, Field, validator

from .base import ProcessingStatus


class StorageConfig(BaseModel):
    """Storage configuration model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Storage name")
    type: str = Field(..., description="Storage type")
    endpoint: str = Field(..., description="Storage endpoint")
    credentials: Dict[str, Any] = Field(default_factory=dict, description="Storage credentials")
    bucket: Optional[str] = Field(None, description="Storage bucket/container")
    region: Optional[str] = Field(None, description="Storage region")
    max_file_size: int = Field(default=104857600, description="Maximum file size in bytes")
    allowed_extensions: list = Field(default_factory=list, description="Allowed file extensions")
    compression: bool = Field(default=False, description="Enable compression")
    encryption: bool = Field(default=False, description="Enable encryption")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('type')
    def validate_storage_type(cls, v):
        """Validate storage type."""
        valid_types = ["local", "s3", "gcs", "azure", "minio"]
        if v not in valid_types:
            raise ValueError(f"Storage type must be one of: {valid_types}")
        return v
    
    @validator('max_file_size')
    def validate_max_file_size(cls, v):
        """Validate maximum file size."""
        if v <= 0:
            raise ValueError("Maximum file size must be positive")
        return v
    
    def get_credential(self, key: str, default: Any = None) -> Any:
        """Get storage credential."""
        return self.credentials.get(key, default)
    
    def set_credential(self, key: str, value: Any) -> None:
        """Set storage credential."""
        self.credentials[key] = value
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "endpoint": self.endpoint,
            "credentials": self.credentials,
            "bucket": self.bucket,
            "region": self.region,
            "max_file_size": self.max_file_size,
            "allowed_extensions": self.allowed_extensions,
            "compression": self.compression,
            "encryption": self.encryption,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class StorageMetadata(BaseModel):
    """Storage metadata model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = Field(..., description="Document ID")
    storage_id: str = Field(..., description="Storage ID")
    file_path: str = Field(..., description="File path in storage")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    checksum: Optional[str] = Field(None, description="File checksum")
    compression_type: Optional[str] = Field(None, description="Compression type")
    encryption_type: Optional[str] = Field(None, description="Encryption type")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('file_size')
    def validate_file_size(cls, v):
        """Validate file size."""
        if v < 0:
            raise ValueError("File size cannot be negative")
        return v
    
    @validator('mime_type')
    def validate_mime_type(cls, v):
        """Validate MIME type."""
        if not v or len(v.strip()) == 0:
            raise ValueError("MIME type cannot be empty")
        return v.strip()
    
    def get_file_extension(self) -> str:
        """Get file extension from path."""
        import os
        return os.path.splitext(self.file_path)[1]
    
    def is_compressed(self) -> bool:
        """Check if file is compressed."""
        return self.compression_type is not None
    
    def is_encrypted(self) -> bool:
        """Check if file is encrypted."""
        return self.encryption_type is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "storage_id": self.storage_id,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "checksum": self.checksum,
            "compression_type": self.compression_type,
            "encryption_type": self.encryption_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        } 