"""
Consistency models for the distributed data indexing system.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, validator

from .base import ConsistencyLevel


class ReplicationConfig(BaseModel):
    """Replication configuration model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Replication configuration name")
    replication_factor: int = Field(default=3, description="Number of replicas")
    consistency_level: ConsistencyLevel = Field(default=ConsistencyLevel.QUORUM, description="Consistency level")
    sync_replication: bool = Field(default=True, description="Synchronous replication")
    timeout_ms: int = Field(default=5000, description="Replication timeout in milliseconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    nodes: List[str] = Field(default_factory=list, description="Replica nodes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        extra = "allow"
    
    @validator('replication_factor')
    def validate_replication_factor(cls, v):
        """Validate replication factor."""
        if v < 1:
            raise ValueError("Replication factor must be at least 1")
        if v > 10:
            raise ValueError("Replication factor cannot exceed 10")
        return v
    
    @validator('timeout_ms')
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @validator('retry_attempts')
    def validate_retry_attempts(cls, v):
        """Validate retry attempts."""
        if v < 0:
            raise ValueError("Retry attempts cannot be negative")
        if v > 10:
            raise ValueError("Retry attempts cannot exceed 10")
        return v
    
    def get_quorum_size(self) -> int:
        """Get quorum size based on replication factor."""
        return (self.replication_factor // 2) + 1
    
    def add_node(self, node: str) -> None:
        """Add replica node."""
        if node not in self.nodes:
            self.nodes.append(node)
            self.updated_at = datetime.utcnow()
    
    def remove_node(self, node: str) -> None:
        """Remove replica node."""
        if node in self.nodes:
            self.nodes.remove(node)
            self.updated_at = datetime.utcnow()
    
    def is_quorum_available(self, available_nodes: int) -> bool:
        """Check if quorum is available."""
        return available_nodes >= self.get_quorum_size()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "replication_factor": self.replication_factor,
            "consistency_level": self.consistency_level.value,
            "sync_replication": self.sync_replication,
            "timeout_ms": self.timeout_ms,
            "retry_attempts": self.retry_attempts,
            "nodes": self.nodes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        } 