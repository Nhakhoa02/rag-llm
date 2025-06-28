"""
CSV Index models for storing and retrieving CSV structure information.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pydantic import BaseModel, Field, validator
import json

from .base import BaseDocument, DataType, ProcessingStatus


class CSVIndex(BaseModel):
    """CSV Index model for storing first and second row information."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    csv_file_id: str = Field(..., description="ID of the CSV file document")
    csv_filename: str = Field(..., description="Original CSV filename")
    
    # First row (column headers)
    column_headers: List[str] = Field(..., description="Column headers from first row")
    
    # Second row (sample data)
    sample_data: Optional[List[str]] = Field(None, description="Sample data from second row")
    
    # Additional metadata
    total_rows: int = Field(..., description="Total number of rows in CSV")
    total_columns: int = Field(..., description="Total number of columns in CSV")
    
    # Data types inferred from sample
    inferred_types: Optional[Dict[str, str]] = Field(None, description="Inferred data types for columns")
    
    # Index content for vector search
    index_content: str = Field(default="", description="Formatted content for vector indexing")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Generate index content if not provided
        if not self.index_content:
            self.index_content = self.generate_index_content()
    
    @validator('column_headers')
    def validate_column_headers(cls, v):
        """Validate column headers."""
        if not v or len(v) == 0:
            raise ValueError("Column headers cannot be empty")
        return [str(header).strip() for header in v if str(header).strip()]
    
    @validator('sample_data')
    def validate_sample_data(cls, v, values):
        """Validate sample data matches column count."""
        if v is not None:
            column_count = len(values.get('column_headers', []))
            if len(v) != column_count:
                # Pad or truncate to match column count
                if len(v) < column_count:
                    v.extend([''] * (column_count - len(v)))
                else:
                    v = v[:column_count]
        return v
    
    @validator('total_rows')
    def validate_total_rows(cls, v):
        """Validate total rows."""
        if v < 1:
            raise ValueError("Total rows must be at least 1")
        return v
    
    @validator('total_columns')
    def validate_total_columns(cls, v):
        """Validate total columns."""
        if v < 1:
            raise ValueError("Total columns must be at least 1")
        return v
    
    def generate_index_content(self) -> str:
        """Generate formatted content for vector indexing."""
        content_parts = []
        
        # Add CSV overview
        content_parts.append(f"CSV File: {self.csv_filename}")
        content_parts.append(f"Total Rows: {self.total_rows}")
        content_parts.append(f"Total Columns: {self.total_columns}")
        content_parts.append("")
        
        # Add column headers
        content_parts.append("Column Headers:")
        for i, header in enumerate(self.column_headers, 1):
            content_parts.append(f"{i}. {header}")
        content_parts.append("")
        
        # Add sample data if available
        if self.sample_data:
            content_parts.append("Sample Data (Row 2):")
            for i, (header, value) in enumerate(zip(self.column_headers, self.sample_data), 1):
                content_parts.append(f"{i}. {header}: {value}")
            content_parts.append("")
        
        # Add inferred types if available
        if self.inferred_types:
            content_parts.append("Inferred Data Types:")
            for col, dtype in self.inferred_types.items():
                content_parts.append(f"- {col}: {dtype}")
            content_parts.append("")
        
        return "\n".join(content_parts)
    
    def update_index_content(self) -> None:
        """Update the index content."""
        self.index_content = self.generate_index_content()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CSV index to dictionary."""
        return {
            "id": self.id,
            "csv_file_id": self.csv_file_id,
            "csv_filename": self.csv_filename,
            "column_headers": self.column_headers,
            "sample_data": self.sample_data,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "inferred_types": self.inferred_types,
            "index_content": self.index_content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CSVIndex":
        """Create CSV index from dictionary."""
        return cls(**data)
    
    def get_column_mapping(self) -> Dict[str, int]:
        """Get mapping of column names to their positions."""
        return {header: i for i, header in enumerate(self.column_headers)}
    
    def get_relevant_columns(self, query_terms: List[str]) -> List[str]:
        """Get columns that are relevant to the given query terms."""
        relevant_columns = []
        query_terms_lower = [term.lower() for term in query_terms]
        
        for header in self.column_headers:
            header_lower = header.lower()
            if any(term in header_lower for term in query_terms_lower):
                relevant_columns.append(header)
        
        return relevant_columns


class CSVIndexDocument(BaseDocument):
    """Document wrapper for CSV Index to integrate with the vector storage system."""
    
    csv_index: Optional[CSVIndex] = Field(None, description="The CSV index data")
    
    class Config:
        extra = "allow"
    
    def __init__(self, csv_index: Optional[CSVIndex] = None, **kwargs):
        # Set the content to the index content for vector search
        if csv_index:
            kwargs["content"] = csv_index.index_content
            kwargs["metadata"] = kwargs.get("metadata", {})
            
            # Add CSV-specific metadata
            kwargs["metadata"].update({
                "csv_file_id": csv_index.csv_file_id,
                "csv_filename": csv_index.csv_filename,
                "total_rows": csv_index.total_rows,
                "total_columns": csv_index.total_columns,
                "column_headers": json.dumps(csv_index.column_headers),  # Convert list to JSON string
                "index_type": "csv_index"
            })
        else:
            # Handle case where csv_index is not provided (e.g., from search results)
            kwargs["content"] = kwargs.get("content", "")
            kwargs["metadata"] = kwargs.get("metadata", {})
        
        kwargs["type"] = DataType.TABULAR
        
        super().__init__(**kwargs)
        self.csv_index = csv_index
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        if self.csv_index:
            base_dict["csv_index"] = self.csv_index.to_dict()
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CSVIndexDocument":
        """Create from dictionary."""
        csv_index_data = data.pop("csv_index", None)
        csv_index = None
        if csv_index_data:
            csv_index = CSVIndex.from_dict(csv_index_data)
        return cls(csv_index=csv_index, **data) 