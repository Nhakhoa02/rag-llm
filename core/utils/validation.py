"""
Data validation and sanitization utilities.
"""

import re
import mimetypes
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import magic
from pydantic import BaseModel, ValidationError, validator
import structlog

from .logging import get_logger


class DataValidator:
    """Validator for data input and sanitization."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Supported file types and their MIME types
        self.supported_types = {
            'pdf': ['application/pdf'],
            'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/zip'
            ],
            'txt': ['text/plain'],
            'csv': ['text/csv', 'application/csv'],
            'json': ['application/json'],
            'xml': ['application/xml', 'text/xml'],
            'jpg': ['image/jpeg'],
            'jpeg': ['image/jpeg'],
            'png': ['image/png'],
            'gif': ['image/gif'],
            'bmp': ['image/bmp'],
            'tiff': ['image/tiff'],
            'webp': ['image/webp']
        }
        
        # File size limits (in bytes)
        self.size_limits = {
            'pdf': 100 * 1024 * 1024,  # 100MB
            'docx': 50 * 1024 * 1024,  # 50MB
            'txt': 10 * 1024 * 1024,   # 10MB
            'csv': 50 * 1024 * 1024,   # 50MB
            'json': 10 * 1024 * 1024,  # 10MB
            'xml': 10 * 1024 * 1024,   # 10MB
            'image': 20 * 1024 * 1024  # 20MB for all images
        }
        
        # Malicious patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'onclick=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>'
        ]
    
    def validate_file_type(self, file_path: Union[str, Path], 
                          allowed_types: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Validate file type using both extension and MIME type.
        
        Args:
            file_path: Path to the file
            allowed_types: List of allowed file types (extensions)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Get file extension
            extension = file_path.suffix.lower().lstrip('.')
            
            # Check if extension is in allowed types
            if allowed_types and extension not in allowed_types:
                return False, f"File type '{extension}' not allowed. Allowed: {allowed_types}"
            
            # Read file header for MIME type detection
            with open(file_path, 'rb') as f:
                header = f.read(2048)  # Read first 2KB
            
            # Detect MIME type using python-magic
            mime_type = magic.from_buffer(header, mime=True)
            
            # Validate MIME type
            if extension in self.supported_types:
                if mime_type not in self.supported_types[extension]:
                    return False, f"MIME type '{mime_type}' does not match extension '{extension}'"
            else:
                return False, f"Unsupported file type: {extension}"
            
            return True, "File type validation passed"
        
        except Exception as e:
            self.logger.error("File type validation failed", error=str(e), file_path=str(file_path))
            return False, f"File type validation error: {str(e)}"
    
    def validate_file_size(self, file_path: Union[str, Path], 
                          max_size: Optional[int] = None) -> Tuple[bool, str]:
        """
        Validate file size.
        
        Args:
            file_path: Path to the file
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"
            
            file_size = file_path.stat().st_size
            
            # Determine size limit based on file type
            if max_size is None:
                extension = file_path.suffix.lower().lstrip('.')
                if extension in self.size_limits:
                    max_size = self.size_limits[extension]
                elif extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
                    max_size = self.size_limits['image']
                else:
                    max_size = 10 * 1024 * 1024  # Default 10MB
            
            if file_size > max_size:
                return False, f"File size {file_size} bytes exceeds limit {max_size} bytes"
            
            return True, "File size validation passed"
        
        except Exception as e:
            self.logger.error("File size validation failed", error=str(e), file_path=str(file_path))
            return False, f"File size validation error: {str(e)}"
    
    def validate_content_safety(self, content: str) -> Tuple[bool, str]:
        """
        Validate content for malicious patterns.
        
        Args:
            content: Content to validate
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            # Check for malicious patterns
            for pattern in self.malicious_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    return False, f"Content contains malicious pattern: {pattern}"
            
            # Check for suspicious file paths
            suspicious_paths = [
                r'\.\./',  # Directory traversal
                r'\.\.\\',  # Windows directory traversal
                r'\/etc\/',  # System directories
                r'C:\\',    # Windows system drive
            ]
            
            for pattern in suspicious_paths:
                if re.search(pattern, content, re.IGNORECASE):
                    return False, f"Content contains suspicious path pattern: {pattern}"
            
            return True, "Content safety validation passed"
        
        except Exception as e:
            self.logger.error("Content safety validation failed", error=str(e))
            return False, f"Content safety validation error: {str(e)}"
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text content.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove control characters
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            return text.strip()
        
        except Exception as e:
            self.logger.error("Text sanitization failed", error=str(e))
            return text
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate metadata structure and content.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required fields
            required_fields = ['id', 'type', 'created_at']
            for field in required_fields:
                if field not in metadata:
                    return False, f"Missing required field: {field}"
            
            # Validate ID format
            if not isinstance(metadata['id'], str) or len(metadata['id']) == 0:
                return False, "ID must be a non-empty string"
            
            # Validate type
            if metadata['type'] not in ['document', 'image', 'video', 'audio']:
                return False, f"Invalid type: {metadata['type']}"
            
            # Validate timestamp
            if not isinstance(metadata['created_at'], str):
                return False, "created_at must be a string"
            
            # Check for suspicious metadata
            suspicious_keys = ['password', 'token', 'key', 'secret']
            for key in suspicious_keys:
                if key in metadata:
                    return False, f"Suspicious metadata key found: {key}"
            
            return True, "Metadata validation passed"
        
        except Exception as e:
            self.logger.error("Metadata validation failed", error=str(e))
            return False, f"Metadata validation error: {str(e)}"
    
    def calculate_file_hash(self, file_path: Union[str, Path], 
                           algorithm: str = "sha256") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            
        Returns:
            Hexadecimal hash string
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            hash_func = getattr(hashlib, algorithm)()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
        
        except Exception as e:
            self.logger.error("File hash calculation failed", error=str(e), file_path=str(file_path))
            raise
    
    def validate_vector_dimensions(self, vector: List[float], 
                                 expected_dim: int) -> Tuple[bool, str]:
        """
        Validate vector dimensions.
        
        Args:
            vector: Vector to validate
            expected_dim: Expected dimension
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not isinstance(vector, list):
                return False, "Vector must be a list"
            
            if len(vector) != expected_dim:
                return False, f"Vector dimension {len(vector)} does not match expected {expected_dim}"
            
            # Check if all elements are numeric
            if not all(isinstance(x, (int, float)) for x in vector):
                return False, "All vector elements must be numeric"
            
            # Check for NaN or infinite values
            if any(not (x == x) or (x == float('inf')) or (x == float('-inf')) for x in vector):
                return False, "Vector contains NaN or infinite values"
            
            return True, "Vector validation passed"
        
        except Exception as e:
            self.logger.error("Vector validation failed", error=str(e))
            return False, f"Vector validation error: {str(e)}"
    
    def validate_query_parameters(self, query: str, 
                                max_length: int = 1000) -> Tuple[bool, str]:
        """
        Validate query parameters.
        
        Args:
            query: Query string to validate
            max_length: Maximum query length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not isinstance(query, str):
                return False, "Query must be a string"
            
            if len(query) == 0:
                return False, "Query cannot be empty"
            
            if len(query) > max_length:
                return False, f"Query length {len(query)} exceeds maximum {max_length}"
            
            # Check for malicious patterns
            is_safe, error_msg = self.validate_content_safety(query)
            if not is_safe:
                return False, error_msg
            
            return True, "Query validation passed"
        
        except Exception as e:
            self.logger.error("Query validation failed", error=str(e))
            return False, f"Query validation error: {str(e)}"


class ValidationSchema(BaseModel):
    """Base validation schema for data models."""
    
    @validator('*', pre=True)
    def validate_all_fields(cls, v):
        """Validate all fields."""
        if isinstance(v, str):
            return v.strip()
        return v


# Global validator instance
data_validator = DataValidator() 