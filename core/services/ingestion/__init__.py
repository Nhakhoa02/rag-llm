"""
Data ingestion and processing module for the distributed indexing system.
"""

from core.types.document_processor import DocumentProcessor
from core.types.image_processor import ImageProcessor
from core.types.tabular_processor import TabularProcessor
from .pipeline import IngestionPipeline
from .chunker import TextChunker, ImageChunker
from .extractor import ContentExtractor, MetadataExtractor

__all__ = [
    "DocumentProcessor",
    "ImageProcessor", 
    "TabularProcessor",
    "IngestionPipeline",
    "TextChunker",
    "ImageChunker",
    "ContentExtractor",
    "MetadataExtractor",
] 