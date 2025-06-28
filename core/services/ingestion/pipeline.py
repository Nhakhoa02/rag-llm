"""
Data ingestion pipeline for processing and indexing documents.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from core.models.base import BaseDocument, ProcessingStatus
from core.utils.logging import get_logger
from core.types.document_processor import DocumentProcessor
from core.types.image_processor import ImageProcessor
from core.types.tabular_processor import TabularProcessor
from .chunker import TextChunker, ImageChunker
from .extractor import ContentExtractor, MetadataExtractor


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    
    enable_chunking: bool = Field(default=True, description="Enable document chunking")
    enable_metadata_extraction: bool = Field(default=True, description="Enable metadata extraction")
    enable_content_extraction: bool = Field(default=True, description="Enable content extraction")
    chunk_size: int = Field(default=1000, description="Chunk size for text documents")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    max_file_size: int = Field(default=104857600, description="Maximum file size in bytes")
    
    class Config:
        extra = "allow"


class IngestionPipeline:
    """Orchestrates the document ingestion process."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = get_logger(__name__)
        
        # Initialize processors
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
        self.tabular_processor = TabularProcessor()
        
        # Initialize chunkers
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.image_chunker = ImageChunker()
        
        # Initialize extractors
        self.content_extractor = ContentExtractor()
        self.metadata_extractor = MetadataExtractor()
    
    async def process_file(self, file_path: str) -> List[BaseDocument]:
        """
        Process a single file through the ingestion pipeline.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of processed documents/chunks
        """
        self.logger.info(f"Starting pipeline processing for: {file_path}")
        
        try:
            # Step 1: Determine file type and select processor
            processor = self._select_processor(file_path)
            if not processor:
                self.logger.error(f"No suitable processor found for: {file_path}")
                return []
            
            # Step 2: Process document
            if isinstance(processor, ImageProcessor):
                document = await processor.process_image(file_path)
            elif isinstance(processor, TabularProcessor):
                document = await processor.process_tabular(file_path)
            else:
                document = await processor.process_document(file_path)
            
            if not document:
                self.logger.error(f"Failed to process document: {file_path}")
                return []
            
            self.logger.info(f"Document processed successfully: {document.id}")
            
            # Step 3: Extract content (if enabled)
            if self.config.enable_content_extraction:
                document = await self._extract_content(document)
            
            # Step 4: Extract metadata (if enabled)
            if self.config.enable_metadata_extraction:
                document = await self._extract_metadata(document)
            
            # Step 5: Chunk document (if enabled)
            if self.config.enable_chunking:
                chunks = await self._chunk_document(document)
                self.logger.info(f"Document chunked into {len(chunks)} pieces")
                return chunks
            else:
                return [document]
                
        except Exception as e:
            self.logger.error(f"Pipeline processing failed for {file_path}: {e}")
            return []
    
    async def process_directory(self, directory_path: str) -> List[BaseDocument]:
        """
        Process all files in a directory through the ingestion pipeline.
        
        Args:
            directory_path: Path to the directory to process
            
        Returns:
            List of processed documents/chunks
        """
        self.logger.info(f"Starting directory processing: {directory_path}")
        
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            self.logger.error(f"Invalid directory path: {directory_path}")
            return []
        
        all_documents = []
        
        # Process all files in directory
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    documents = await self.process_file(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    self.logger.error(f"Failed to process file {file_path}: {e}")
                    continue
        
        self.logger.info(f"Directory processing completed. Total documents: {len(all_documents)}")
        return all_documents
    
    def _select_processor(self, file_path: str):
        """Select appropriate processor based on file type."""
        file_path_lower = file_path.lower()
        
        # Image files
        if any(ext in file_path_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
            return self.image_processor
        
        # Tabular files
        elif any(ext in file_path_lower for ext in ['.csv', '.xlsx', '.xls', '.tsv']):
            return self.tabular_processor
        
        # Document files
        elif any(ext in file_path_lower for ext in ['.txt', '.pdf', '.doc', '.docx', '.md']):
            return self.document_processor
        
        return None
    
    async def _extract_content(self, document: BaseDocument) -> BaseDocument:
        """Extract content from document."""
        try:
            return await self.content_extractor.extract(document)
        except Exception as e:
            self.logger.warning(f"Content extraction failed: {e}")
            return document
    
    async def _extract_metadata(self, document: BaseDocument) -> BaseDocument:
        """Extract metadata from document."""
        try:
            return await self.metadata_extractor.extract(document)
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {e}")
            return document
    
    async def _chunk_document(self, document: BaseDocument) -> List[BaseDocument]:
        """Chunk document into smaller pieces."""
        try:
            if document.type.value == "image":
                return self.image_chunker.chunk_document(document, "auto")
            else:
                return self.text_chunker.chunk_document(document, "auto")
        except Exception as e:
            self.logger.warning(f"Document chunking failed: {e}")
            return [document]
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "config": self.config.dict(),
            "processors": {
                "document": self.document_processor is not None,
                "image": self.image_processor is not None,
                "tabular": self.tabular_processor is not None
            },
            "chunkers": {
                "text": self.text_chunker is not None,
                "image": self.image_chunker is not None
            },
            "extractors": {
                "content": self.content_extractor is not None,
                "metadata": self.metadata_extractor is not None
            }
        } 