"""
Document chunking module for splitting large documents into smaller chunks.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from core.models.base import BaseDocument, DataType
from core.utils.logging import LoggerMixin
from core.utils.metrics import monitor_function


class TextChunker(LoggerMixin):
    """Text chunking strategies for different types of content."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @monitor_function("text_chunker", "chunk_text", "text")
    def chunk_text(self, text: str, strategy: str = "fixed_size") -> List[str]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy ('fixed_size', 'semantic', 'paragraph')
            
        Returns:
            List of text chunks
        """
        if strategy == "fixed_size":
            return self._fixed_size_chunking(text)
        elif strategy == "semantic":
            return self._semantic_chunking(text)
        elif strategy == "paragraph":
            return self._paragraph_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _fixed_size_chunking(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    last_ending = text.rfind(ending, start, end)
                    if last_ending > start + self.chunk_size * 0.7:  # At least 70% of chunk size
                        end = last_ending + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """Chunk text based on semantic similarity."""
        try:
            # Split into sentences first
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= 1:
                return [text]
            
            # Create embeddings for sentences
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(sentences)
            
            # Determine number of clusters based on text length
            target_chunk_size = self.chunk_size
            num_clusters = max(1, len(text) // target_chunk_size)
            num_clusters = min(num_clusters, len(sentences))
            
            if num_clusters == 1:
                return [text]
            
            # Cluster sentences
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Group sentences by cluster
            chunked_text = [[] for _ in range(num_clusters)]
            for i, cluster_id in enumerate(clusters):
                chunked_text[cluster_id].append(sentences[i])
            
            # Convert clusters to text chunks
            chunks = []
            for cluster in chunked_text:
                if cluster:
                    chunk = ' '.join(cluster)
                    if len(chunk) > 50:  # Minimum chunk size
                        chunks.append(chunk)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            self.logger.warning("Semantic chunking failed, falling back to fixed size", error=str(e))
            return self._fixed_size_chunking(text)
    
    def _paragraph_chunking(self, text: str) -> List[str]:
        """Chunk text by paragraphs."""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
        # """Chunk text by paragraphs."""
        # # Split by double newlines (paragraphs)
        # paragraphs = re.split(r'\n\s*\n', text)
        
        # chunks = []
        # current_chunk = ""
        
        # for paragraph in paragraphs:
        #     paragraph = paragraph.strip()
        #     if not paragraph:
        #         continue
            
        #     # If adding this paragraph would exceed chunk size, start new chunk
        #     if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
        #         chunks.append(current_chunk.strip())
        #         current_chunk = paragraph
        #     else:
        #         if current_chunk:
        #             current_chunk += "\n\n" + paragraph
        #         else:
        #             current_chunk = paragraph
        
        # # Add the last chunk
        # if current_chunk:
        #     chunks.append(current_chunk.strip())
        
        # return chunks if chunks else [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def chunk_document(self, document: BaseDocument, strategy: str = "auto") -> List[BaseDocument]:
        """
        Chunk a document into smaller documents.
        
        Args:
            document: Document to chunk
            strategy: Chunking strategy ('auto', 'fixed_size', 'semantic', 'paragraph')
            
        Returns:
            List of chunked documents
        """
        if not document.content:
            return [document]
        
        # Auto-select strategy based on document type
        if strategy == "auto":
            if document.type == DataType.DOCUMENT:
                strategy = "paragraph"#semantic
            elif document.type == DataType.TABULAR:
                strategy = "paragraph"
            else:
                strategy = "fixed_size"
        
        # Chunk the text
        text_chunks = self.chunk_text(document.content, strategy)
        
        # Create chunked documents
        chunked_docs = []
        for i, chunk in enumerate(text_chunks):
            chunk_doc = BaseDocument(
                type=document.type,
                content=chunk,
                metadata=document.metadata.copy()
            )
            
            # Add chunk-specific metadata
            chunk_doc.update_metadata("chunk_index", i)
            chunk_doc.update_metadata("total_chunks", len(text_chunks))
            chunk_doc.update_metadata("parent_document_id", document.id)
            chunk_doc.update_metadata("chunk_strategy", strategy)
            chunk_doc.update_metadata("chunk_size", len(chunk))
            
            chunked_docs.append(chunk_doc)
        
        self.logger.info("Document chunked successfully", 
                        original_id=document.id, 
                        num_chunks=len(chunked_docs),
                        strategy=strategy)
        
        return chunked_docs


class ImageChunker(LoggerMixin):
    """Image chunking for large images or image collections."""
    
    def __init__(self, max_image_size: int = 1024):
        """
        Initialize image chunker.
        
        Args:
            max_image_size: Maximum size for image chunks
        """
        super().__init__()
        self.max_image_size = max_image_size
    
    @monitor_function("image_chunker", "chunk_image", "image")
    def chunk_image(self, image_path: str, strategy: str = "grid") -> List[Dict[str, Any]]:
        """
        Chunk a large image into smaller regions.
        
        Args:
            image_path: Path to the image
            strategy: Chunking strategy ('grid', 'sliding_window')
            
        Returns:
            List of image chunk information
        """
        try:
            from PIL import Image
            import cv2
            
            image = Image.open(image_path)
            width, height = image.size
            
            # If image is small enough, return as single chunk
            if width <= self.max_image_size and height <= self.max_image_size:
                return [{
                    "region": (0, 0, width, height),
                    "content": "Full image",
                    "metadata": {"size": (width, height)}
                }]
            
            if strategy == "grid":
                return self._grid_chunking(image)
            elif strategy == "sliding_window":
                return self._sliding_window_chunking(image)
            else:
                raise ValueError(f"Unknown image chunking strategy: {strategy}")
                
        except Exception as e:
            self.logger.error("Image chunking failed", error=str(e), image_path=image_path)
            return []
    
    def _grid_chunking(self, image) -> List[Dict[str, Any]]:
        """Chunk image into a grid of smaller regions."""
        width, height = image.size
        
        # Calculate grid dimensions
        cols = (width + self.max_image_size - 1) // self.max_image_size
        rows = (height + self.max_image_size - 1) // self.max_image_size
        
        chunks = []
        for row in range(rows):
            for col in range(cols):
                x1 = col * self.max_image_size
                y1 = row * self.max_image_size
                x2 = min(x1 + self.max_image_size, width)
                y2 = min(y1 + self.max_image_size, height)
                
                region = (x1, y1, x2, y2)
                chunk_info = {
                    "region": region,
                    "content": f"Image region {row}-{col}",
                    "metadata": {
                        "grid_position": (row, col),
                        "region_size": (x2 - x1, y2 - y1),
                        "total_grid": (rows, cols)
                    }
                }
                chunks.append(chunk_info)
        
        return chunks
    
    def _sliding_window_chunking(self, image) -> List[Dict[str, Any]]:
        """Chunk image using sliding window approach."""
        width, height = image.size
        step_size = self.max_image_size // 2  # 50% overlap
        
        chunks = []
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                x2 = min(x + self.max_image_size, width)
                y2 = min(y + self.max_image_size, height)
                
                region = (x, y, x2, y2)
                chunk_info = {
                    "region": region,
                    "content": f"Image window at ({x}, {y})",
                    "metadata": {
                        "window_position": (x, y),
                        "region_size": (x2 - x, y2 - y),
                        "step_size": step_size
                    }
                }
                chunks.append(chunk_info)
        
        return chunks 