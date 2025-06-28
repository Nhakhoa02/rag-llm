"""
Content and metadata extraction for documents.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from core.models.base import BaseDocument
from core.utils.logging import get_logger


class ExtractionConfig(BaseModel):
    """Extraction configuration."""
    
    enable_ai_extraction: bool = Field(default=True, description="Enable AI-powered extraction")
    enable_keyword_extraction: bool = Field(default=True, description="Enable keyword extraction")
    enable_entity_extraction: bool = Field(default=True, description="Enable entity extraction")
    max_keywords: int = Field(default=10, description="Maximum keywords to extract")
    
    class Config:
        extra = "allow"


class ContentExtractor:
    """Extracts and enhances document content."""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.logger = get_logger(__name__)
    
    async def extract(self, document: BaseDocument) -> BaseDocument:
        """
        Extract and enhance content from document.
        
        Args:
            document: Document to extract content from
            
        Returns:
            Enhanced document with extracted content
        """
        try:
            if not document.content:
                self.logger.warning(f"No content to extract from document: {document.id}")
                return document
            
            # Extract keywords
            if self.config.enable_keyword_extraction:
                keywords = await self._extract_keywords(document.content)
                document.update_metadata("keywords", keywords)
            
            # Extract entities
            if self.config.enable_entity_extraction:
                entities = await self._extract_entities(document.content)
                document.update_metadata("entities", entities)
            
            # AI-powered content enhancement
            if self.config.enable_ai_extraction:
                enhanced_content = await self._enhance_content(document.content)
                if enhanced_content:
                    document.content = enhanced_content
            
            self.logger.info(f"Content extraction completed for document: {document.id}")
            return document
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for document {document.id}: {e}")
            return document
    
    async def _extract_keywords(self, content: str) -> list:
        """Extract keywords from content."""
        try:
            # Simple keyword extraction (can be enhanced with NLP libraries)
            import re
            from collections import Counter
            
            # Remove punctuation and convert to lowercase
            words = re.findall(r'\b\w+\b', content.lower())
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Return top keywords
            return [word for word, count in word_counts.most_common(self.config.max_keywords)]
            
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    async def _extract_entities(self, content: str) -> Dict[str, list]:
        """Extract named entities from content."""
        try:
            # Simple entity extraction (can be enhanced with NER libraries)
            entities = {
                "organizations": [],
                "people": [],
                "locations": [],
                "dates": []
            }
            
            # Basic pattern matching for entities
            import re
            
            # Extract potential organizations (words starting with capital letters)
            org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            potential_orgs = re.findall(org_pattern, content)
            entities["organizations"] = potential_orgs[:5]  # Limit to 5
            
            # Extract dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
            dates = re.findall(date_pattern, content)
            entities["dates"] = dates
            
            return entities
            
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
            return {"organizations": [], "people": [], "locations": [], "dates": []}
    
    async def _enhance_content(self, content: str) -> Optional[str]:
        """Enhance content using AI (placeholder for future implementation)."""
        try:
            # This would integrate with AI services like Gemini for content enhancement
            # For now, return the original content
            return content
            
        except Exception as e:
            self.logger.warning(f"Content enhancement failed: {e}")
            return None


class MetadataExtractor:
    """Extracts metadata from documents."""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.logger = get_logger(__name__)
    
    async def extract(self, document: BaseDocument) -> BaseDocument:
        """
        Extract metadata from document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Document with extracted metadata
        """
        try:
            # Extract basic metadata
            basic_metadata = await self._extract_basic_metadata(document)
            for key, value in basic_metadata.items():
                document.update_metadata(key, value)
            
            # Extract content-based metadata
            if document.content:
                content_metadata = await self._extract_content_metadata(document.content)
                for key, value in content_metadata.items():
                    document.update_metadata(key, value)
            
            # AI-powered metadata extraction
            if self.config.enable_ai_extraction:
                ai_metadata = await self._extract_ai_metadata(document)
                for key, value in ai_metadata.items():
                    document.update_metadata(key, value)
            
            self.logger.info(f"Metadata extraction completed for document: {document.id}")
            return document
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed for document {document.id}: {e}")
            return document
    
    async def _extract_basic_metadata(self, document: BaseDocument) -> Dict[str, Any]:
        """Extract basic metadata from document."""
        metadata = {}
        
        try:
            # Extract file information from metadata
            file_path = document.get_metadata("file_path")
            if file_path:
                import os
                
                metadata["file_name"] = os.path.basename(file_path)
                metadata["file_extension"] = os.path.splitext(file_path)[1]
                metadata["file_directory"] = os.path.dirname(file_path)
                
                file_size = document.get_metadata("file_size")
                if file_size:
                    metadata["file_size_mb"] = round(file_size / (1024 * 1024), 2)
            
            # Extract MIME type from metadata
            mime_type = document.get_metadata("mime_type")
            if mime_type:
                metadata["mime_type"] = mime_type
            
            # Extract language from metadata
            language = document.get_metadata("language")
            if language:
                metadata["language"] = language
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Basic metadata extraction failed: {e}")
            return {}
    
    async def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content."""
        metadata = {}
        
        try:
            # Calculate content statistics
            metadata["content_length"] = len(content)
            metadata["word_count"] = len(content.split())
            metadata["character_count"] = len(content.replace(" ", ""))
            
            # Calculate readability metrics
            sentences = content.split('.')
            metadata["sentence_count"] = len([s for s in sentences if s.strip()])
            
            if metadata["sentence_count"] > 0:
                metadata["avg_sentence_length"] = round(metadata["word_count"] / metadata["sentence_count"], 2)
            
            # Detect content type
            if any(keyword in content.lower() for keyword in ['technical', 'specification', 'api', 'code']):
                metadata["content_type"] = "technical"
            elif any(keyword in content.lower() for keyword in ['news', 'article', 'report']):
                metadata["content_type"] = "news"
            elif any(keyword in content.lower() for keyword in ['story', 'narrative', 'fiction']):
                metadata["content_type"] = "narrative"
            else:
                metadata["content_type"] = "general"
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Content metadata extraction failed: {e}")
            return {}
    
    async def _extract_ai_metadata(self, document: BaseDocument) -> Dict[str, Any]:
        """Extract metadata using AI (placeholder for future implementation)."""
        try:
            # This would integrate with AI services for advanced metadata extraction
            # For now, return empty metadata
            return {}
            
        except Exception as e:
            self.logger.warning(f"AI metadata extraction failed: {e}")
            return {} 