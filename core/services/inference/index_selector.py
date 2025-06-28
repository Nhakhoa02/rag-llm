"""
Index selector for choosing appropriate indexes for different query types.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from core.models.base import BaseIndex
from core.utils.logging import get_logger


class IndexSelection(BaseModel):
    """Result of index selection."""
    
    selected_indexes: List[str] = Field(default_factory=list, description="Selected index IDs")
    selection_reason: str = Field(default="", description="Reason for selection")
    confidence: float = Field(default=0.0, description="Selection confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"


class IndexSelector:
    """Selects appropriate indexes for different query types."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.available_indexes: Dict[str, BaseIndex] = {}
    
    def register_index(self, index: BaseIndex) -> None:
        """Register an index for selection."""
        self.available_indexes[index.id] = index
        self.logger.info(f"Registered index: {index.name} ({index.id})")
    
    def unregister_index(self, index_id: str) -> None:
        """Unregister an index."""
        if index_id in self.available_indexes:
            del self.available_indexes[index_id]
            self.logger.info(f"Unregistered index: {index_id}")
    
    def select_indexes(self, query: str, query_type: str = "semantic") -> IndexSelection:
        """
        Select appropriate indexes for a query.
        
        Args:
            query: Search query
            query_type: Type of query (semantic, keyword, hybrid, metadata)
            
        Returns:
            Index selection result
        """
        try:
            if not self.available_indexes:
                return IndexSelection(
                    selected_indexes=[],
                    selection_reason="No indexes available",
                    confidence=0.0
                )
            
            # Select indexes based on query type
            if query_type == "semantic":
                return self._select_semantic_indexes(query)
            elif query_type == "keyword":
                return self._select_keyword_indexes(query)
            elif query_type == "hybrid":
                return self._select_hybrid_indexes(query)
            elif query_type == "metadata":
                return self._select_metadata_indexes(query)
            else:
                return self._select_default_indexes(query)
                
        except Exception as e:
            self.logger.error(f"Index selection failed for query '{query}': {e}")
            return IndexSelection(
                selected_indexes=[],
                selection_reason=f"Selection failed: {str(e)}",
                confidence=0.0
            )
    
    def _select_semantic_indexes(self, query: str) -> IndexSelection:
        """Select indexes for semantic search."""
        selected = []
        reason = "Semantic search requires vector indexes"
        
        for index_id, index in self.available_indexes.items():
            if index.type == "vector":
                selected.append(index_id)
        
        confidence = 0.8 if selected else 0.0
        
        return IndexSelection(
            selected_indexes=selected,
            selection_reason=reason,
            confidence=confidence
        )
    
    def _select_keyword_indexes(self, query: str) -> IndexSelection:
        """Select indexes for keyword search."""
        selected = []
        reason = "Keyword search requires full-text indexes"
        
        for index_id, index in self.available_indexes.items():
            if index.type in ["fulltext", "metadata"]:
                selected.append(index_id)
        
        confidence = 0.7 if selected else 0.0
        
        return IndexSelection(
            selected_indexes=selected,
            selection_reason=reason,
            confidence=confidence
        )
    
    def _select_hybrid_indexes(self, query: str) -> IndexSelection:
        """Select indexes for hybrid search."""
        selected = []
        reason = "Hybrid search requires both vector and keyword indexes"
        
        vector_indexes = []
        keyword_indexes = []
        
        for index_id, index in self.available_indexes.items():
            if index.type == "vector":
                vector_indexes.append(index_id)
            elif index.type in ["fulltext", "metadata"]:
                keyword_indexes.append(index_id)
        
        selected.extend(vector_indexes)
        selected.extend(keyword_indexes)
        
        confidence = 0.9 if vector_indexes and keyword_indexes else 0.6 if selected else 0.0
        
        return IndexSelection(
            selected_indexes=selected,
            selection_reason=reason,
            confidence=confidence
        )
    
    def _select_metadata_indexes(self, query: str) -> IndexSelection:
        """Select indexes for metadata search."""
        selected = []
        reason = "Metadata search requires metadata indexes"
        
        for index_id, index in self.available_indexes.items():
            if index.type == "metadata":
                selected.append(index_id)
        
        confidence = 0.8 if selected else 0.0
        
        return IndexSelection(
            selected_indexes=selected,
            selection_reason=reason,
            confidence=confidence
        )
    
    def _select_default_indexes(self, query: str) -> IndexSelection:
        """Select default indexes when query type is unknown."""
        selected = []
        reason = "Default selection - using all available indexes"
        
        # Select all available indexes
        selected = list(self.available_indexes.keys())
        
        confidence = 0.5 if selected else 0.0
        
        return IndexSelection(
            selected_indexes=selected,
            selection_reason=reason,
            confidence=confidence
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about available indexes."""
        stats = {
            "total_indexes": len(self.available_indexes),
            "index_types": {},
            "index_names": []
        }
        
        for index_id, index in self.available_indexes.items():
            # Count by type
            index_type = index.type
            stats["index_types"][index_type] = stats["index_types"].get(index_type, 0) + 1
            
            # Collect names
            stats["index_names"].append({
                "id": index_id,
                "name": index.name,
                "type": index.type,
                "status": index.status.value
            })
        
        return stats
    
    def recommend_indexes(self, query: str) -> List[str]:
        """Recommend indexes based on query analysis."""
        recommendations = []
        
        # Simple heuristics for recommendations
        query_lower = query.lower()
        
        # Check for semantic indicators
        if any(word in query_lower for word in ['similar', 'like', 'related', 'semantic']):
            for index_id, index in self.available_indexes.items():
                if index.type == "vector":
                    recommendations.append(index_id)
        
        # Check for keyword indicators
        if any(word in query_lower for word in ['exact', 'keyword', 'phrase']):
            for index_id, index in self.available_indexes.items():
                if index.type in ["fulltext", "metadata"]:
                    recommendations.append(index_id)
        
        # If no specific recommendations, return all indexes
        if not recommendations:
            recommendations = list(self.available_indexes.keys())
        
        return recommendations 