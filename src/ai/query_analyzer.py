"""
Query analyzer for understanding and optimizing search queries.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..utils.logging import get_logger


class QueryAnalysis(BaseModel):
    """Result of query analysis."""
    
    intent: str = Field(default="", description="Query intent")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    confidence: float = Field(default=0.0, description="Analysis confidence")
    
    class Config:
        extra = "allow"


class QueryAnalyzer:
    """Analyzes search queries to understand intent and extract components."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a search query to understand its components.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Query analysis result
        """
        try:
            if not query or not query.strip():
                return QueryAnalysis(
                    intent="empty",
                    confidence=0.0
                )
            
            query = query.strip()
            
            # Extract intent
            intent = self._extract_intent(query)
            
            # Extract entities
            entities = self._extract_entities(query)
            
            # Extract keywords
            keywords = self._extract_keywords(query)
            
            # Extract filters
            filters = self._extract_filters(query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, intent, entities, keywords)
            
            return QueryAnalysis(
                intent=intent,
                entities=entities,
                keywords=keywords,
                filters=filters,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Query analysis failed for '{query}': {e}")
            return QueryAnalysis(
                intent="unknown",
                confidence=0.0
            )
    
    def _extract_intent(self, query: str) -> str:
        """Extract query intent."""
        query_lower = query.lower()
        
        # Question intent
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            return "question"
        
        # Comparison intent
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better', 'best']):
            return "comparison"
        
        # Definition intent
        if any(word in query_lower for word in ['define', 'definition', 'meaning', 'what is']):
            return "definition"
        
        # Search intent
        if any(word in query_lower for word in ['find', 'search', 'look for', 'locate']):
            return "search"
        
        # Default to general search
        return "search"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        import re
        
        entities = []
        
        # Extract potential organizations (words starting with capital letters)
        org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_orgs = re.findall(org_pattern, query)
        entities.extend(potential_orgs)
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
        dates = re.findall(date_pattern, query)
        entities.extend(dates)
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, query)
        entities.extend(numbers)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        import re
        
        # Remove punctuation and convert to lowercase
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from query."""
        filters = {}
        
        query_lower = query.lower()
        
        # Extract date filters
        if 'today' in query_lower:
            filters['date'] = 'today'
        elif 'yesterday' in query_lower:
            filters['date'] = 'yesterday'
        elif 'this week' in query_lower:
            filters['date'] = 'this_week'
        elif 'this month' in query_lower:
            filters['date'] = 'this_month'
        elif 'this year' in query_lower:
            filters['date'] = 'this_year'
        
        # Extract type filters
        if any(word in query_lower for word in ['pdf', 'document']):
            filters['type'] = 'document'
        elif any(word in query_lower for word in ['image', 'photo', 'picture']):
            filters['type'] = 'image'
        elif any(word in query_lower for word in ['video', 'movie']):
            filters['type'] = 'video'
        
        # Extract language filters
        if 'english' in query_lower:
            filters['language'] = 'en'
        elif 'spanish' in query_lower:
            filters['language'] = 'es'
        elif 'french' in query_lower:
            filters['language'] = 'fr'
        
        return filters
    
    def _calculate_confidence(self, query: str, intent: str, entities: List[str], keywords: List[str]) -> float:
        """Calculate confidence score for analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on query length
        if len(query) > 10:
            confidence += 0.1
        
        # Boost confidence based on entities found
        if entities:
            confidence += 0.1
        
        # Boost confidence based on keywords found
        if len(keywords) >= 3:
            confidence += 0.1
        
        # Boost confidence for specific intents
        if intent in ['question', 'definition']:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def suggest_improvements(self, query: str) -> List[str]:
        """Suggest improvements for the query."""
        suggestions = []
        
        if len(query) < 3:
            suggestions.append("Query is too short. Try adding more specific terms.")
        
        if not any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            suggestions.append("Consider making your query more specific with question words.")
        
        if len(query.split()) < 2:
            suggestions.append("Try using multiple keywords for better results.")
        
        return suggestions 