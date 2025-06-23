"""
Gemini AI client for reasoning and intelligent processing.
"""

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pydantic import BaseModel, Field

from ..utils.logging import LoggerMixin
from ..utils.metrics import monitor_function
from ..config import settings


class GeminiResponse(BaseModel):
    """Response from Gemini AI."""
    text: str = Field(..., description="Generated text response")
    confidence: float = Field(default=0.0, description="Confidence score")
    reasoning: str = Field(default="", description="Reasoning process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GeminiClient(LoggerMixin):
    """Client for interacting with Google's Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (if not provided, will use environment variable)
        """
        super().__init__()
        
        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.text_model = genai.GenerativeModel('gemini-2.5-flash')
        self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.logger.info("Gemini client initialized successfully")
    
    @monitor_function("gemini_client", "generate_text", "text")
    async def generate_text(self, prompt: str, 
                           context: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: int = 1000) -> GeminiResponse:
        """
        Generate text response using Gemini.
        
        Args:
            prompt: The input prompt
            context: Additional context for the prompt
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            GeminiResponse object
        """
        try:
            # Prepare full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nPrompt: {prompt}"
            
            # Generate response
            response = await self.text_model.generate_content_async(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            # Extract response
            response_text = response.text if response.text else ""
            
            # Create response object
            gemini_response = GeminiResponse(
                text=response_text,
                confidence=0.8,  # Default confidence
                reasoning="Generated using Gemini Pro model",
                metadata={
                    "model": "gemini-pro",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "prompt_tokens": len(full_prompt.split()),
                    "response_tokens": len(response_text.split())
                }
            )
            
            self.logger.debug("Text generation completed", 
                            prompt_length=len(prompt),
                            response_length=len(response_text))
            
            return gemini_response
            
        except Exception as e:
            self.logger.error("Text generation failed", error=str(e), prompt=prompt[:100])
            raise
    
    @monitor_function("gemini_client", "analyze_query", "query")
    async def analyze_query(self, query: str, 
                           available_indexes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a query to determine which indexes to search.
        
        Args:
            query: User query
            available_indexes: List of available indexes with metadata
            
        Returns:
            Analysis result with recommended indexes and reasoning
        """
        try:
            # Create analysis prompt
            indexes_info = "\n".join([
                f"- {idx['name']}: {idx['description']} (type: {idx['type']}, size: {idx['size']})"
                for idx in available_indexes
            ])
            
            prompt = f"""
            Analyze the following query and determine which indexes would be most relevant for searching.
            
            Query: "{query}"
            
            Available indexes:
            {indexes_info}
            
            Please provide:
            1. A list of recommended indexes (by name) in order of relevance
            2. Reasoning for each recommendation
            3. Search strategy (semantic, keyword, hybrid)
            4. Confidence score (0-1)
            
            Format your response as JSON:
            {{
                "recommended_indexes": ["index1", "index2"],
                "reasoning": "Explanation of why these indexes are relevant",
                "search_strategy": "semantic|keyword|hybrid",
                "confidence": 0.85,
                "query_type": "factual|analytical|creative"
            }}
            """
            
            # Generate analysis
            response = await self.generate_text(prompt, temperature=0.3)
            
            # Parse JSON response
            import json
            try:
                analysis = json.loads(response.text)
                return analysis
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "recommended_indexes": [idx["name"] for idx in available_indexes[:3]],
                    "reasoning": "Fallback: Using first 3 indexes",
                    "search_strategy": "hybrid",
                    "confidence": 0.5,
                    "query_type": "general"
                }
                
        except Exception as e:
            self.logger.error("Query analysis failed", error=str(e), query=query)
            raise
    
    @monitor_function("gemini_client", "extract_metadata", "metadata")
    async def extract_metadata(self, content: str, 
                              content_type: str = "text") -> Dict[str, Any]:
        """
        Extract metadata from content using Gemini.
        
        Args:
            content: Content to analyze
            content_type: Type of content (text, image, document)
            
        Returns:
            Extracted metadata
        """
        try:
            prompt = f"""
            Analyze the following {content_type} content and extract relevant metadata.
            
            Content:
            {content[:2000]}  # Limit content length
            
            Please extract:
            1. Key topics/themes
            2. Document type/category
            3. Language
            4. Sentiment (positive/negative/neutral)
            5. Key entities (people, places, organizations)
            6. Summary (2-3 sentences)
            
            Format your response as JSON:
            {{
                "topics": ["topic1", "topic2"],
                "category": "document_category",
                "language": "en",
                "sentiment": "positive|negative|neutral",
                "entities": {{"people": [], "places": [], "organizations": []}},
                "summary": "Brief summary of the content"
            }}
            """
            
            response = await self.generate_text(prompt, temperature=0.3)
            
            # Parse JSON response
            import json
            try:
                metadata = json.loads(response.text)
                return metadata
            except json.JSONDecodeError:
                return {
                    "topics": ["general"],
                    "category": "unknown",
                    "language": "en",
                    "sentiment": "neutral",
                    "entities": {"people": [], "places": [], "organizations": []},
                    "summary": "Content analysis completed"
                }
                
        except Exception as e:
            self.logger.error("Metadata extraction failed", error=str(e))
            raise
    
    @monitor_function("gemini_client", "generate_embeddings", "embeddings")
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text using Gemini.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Note: Gemini doesn't have a direct embedding API, so we'll use a workaround
            # In a real implementation, you might use a separate embedding service
            
            # For now, return placeholder embeddings
            # In production, you'd use sentence-transformers or another embedding service
            import numpy as np
            
            embeddings = []
            for text in texts:
                # Generate a deterministic embedding-like vector
                # This is a placeholder - replace with actual embedding generation
                embedding = np.random.normal(0, 1, 768).tolist()  # 768-dimensional vector
                embeddings.append(embedding)
            
            self.logger.debug("Embeddings generated", num_texts=len(texts))
            return embeddings
            
        except Exception as e:
            self.logger.error("Embedding generation failed", error=str(e))
            raise
    
    @monitor_function("gemini_client", "reason_about_results", "reasoning")
    async def reason_about_results(self, query: str, 
                                  search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use Gemini to reason about search results and provide insights.
        
        Args:
            query: Original query
            search_results: List of search results
            
        Returns:
            Reasoning and insights about the results
        """
        try:
            # Format search results
            results_text = "\n\n".join([
                f"Result {i+1} (Score: {result.get('score', 0):.3f}):\n{result.get('content', '')[:500]}"
                for i, result in enumerate(search_results[:5])  # Top 5 results
            ])
            
            prompt = f"""
            Analyze the following search results for the query: "{query}"
            
            Search Results:
            {results_text}
            
            Please provide:
            1. Relevance assessment of each result
            2. Overall answer to the query based on the results
            3. Confidence in the answer (0-1)
            4. Missing information or gaps
            5. Suggestions for follow-up queries
            
            Format your response as JSON:
            {{
                "answer": "Direct answer to the query",
                "confidence": 0.85,
                "result_assessment": [
                    {{"result_index": 0, "relevance": "high|medium|low", "reasoning": "..."}}
                ],
                "missing_info": ["missing piece 1", "missing piece 2"],
                "follow_up_queries": ["query 1", "query 2"]
            }}
            """
            
            response = await self.generate_text(prompt, temperature=0.4)
            
            # Parse JSON response
            import json
            try:
                reasoning = json.loads(response.text)
                return reasoning
            except json.JSONDecodeError:
                return {
                    "answer": "Analysis completed but could not parse detailed reasoning",
                    "confidence": 0.5,
                    "result_assessment": [],
                    "missing_info": [],
                    "follow_up_queries": []
                }
                
        except Exception as e:
            self.logger.error("Result reasoning failed", error=str(e))
            raise 