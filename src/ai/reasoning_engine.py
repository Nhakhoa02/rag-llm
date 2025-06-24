"""
AI reasoning engine for intelligent document processing and analysis.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..models.base import BaseDocument
from ..utils.logging import get_logger
from .gemini_client import GeminiClient, extract_gemini_text


class ReasoningConfig(BaseModel):
    """Reasoning engine configuration."""
    
    enable_reasoning: bool = Field(default=True, description="Enable AI reasoning")
    reasoning_model: str = Field(default="gemini-2.5-pro", description="Reasoning model to use")
    max_tokens: int = Field(default=1000, description="Maximum tokens for reasoning")
    temperature: float = Field(default=0.3, description="Reasoning temperature")
    enable_explanation: bool = Field(default=True, description="Enable reasoning explanations")
    
    class Config:
        extra = "allow"


class ReasoningResult(BaseModel):
    """Result of AI reasoning operation."""
    
    success: bool = Field(..., description="Whether reasoning was successful")
    reasoning: str = Field(default="", description="Reasoning explanation")
    conclusion: str = Field(default="", description="Reasoning conclusion")
    confidence: float = Field(default=0.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"


class ReasoningEngine:
    """AI-powered reasoning engine for document analysis."""
    
    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.logger = get_logger(__name__)
        self.gemini_client = GeminiClient() if self.config.enable_reasoning else None
    
    async def analyze_document(self, document: BaseDocument) -> ReasoningResult:
        """
        Analyze document using AI reasoning.
        
        Args:
            document: Document to analyze
            
        Returns:
            Reasoning result with analysis
        """
        try:
            if not self.config.enable_reasoning or not self.gemini_client:
                return ReasoningResult(
                    success=False,
                    reasoning="Reasoning engine is disabled",
                    conclusion="No analysis performed"
                )
            
            if not document.content:
                return ReasoningResult(
                    success=False,
                    reasoning="Document has no content to analyze",
                    conclusion="Cannot analyze empty document"
                )
            
            # Create reasoning prompt
            prompt = self._create_analysis_prompt(document)
            
            # Get AI response
            response = await self.gemini_client.generate_text(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            if response:
                # Extract text from response using proper method
                response_text = extract_gemini_text(response)
                if response_text:
                    result = self._parse_reasoning_response(response_text)
                else:
                    result = ReasoningResult(
                        success=False,
                        reasoning="No response from AI model",
                        conclusion="Analysis could not be completed"
                    )
            else:
                result = ReasoningResult(
                    success=False,
                    reasoning="No response from AI model",
                    conclusion="Analysis could not be completed"
                )
            
            self.logger.info(f"Document analysis completed for: {document.id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Document analysis failed for {document.id}: {e}")
            return ReasoningResult(
                success=False,
                reasoning=f"Analysis failed: {str(e)}",
                conclusion="Analysis could not be completed"
            )
    
    async def reason_about_query(self, query: str, context: List[BaseDocument]) -> ReasoningResult:
        """
        Use AI reasoning to analyze a query in the context of documents.
        
        Args:
            query: Search query
            context: Relevant documents for context
            
        Returns:
            Reasoning result with query analysis
        """
        try:
            if not self.config.enable_reasoning or not self.gemini_client:
                return ReasoningResult(
                    success=False,
                    reasoning="Reasoning engine is disabled",
                    conclusion="No query analysis performed"
                )
            
            # Create reasoning prompt
            prompt = self._create_query_prompt(query, context)
            
            # Get AI response
            response = await self.gemini_client.generate_text(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            if response:
                # Extract text from response using proper method
                response_text = extract_gemini_text(response)
                if response_text:
                    result = self._parse_reasoning_response(response_text)
                else:
                    result = ReasoningResult(
                        success=False,
                        reasoning="No response from AI model",
                        conclusion="Query analysis could not be completed"
                    )
            else:
                result = ReasoningResult(
                    success=False,
                    reasoning="No response from AI model",
                    conclusion="Query analysis could not be completed"
                )
            
            self.logger.info(f"Query reasoning completed for: {query}")
            return result
            
        except Exception as e:
            self.logger.error(f"Query reasoning failed for {query}: {e}")
            return ReasoningResult(
                success=False,
                reasoning=f"Query reasoning failed: {str(e)}",
                conclusion="Query analysis could not be completed"
            )
    
    async def generate_hypothesis(self, documents: List[BaseDocument], topic: str) -> ReasoningResult:
        """
        Generate hypotheses based on document analysis.
        
        Args:
            documents: Documents to analyze
            topic: Topic for hypothesis generation
            
        Returns:
            Reasoning result with generated hypothesis
        """
        try:
            if not self.config.enable_reasoning or not self.gemini_client:
                return ReasoningResult(
                    success=False,
                    reasoning="Reasoning engine is disabled",
                    conclusion="No hypothesis generated"
                )
            
            # Create hypothesis prompt
            prompt = self._create_hypothesis_prompt(documents, topic)
            
            # Get AI response
            response = await self.gemini_client.generate_text(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            if response:
                # Extract text from response using proper method
                response_text = extract_gemini_text(response)
                if response_text:
                    result = self._parse_reasoning_response(response_text)
                else:
                    result = ReasoningResult(
                        success=False,
                        reasoning="No response from AI model",
                        conclusion="Hypothesis could not be generated"
                    )
            else:
                result = ReasoningResult(
                    success=False,
                    reasoning="No response from AI model",
                    conclusion="Hypothesis could not be generated"
                )
            
            self.logger.info(f"Hypothesis generation completed for topic: {topic}")
            return result
            
        except Exception as e:
            self.logger.error(f"Hypothesis generation failed for topic {topic}: {e}")
            return ReasoningResult(
                success=False,
                reasoning=f"Hypothesis generation failed: {str(e)}",
                conclusion="Hypothesis could not be generated"
            )
    
    def _create_analysis_prompt(self, document: BaseDocument) -> str:
        """Create prompt for document analysis."""
        prompt = f"""
        Analyze the following document and provide insights:
        
        Document Type: {document.type.value}
        Document ID: {document.id}
        
        Content:
        {document.content[:2000]}  # Limit content length
        
        Please provide:
        1. Main topics and themes
        2. Key insights and conclusions
        3. Document quality assessment
        4. Suggested tags or categories
        5. Confidence level in your analysis
        
        Format your response as:
        REASONING: [Your detailed reasoning]
        CONCLUSION: [Your main conclusion]
        CONFIDENCE: [0.0-1.0]
        """
        return prompt
    
    def _create_query_prompt(self, query: str, context: List[BaseDocument]) -> str:
        """Create prompt for query reasoning."""
        context_text = "\n\n".join([doc.content[:500] for doc in context[:3]])  # Limit context
        
        prompt = f"""
        Analyze the following query in the context of these documents:
        
        Query: {query}
        
        Context Documents:
        {context_text}
        
        Please provide:
        1. Query intent analysis
        2. Relevance to context documents
        3. Suggested search strategies
        4. Potential missing information
        5. Confidence in understanding
        
        Format your response as:
        REASONING: [Your detailed reasoning]
        CONCLUSION: [Your main conclusion]
        CONFIDENCE: [0.0-1.0]
        """
        return prompt
    
    def _create_hypothesis_prompt(self, documents: List[BaseDocument], topic: str) -> str:
        """Create prompt for hypothesis generation."""
        content_summary = "\n".join([f"- {doc.content[:200]}..." for doc in documents[:5]])
        
        prompt = f"""
        Based on the following documents, generate hypotheses about: {topic}
        
        Document Summaries:
        {content_summary}
        
        Please provide:
        1. Key observations from the documents
        2. Generated hypotheses about the topic
        3. Supporting evidence for each hypothesis
        4. Confidence levels for each hypothesis
        5. Areas needing further investigation
        
        Format your response as:
        REASONING: [Your detailed reasoning]
        CONCLUSION: [Your main hypotheses]
        CONFIDENCE: [0.0-1.0]
        """
        return prompt
    
    def _parse_reasoning_response(self, response: str) -> ReasoningResult:
        """Parse AI response into structured result."""
        try:
            # Extract sections from response
            reasoning = ""
            conclusion = ""
            confidence = 0.5  # Default confidence
            
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('REASONING:'):
                    current_section = 'reasoning'
                    reasoning = line.replace('REASONING:', '').strip()
                elif line.startswith('CONCLUSION:'):
                    current_section = 'conclusion'
                    conclusion = line.replace('CONCLUSION:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence_str = line.replace('CONFIDENCE:', '').strip()
                        confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    except ValueError:
                        confidence = 0.5
                elif current_section == 'reasoning' and line:
                    reasoning += " " + line
                elif current_section == 'conclusion' and line:
                    conclusion += " " + line
            
            return ReasoningResult(
                success=True,
                reasoning=reasoning,
                conclusion=conclusion,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse reasoning response: {e}")
            return ReasoningResult(
                success=True,
                reasoning=response,
                conclusion="Analysis completed",
                confidence=0.5
            ) 