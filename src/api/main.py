"""
Main FastAPI application for the distributed indexing system.
"""

import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import requests
import json

from ..config import settings
from ..utils.logging import get_logger
from ..utils.metrics import metrics_collector
from ..ingestion.processor import DocumentProcessor, ImageProcessor, TabularProcessor
from ..ingestion.chunker import TextChunker
from ..ai.gemini_client import GeminiClient, extract_gemini_text
from ..storage.qdrant_client import QdrantClient
from ..models.base import BaseDocument

# Initialize components
logger = get_logger(__name__)
document_processor = DocumentProcessor()
image_processor = ImageProcessor()
tabular_processor = TabularProcessor()
text_chunker = TextChunker()
gemini_client = GeminiClient()
qdrant_client = QdrantClient()

async def determine_collections(document: BaseDocument, ai_metadata: Dict[str, Any]) -> List[str]:
    """
    Use AI to determine which collections this document should be stored in.
    Creates specialized collections based on content analysis.
    """
    try:
        # Base collections based on document type
        base_collections = [f"index_{document.type.value}"]
        
        # Analyze content for specialized collections
        content_preview = document.content[:1000] if document.content else "No content available"
        content_analysis_prompt = f"""
        Analyze this content and determine which specialized collections it should be stored in.
        
        Content Type: {document.type.value}
        Content Preview: {content_preview}...
        
        Available metadata: {ai_metadata}
        
        Consider creating specialized collections for:
        - Content topics (e.g., index_technology, index_finance, index_healthcare)
        - Content formats (e.g., index_reports, index_manuals, index_research)
        - Content domains (e.g., index_business, index_academic, index_creative)
        - Content languages (e.g., index_english, index_spanish, index_french)
        - Content sentiment (e.g., index_positive, index_negative, index_neutral)
        
        Return a JSON list of collection names (without 'index_' prefix):
        ["technology", "reports", "business", "english"]
        
        Guidelines:
        - Create 2-4 specialized collections
        - Use descriptive, lowercase names
        - Consider the main topics and characteristics
        - Don't include the base collection (already handled)
        """
        
        response = await gemini_client.generate_text(content_analysis_prompt, temperature=0.3)
        response_text = extract_gemini_text(response)
        
        # Parse the response
        try:
            if "[" in response_text and "]" in response_text:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                json_str = response_text[start:end]
                specialized_collections = json.loads(json_str)
                
                # Add 'index_' prefix to specialized collections
                specialized_collections = [f"index_{col}" for col in specialized_collections if isinstance(col, str)]
                
                # Combine base and specialized collections
                all_collections = base_collections + specialized_collections
                logger.info(f"AI determined collections: {all_collections}")
                return all_collections
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse collection analysis: {e}")
        
        # Fallback to base collection
        return base_collections
        
    except Exception as e:
        logger.error(f"Collection determination failed: {e}")
        return [f"index_{document.type.value}"]

# Create FastAPI app
app = FastAPI(
    title="Distributed Indexing System",
    description="Scalable, fault-tolerant distributed indexing and search system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=settings.api.cors_methods,
    allow_headers=settings.api.cors_headers,
)

# Request/Response models
class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str
    chunks_created: int
    metadata: Dict[str, Any]
    content: str = ""

class SearchRequest(BaseModel):
    query: str
    index_names: List[str] = []
    limit: int = 10
    score_threshold: float = 0.7
    search_strategy: str = "hybrid"

class SearchResult(BaseModel):
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_index: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    query_analysis: Dict[str, Any]
    reasoning: Dict[str, Any]

class AskRequest(BaseModel):
    question: str
    index_names: List[str] = []
    limit: int = 10
    score_threshold: float = 0.5
    search_strategy: str = "hybrid"
    include_sources: bool = True

class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[SearchResult]
    reasoning: Dict[str, Any]
    query_analysis: Dict[str, Any]

class IndexInfo(BaseModel):
    name: str
    type: str
    size: int
    description: str
    status: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check all components
        qdrant_health = await qdrant_client.get_collection_info("test") if "test" in [c["name"] for c in await qdrant_client.list_collections()] else True
        
        return {
            "status": "healthy",
            "components": {
                "qdrant": "healthy" if qdrant_health else "unhealthy",
                "gemini": "healthy",
                "processors": "healthy"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    chunk_strategy: str = Form("auto")
):
    """
    Upload and process a file for indexing.
    
    This endpoint handles the complete flow:
    1. File validation and processing
    2. Content extraction and chunking
    3. Metadata extraction using Gemini AI
    4. Vector embedding generation
    5. Distributed storage in Qdrant
    """
    try:
        import tempfile
        import os
        
        logger.info(f"Received upload: {file.filename}")
        # Parse metadata
        try:
            file_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            file_metadata = {}
        
        # Save uploaded file temporarily
        filename = file.filename or "unknown_file"
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process file based on type
            file_extension = os.path.splitext(filename)[1].lower()
            logger.info(f"Processing file type: {file_extension}")
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                document = image_processor.process_image(temp_file_path, file_metadata)
            elif file_extension in ['.csv', '.xlsx', '.xls', '.parquet', '.json']:
                document = tabular_processor.process_tabular(temp_file_path, file_metadata)
            else:
                document = document_processor.process_document(temp_file_path, file_metadata)
            logger.info(f"Document processed: {document.id}")
            
            # Extract additional metadata using Gemini
            ai_metadata = {}  # Initialize to empty dict
            if document.content:
                try:
                    ai_metadata = await gemini_client.extract_metadata(
                        document.content, 
                        document.type.value
                    )
                    document.metadata.update(ai_metadata)
                    logger.info(f"AI metadata extracted for document: {document.id}")
                except Exception as e:
                    logger.error(f"AI metadata extraction failed: {e}")
                    # Keep ai_metadata as empty dict
            else:
                logger.warning(f"No content extracted from document: {document.id}")
            
            # Chunk document if needed
            chunked_docs = text_chunker.chunk_document(document, chunk_strategy)
            logger.info(f"Document chunked: {len(chunked_docs)} chunks")
            
            # Generate embeddings for chunks
            chunk_texts = [doc.content for doc in chunked_docs if doc.content]
            embeddings = []
            if document.type.value == "image":
                # Use Gemini text embedding for image caption
                if chunk_texts:
                    try:
                        embeddings = await gemini_client.generate_embeddings(chunk_texts)
                        logger.info(f"Gemini text embedding used for image document: {document.id}")
                    except Exception as e:
                        logger.error(f"Image embedding generation failed: {e}")
                        raise HTTPException(status_code=500, detail=f"Image embedding generation failed: {e}")
                else:
                    logger.warning(f"No caption content to embed for image document: {document.id}")
            elif chunk_texts:
                try:
                    embeddings = await gemini_client.generate_embeddings(chunk_texts)
                    logger.info(f"Embeddings generated for {len(chunk_texts)} chunks")
                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")
            else:
                logger.warning(f"No chunk texts to embed for document: {document.id}")

            if embeddings:
                # Determine collection names using AI analysis
                collection_names = await determine_collections(document, ai_metadata)
                logger.info(f"AI determined collections: {collection_names}")
                
                # Store in multiple collections for better search coverage
                vector_size = 384  # Use 384 for sentence-transformers embeddings
                stored_collections = []
                
                for collection_name in collection_names:
                    try:
                        # Create collection if it doesn't exist
                        await qdrant_client.create_collection(
                            collection_name=collection_name,
                            vector_size=vector_size,
                            shard_number=3,
                            replication_factor=2
                        )
                        logger.info(f"Collection created or exists: {collection_name}")
                        
                        # Store embeddings in this collection
                        await qdrant_client.upsert_vectors(
                            collection_name=collection_name,
                            vectors=embeddings,
                            documents=chunked_docs
                        )
                        stored_collections.append(collection_name)
                        logger.info(f"Document stored in collection: {collection_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to store in collection {collection_name}: {e}")
                        continue
                
                if not stored_collections:
                    # Fallback to default collection
                    default_collection = f"index_{document.type.value}"
                    await qdrant_client.create_collection(
                        collection_name=default_collection,
                        vector_size=vector_size,
                        shard_number=3,
                        replication_factor=2
                    )
                    await qdrant_client.upsert_vectors(
                        collection_name=default_collection,
                        vectors=embeddings,
                        documents=chunked_docs
                    )
                    stored_collections = [default_collection]
                    logger.info(f"Fallback to default collection: {default_collection}")
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                return UploadResponse(
                    document_id=document.id,
                    status="success",
                    message=f"Document processed and stored in {len(stored_collections)} collections",
                    chunks_created=len(chunked_docs),
                    metadata=document.metadata,
                    content=document.content[:500] + "..." if len(document.content) > 500 else document.content
                )
            else:
                raise HTTPException(status_code=500, detail="No embeddings generated")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error("File upload failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search across multiple distributed indexes using Gemini AI for reasoning.
    
    This endpoint demonstrates the full distributed search flow:
    1. Query analysis using Gemini AI to determine relevant indexes
    2. Parallel search across multiple distributed indexes
    3. Result aggregation and ranking
    4. AI-powered reasoning about results
    """
    try:
        # Get available indexes
        collections = await qdrant_client.list_collections()
        available_indexes = [
            {
                "name": col["name"],
                "description": f"Index for {col['name']}",
                "type": "vector",
                "size": col.get("vectors_count", 0)
            }
            for col in collections
        ]
        
        # Use Gemini to analyze query and select indexes
        if not request.index_names:
            query_analysis = await gemini_client.analyze_query(
                request.query, 
                available_indexes
            )
            selected_indexes = query_analysis.get("recommended_indexes", [])
            search_strategy = query_analysis.get("search_strategy", "hybrid")
        else:
            selected_indexes = request.index_names
            search_strategy = request.search_strategy
            query_analysis = {
                "recommended_indexes": selected_indexes,
                "search_strategy": search_strategy,
                "confidence": 0.8,
                "reasoning": "User-specified indexes"
            }
        
        # Generate query embedding
        query_embedding = await gemini_client.generate_embeddings([request.query])
        
        # Search across selected indexes in parallel
        search_tasks = []
        available_collection_names = [col["name"] for col in collections]
        
        for index_name in selected_indexes:
            if index_name in available_collection_names:
                task = qdrant_client.search_vectors(
                    collection_name=index_name,
                    query_vector=query_embedding[0],
                    limit=request.limit,
                    score_threshold=request.score_threshold
                )
                search_tasks.append((index_name, task))
            else:
                logger.warning(f"Requested collection '{index_name}' does not exist, skipping")
        
        # If no valid collections found, try to find similar ones
        if not search_tasks and selected_indexes:
            logger.info("No requested collections found, searching in available collections")
            # Search in all available collections as fallback
            for col in collections:
                task = qdrant_client.search_vectors(
                    collection_name=col["name"],
                    query_vector=query_embedding[0],
                    limit=request.limit,
                    score_threshold=request.score_threshold
                )
                search_tasks.append((col["name"], task))
        
        # Execute searches in parallel
        search_results = []
        for index_name, task in search_tasks:
            try:
                results = await task
                for result in results:
                    result["source_index"] = index_name
                    search_results.append(result)
            except Exception as e:
                logger.warning(f"Search failed for index {index_name}", error=str(e))
        
        # Sort results by score
        search_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit results
        search_results = search_results[:request.limit]
        
        # Use Gemini to reason about results
        reasoning = await gemini_client.reason_about_results(
            request.query, 
            search_results
        )
        
        # Format response
        formatted_results = [
            SearchResult(
                document_id=result["document_id"],
                content=result["content"],
                score=result["score"],
                metadata=result["metadata"],
                source_index=result["source_index"]
            )
            for result in search_results
        ]
        
        return SearchResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_analysis=query_analysis,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error("Search failed", error=str(e), query=request.query)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Ask endpoint - AI-powered question answering
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an AI-generated answer based on indexed content.
    
    This endpoint:
    1. Searches for relevant content across indexes
    2. Uses Gemini AI to generate a comprehensive answer
    3. Provides reasoning and confidence scores
    4. Includes source documents for transparency
    """
    try:
        # First, perform a search to find relevant content
        search_request = SearchRequest(
            query=request.question,
            index_names=request.index_names,
            limit=request.limit,
            score_threshold=request.score_threshold,
            search_strategy=request.search_strategy
        )
        
        # Get search results
        search_response = await search_documents(search_request)
        
        if not search_response.results:
            # No relevant content found
            return AskResponse(
                answer="I couldn't find any relevant information to answer your question. Please try rephrasing or uploading more relevant documents.",
                confidence=0.0,
                sources=[],
                reasoning={
                    "answer": "No relevant content found",
                    "confidence": 0.0,
                    "result_assessment": [],
                    "missing_info": ["No relevant documents found"],
                    "follow_up_queries": ["Try uploading relevant documents", "Rephrase your question"]
                },
                query_analysis=search_response.query_analysis
            )
        
        # Use Gemini to generate a comprehensive answer
        try:
            # Create a prompt for answer generation
            sources_text = "\n\n".join([
                f"Source {i+1} (Score: {result.score:.3f}):\n{result.content[:1000]}"
                for i, result in enumerate(search_response.results[:5])  # Top 5 sources
            ])
            
            answer_prompt = f"""
            Based on the following sources, please provide a comprehensive answer to the user's question.
            
            Question: "{request.question}"
            
            Sources:
            {sources_text}
            
            Please provide:
            1. A direct, comprehensive answer to the question
            2. Your confidence level (0-1) in the answer
            3. Reasoning for your answer
            4. Any limitations or uncertainties
            
            Format your response as JSON:
            {{
                "answer": "Your comprehensive answer here",
                "confidence": 0.85,
                "reasoning": "Explanation of how you arrived at this answer",
                "limitations": ["Any limitations or uncertainties"],
                "sources_used": [0, 1, 2]  // Indices of most relevant sources
            }}
            
            Guidelines:
            - Be direct and comprehensive in your answer
            - If the sources don't fully answer the question, acknowledge this
            - If there are conflicting information in sources, mention this
            - Provide specific details from the sources when relevant
            - Be honest about confidence levels and limitations
            """
            
            # Generate answer using Gemini
            answer_response = await gemini_client.generate_text(answer_prompt, temperature=0.3)
            
            # Parse the answer
            try:
                answer_text = extract_gemini_text(answer_response)
                # Extract JSON from response
                if "{" in answer_text and "}" in answer_text:
                    start = answer_text.find("{")
                    end = answer_text.rfind("}") + 1
                    json_str = answer_text[start:end]
                    answer_data = json.loads(json_str)
                    
                    answer = answer_data.get("answer", "I couldn't generate a proper answer.")
                    confidence = answer_data.get("confidence", 0.5)
                    reasoning_text = answer_data.get("reasoning", "Analysis completed")
                    limitations = answer_data.get("limitations", [])
                    sources_used = answer_data.get("sources_used", [])
                    
                else:
                    # Fallback if JSON parsing fails
                    answer = answer_text
                    confidence = 0.6
                    reasoning_text = "Answer generated from search results"
                    limitations = ["JSON parsing failed"]
                    sources_used = list(range(min(3, len(search_response.results))))
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Answer JSON parsing failed: {e}")
                answer = extract_gemini_text(answer_response)
                confidence = 0.5
                reasoning_text = "Answer generated from search results"
                limitations = ["JSON parsing failed"]
                sources_used = list(range(min(3, len(search_response.results))))
            
            # Prepare sources for response
            sources = search_response.results if request.include_sources else []
            
            # Create reasoning object
            reasoning = {
                "answer": reasoning_text,
                "confidence": confidence,
                "limitations": limitations,
                "sources_used": sources_used,
                "result_assessment": search_response.reasoning.get("result_assessment", []),
                "missing_info": search_response.reasoning.get("missing_info", []),
                "follow_up_queries": search_response.reasoning.get("follow_up_queries", [])
            }
            
            return AskResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                reasoning=reasoning,
                query_analysis=search_response.query_analysis
            )
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Fallback response
            return AskResponse(
                answer=f"I found some relevant information but couldn't generate a proper answer. Here are the search results: {search_response.results[0].content[:200]}...",
                confidence=0.3,
                sources=search_response.results if request.include_sources else [],
                reasoning={
                    "answer": "Answer generation failed, providing search results",
                    "confidence": 0.3,
                    "limitations": ["AI answer generation failed"],
                    "sources_used": [0],
                    "result_assessment": [],
                    "missing_info": [],
                    "follow_up_queries": []
                },
                query_analysis=search_response.query_analysis
            )
        
    except Exception as e:
        logger.error("Ask question failed", error=str(e), question=request.question)
        raise HTTPException(status_code=500, detail=f"Ask question failed: {str(e)}")

# Get indexes endpoint
@app.get("/indexes", response_model=List[IndexInfo])
async def get_indexes():
    """Get information about all available indexes."""
    try:
        collections = await qdrant_client.list_collections()
        
        indexes = []
        for col in collections:
            # Get detailed collection info to get accurate count
            try:
                collection_name = col["name"]
                url = f"http://{settings.database.qdrant_host}:{settings.database.qdrant_port}/collections/{collection_name}"
                response = requests.get(url)
                response.raise_for_status()
                collection_info = response.json()
                
                # Extract vectors count from the proper location in response
                vectors_count = 0
                if "result" in collection_info:
                    result = collection_info["result"]
                    # Try different possible field names for vectors count
                    vectors_count = (
                        result.get("vectors_count") or 
                        result.get("points_count") or 
                        result.get("count") or 
                        0
                    )
                
                index_info = IndexInfo(
                    name=collection_name,
                    type="vector",
                    size=vectors_count,
                    description=f"Vector index for {collection_name}",
                    status=col.get("status", "green")
                )
                indexes.append(index_info)
                
            except Exception as e:
                logger.warning(f"Failed to get detailed info for collection {col['name']}: {e}")
                # Fallback to basic info
                index_info = IndexInfo(
                    name=col["name"],
                    type="vector",
                    size=0,  # Unknown count
                    description=f"Vector index for {col['name']}",
                    status=col.get("status", "unknown")
                )
                indexes.append(index_info)
        
        return indexes
        
    except Exception as e:
        logger.error("Failed to get indexes", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get indexes: {str(e)}")

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        return metrics_collector.get_metrics()
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting distributed indexing system")
    
    # Initialize default collections
    try:
        default_collections = ["index_document", "index_image", "index_tabular"]
        for collection_name in default_collections:
            try:
                await qdrant_client.create_collection(
                    collection_name=collection_name,
                    vector_size=384,  # Use 384 for sentence-transformers embeddings
                    shard_number=3,
                    replication_factor=2
                )
                logger.info(f"Created default collection: {collection_name}")
            except Exception:
                # Collection might already exist
                pass
    except Exception as e:
        logger.warning("Failed to initialize default collections", error=str(e))

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down distributed indexing system")

@app.get("/documents")
async def list_documents(
    collection: str = Query(..., description="Collection name"),
    limit: int = Query(10, description="Number of documents to return")
):
    """
    List documents in a Qdrant collection.
    """
    try:
        # Use Qdrant REST API directly since scroll is not implemented in the wrapper
        url = f"http://{settings.database.qdrant_host}:{settings.database.qdrant_port}/collections/{collection}/points/scroll"
        response = requests.post(url, json={"limit": limit})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/delete_index")
async def delete_index(collection: str = Query(..., description="Collection name to delete")):
    """
    Delete a Qdrant collection (index) by name.
    """
    try:
        await qdrant_client.delete_collection(collection)
        logger.info(f"Collection deleted: {collection}")
        return {"status": "ok", "message": f"Collection '{collection}' deleted."}
    except Exception as e:
        logger.error(f"Failed to delete collection {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

# Document inspection endpoints
@app.get("/document/{document_id}")
async def get_document_by_id(
    document_id: str = Path(..., description="Document ID to retrieve"),
    collection: str = Query(..., description="Collection name")
):
    """
    Get a specific document by its ID from a collection.
    """
    try:
        # Use Qdrant REST API to get specific point
        url = f"http://{settings.database.qdrant_host}:{settings.database.qdrant_port}/collections/{collection}/points/{document_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get document {document_id} from {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.get("/collection_stats/{collection}")
async def get_collection_statistics(collection: str):
    """
    Get detailed statistics about a collection.
    """
    try:
        # Get collection info
        url = f"http://{settings.database.qdrant_host}:{settings.database.qdrant_port}/collections/{collection}"
        response = requests.get(url)
        response.raise_for_status()
        collection_info = response.json()
        
        # Get some sample documents
        scroll_url = f"http://{settings.database.qdrant_host}:{settings.database.qdrant_port}/collections/{collection}/points/scroll"
        scroll_response = requests.post(scroll_url, json={"limit": 5})
        scroll_response.raise_for_status()
        sample_docs = scroll_response.json()
        
        return {
            "collection_name": collection,
            "info": collection_info.get("result", {}),
            "sample_documents": sample_docs.get("result", {}).get("points", []),
            "total_documents": collection_info.get("result", {}).get("vectors_count", 0)
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats for {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {str(e)}")

@app.post("/search_by_metadata")
async def search_by_metadata(
    collection: str = Query(..., description="Collection name"),
    metadata_filter: str = Query(..., description="JSON string with metadata filters"),
    limit: int = Query(10, description="Number of documents to return")
):
    """
    Search documents by metadata filters.
    
    Example metadata_filter: '{"file_path": {"$contains": "pdf"}}'
    """
    try:
        # Parse metadata filter
        try:
            filter_data = json.loads(metadata_filter)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in metadata_filter")
        
        # Use Qdrant REST API with filter
        url = f"http://{settings.database.qdrant_host}:{settings.database.qdrant_port}/collections/{collection}/points/scroll"
        payload = {
            "limit": limit,
            "filter": filter_data
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to search by metadata in {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search by metadata: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers
    ) 