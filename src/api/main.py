"""
Main FastAPI application for the distributed indexing system.
"""

import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import requests

from ..config import settings
from ..utils.logging import get_logger
from ..utils.metrics import metrics_collector
from ..ingestion.processor import DocumentProcessor, ImageProcessor, TabularProcessor
from ..ingestion.chunker import TextChunker
from ..ai.gemini_client import GeminiClient
from ..storage.qdrant_client import QdrantClient

# Initialize components
logger = get_logger(__name__)
document_processor = DocumentProcessor()
image_processor = ImageProcessor()
tabular_processor = TabularProcessor()
text_chunker = TextChunker()
gemini_client = GeminiClient()
qdrant_client = QdrantClient()

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
        import json
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
                # Store in Qdrant
                collection_name = f"index_{document.type.value}"
                vector_size = 384  # Use 384 for sentence-transformers embeddings
                try:
                    await qdrant_client.create_collection(
                        collection_name=collection_name,
                        vector_size=vector_size,
                        shard_number=3,  # Distribute across 3 shards
                        replication_factor=2  # 2 replicas for fault tolerance
                    )
                    logger.info(f"Collection created or exists: {collection_name} (vector_size={vector_size})")
                except Exception as e:
                    logger.warning(f"Collection creation failed or already exists: {e}")
                try:
                    await qdrant_client.upsert_vectors(
                        collection_name=collection_name,
                        vectors=embeddings,
                        documents=chunked_docs
                    )
                    logger.info(f"Vectors upserted to Qdrant: {collection_name}")
                except Exception as e:
                    logger.error(f"Qdrant upsert failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Qdrant upsert failed: {e}")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return UploadResponse(
                document_id=document.id,
                status="success",
                message="File processed and indexed successfully",
                chunks_created=len(chunked_docs),
                metadata=document.metadata,
                content=document.content or ""
            ), {"content": document.content}
            
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
        for index_name in selected_indexes:
            if index_name in [col["name"] for col in collections]:
                task = qdrant_client.search_vectors(
                    collection_name=index_name,
                    query_vector=query_embedding[0],
                    limit=request.limit,
                    score_threshold=request.score_threshold
                )
                search_tasks.append((index_name, task))
        
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

# Get indexes endpoint
@app.get("/indexes", response_model=List[IndexInfo])
async def get_indexes():
    """Get information about all available indexes."""
    try:
        collections = await qdrant_client.list_collections()
        
        indexes = []
        for col in collections:
            index_info = IndexInfo(
                name=col["name"],
                type="vector",
                size=col.get("vectors_count") or 0,
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

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers
    ) 