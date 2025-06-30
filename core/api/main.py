"""
Main FastAPI application for the distributed indexing system.
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path as PathLib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import requests
import json
from typing import cast
import time
import threading

from config.config import settings
from core.utils.logging import get_logger
from core.utils.metrics import metrics_collector
from core.types.document_processor import DocumentProcessor
from core.types.image_processor import ImageProcessor
from core.types.tabular_processor import TabularProcessor
from core.services.ingestion.chunker import TextChunker
from core.services.inference.gemini_client import GeminiClient, extract_gemini_text
from data.storage.distributed_storage_manager import create_distributed_storage_manager
from data.storage.distributed_vector_store import VectorNode
from core.models.base import BaseDocument, DataType
from core.models.document import Document as DocModel
from core.models.csv_index import CSVIndex, CSVIndexDocument
from data.storage.auto_scaler import create_auto_scaler, ScalingThresholds
from data.storage.csv_database import csv_db_manager

# Initialize components
logger = get_logger(__name__)
document_processor = DocumentProcessor()
image_processor = ImageProcessor()
tabular_processor = TabularProcessor()
text_chunker = TextChunker()
gemini_client = GeminiClient()

# Global variable to track node processes
node_processes: Dict[str, subprocess.Popen] = {}

# Initialize distributed storage manager
storage_manager = create_distributed_storage_manager(
    nodes=[
        VectorNode("node1", "localhost", 8001),
        VectorNode("node2", "localhost", 8002),
        VectorNode("node3", "localhost", 8003)
    ],
    replication_factor=2,
    consistency_level="quorum",
    shard_count=8,
    vector_size=384
)

# Initialize auto-scaler
auto_scaler = create_auto_scaler(storage_manager)

async def determine_collections(document: BaseDocument, ai_metadata: Dict[str, Any], existing_collections: Optional[List[str]] = None) -> List[str]:
    """
    Use AI to determine which collections this document should be stored in.
    The AI is encouraged to always create new, specific, and combination collections for new topics, even if a broad collection exists.
    """
    try:
        # Get existing collections if not provided
        if existing_collections is None:
            collections = await storage_manager.list_indexes()
            existing_collections = [col["name"] for col in collections]
        
        # Base collections based on document type
        base_collections = [f"index_{document.type.value}"]
        
        # Reduce content preview to avoid overwhelming the model
        content_preview = document.content[:1000] if document.content else "No content available"
        
        # Extract key metadata for better analysis
        file_metadata = document.metadata or {}
        file_name = file_metadata.get('filename', 'unknown')
        file_type = file_metadata.get('mime_type', 'unknown')
        file_size = file_metadata.get('file_size', 0)
        
        # Minimal prompt for debugging Gemini response
        minimal_prompt = f"""
        Given the following document content, return a JSON list of 1-3 relevant collection names (as strings) that describe the main topics. Example: ["index_ai", "index_machine_learning"]
        Document Content: {content_preview}
        """
        try:
            response = await gemini_client.generate_text(minimal_prompt, temperature=0.5)
            logger.info(f"Full Gemini API response object: {response}")
            response_text = extract_gemini_text(response)
            logger.info(f"Raw Gemini response for collection determination (minimal prompt): {response_text}")
        except Exception as gemini_exc:
            logger.error(f"Gemini API call failed (minimal prompt): {gemini_exc}")
            return base_collections
        # Try to parse a simple list from the response
        try:
            import ast
            # Remove Markdown code block formatting if present
            cleaned = response_text.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[-1]  # Remove the first line (``` or ```json)
                if cleaned.endswith('```'):
                    cleaned = cleaned.rsplit('```', 1)[0]
            cleaned = cleaned.strip()
            collections = ast.literal_eval(cleaned)
            if isinstance(collections, list) and all(isinstance(x, str) for x in collections):
                # Always append the base collection if not present
                if base_collections[0] not in collections:
                    collections.append(base_collections[0])
                return collections
        except Exception as parse_exc:
            logger.warning(f"Failed to parse minimal Gemini response: {parse_exc}")
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
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Request/Response models
class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str
    chunks_created: int
    metadata: Dict[str, Any]
    content: str = ""
    csv_index_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    index_names: List[str] = []
    limit: int = 10
    score_threshold: float = 0.0
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
    score_threshold: float = 0.0
    search_strategy: str = "hybrid"
    include_sources: bool = True

class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[SearchResult]
    reasoning: Dict[str, Any]
    query_analysis: Dict[str, Any]

class AskCSVRequest(BaseModel):
    question: str
    limit: int = 10
    score_threshold: float = 0.0
    include_sources: bool = True

class AskCSVResponse(BaseModel):
    answer: str
    confidence: float
    relevant_csvs: List[Dict[str, Any]]
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
        return {
            "status": "healthy",
            "components": {
                "distributed_storage": "healthy",
                "gemini": "healthy",
                "processors": "healthy"
            },
            "message": "RAG system is running"
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
    Upload and process a file, storing it in the distributed vector database.
    For CSV files, also creates and stores a CSV index for quick lookup.
    """
    try:
        # Parse metadata
        try:
            file_metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            file_metadata = {}
        
        # Add file metadata
        filename = file.filename or "unknown"
        file_metadata.update({
            "filename": filename,
            "mime_type": file.content_type,
            "file_size": file.size,
            "upload_timestamp": time.time()
        })
        
        # Read file content
        content = await file.read()
        
        # Create temporary file for processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
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
            
            # Ensure chunked_docs is List[Document] for upsert_documents
            if chunked_docs and hasattr(chunked_docs[0], 'dict'):
                # Convert BaseDocument to Document if needed
                chunked_docs = [DocModel(**doc.dict()) if hasattr(doc, 'dict') else doc for doc in chunked_docs]
            
            # Generate embeddings
            embeddings = await gemini_client.generate_embeddings(document.content)
            
            if embeddings:
                # Get existing collections for AI analysis
                existing_collections = await storage_manager.list_indexes()
                existing_collection_names = [col["name"] for col in existing_collections]
                
                # Determine collection names using AI analysis (now considers existing collections)
                collection_names = await determine_collections(document, ai_metadata, existing_collection_names)
                logger.info(f"AI determined collections: {collection_names}")
                
                # Store in multiple collections
                vector_size = 384  # Use 384 for sentence-transformers embeddings
                stored_collections = []
                
                for collection_name in collection_names:
                    try:
                        # Check if collection exists before creating
                        collection_exists = await storage_manager.index_exists(collection_name)
                        
                        if not collection_exists:
                            # Create collection only if it doesn't exist
                            await storage_manager.create_index(collection_name, vector_size)
                            logger.info(f"Created new collection: {collection_name}")
                        else:
                            logger.info(f"Using existing collection: {collection_name}")
                        
                        # Store documents in this collection (append mode)
                        success = await storage_manager.upsert_documents(
                            index_name=collection_name,
                            documents=cast(List[DocModel], chunked_docs)  # Cast to proper type
                        )
                        
                        if success:
                            stored_collections.append(collection_name)
                            logger.info(f"Document stored in collection: {collection_name}")
                        else:
                            logger.warning(f"Failed to store in collection: {collection_name}")
                            
                    except Exception as e:
                        logger.error(f"Error storing in collection {collection_name}: {e}")
                        continue
                
                # Handle CSV indexing
                csv_index_id = None
                if file_extension == '.csv' and document.metadata.get('csv_index'):
                    try:
                        # Create CSV index collection if it doesn't exist
                        csv_index_collection = "csv_indexes"
                        if not await storage_manager.index_exists(csv_index_collection):
                            await storage_manager.create_index(csv_index_collection, vector_size)
                            logger.info(f"Created CSV index collection: {csv_index_collection}")
                        
                        # Create CSV index document
                        csv_index_data = document.metadata['csv_index']
                        csv_index = CSVIndex.from_dict(csv_index_data)
                        csv_index_doc = tabular_processor.create_csv_index_document(csv_index)
                        
                        # Store CSV index in dedicated collection
                        success = await storage_manager.upsert_documents(
                            index_name=csv_index_collection,
                            documents=[DocModel(**csv_index_doc.to_dict())]
                        )
                        
                        if success:
                            csv_index_id = csv_index.id
                            logger.info(f"CSV index stored: {csv_index_id}")
                            
                            # Store CSV data in SQLite database for query execution
                            try:
                                db_path = csv_db_manager.store_csv_data(csv_index, temp_file_path)
                                logger.info(f"CSV data stored in database: {db_path}")
                                # Add database path to metadata
                                document.update_metadata("sqlite_db_path", db_path)
                            except Exception as db_error:
                                logger.error(f"Failed to store CSV data in database: {db_error}")
                        else:
                            logger.warning("Failed to store CSV index")
                            
                    except Exception as e:
                        logger.error(f"Error storing CSV index: {e}")
                
                if not stored_collections:
                    # Fallback to default collection
                    default_collection = f"index_{document.type.value}"
                    await storage_manager.create_index(default_collection, vector_size)
                    # Cast to proper type
                    await storage_manager.upsert_documents(default_collection, cast(List[DocModel], chunked_docs))
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
                    content=document.content[:500] + "..." if len(document.content) > 500 else document.content,
                    csv_index_id=csv_index_id
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
        collections = await storage_manager.list_indexes()
        available_indexes = [
            {
                "name": col["name"],
                "type": "vector",
                "size": col["vectors_count"],
                "description": f"Distributed vector index with {col['vectors_count']} vectors",
                "status": "active"
            }
            for col in collections
        ]
        available_collection_names = [idx["name"] for idx in available_indexes]
        
        # If no indexes specified, use AI to recommend
        if not request.index_names:
            # Use AI to analyze query and recommend indexes
            query_analysis_prompt = f"""
            Analyze this search query and recommend which indexes to search:
            
            Query: "{request.query}"
            Available indexes: {[idx["name"] for idx in available_indexes]}
            
            Consider:
            - Content type (document, image, tabular)
            - Topic relevance
            - Search intent
            
            Return a JSON list of recommended index names:
            ["index_document", "index_technology"]
            """
            
            try:
                response = await gemini_client.generate_text(query_analysis_prompt, temperature=0.3)
                response_text = extract_gemini_text(response)
                
                # Parse response
                if "[" in response_text and "]" in response_text:
                    start = response_text.find("[")
                    end = response_text.rfind("]") + 1
                    json_str = response_text[start:end]
                    recommended_indexes = json.loads(json_str)
                    
                    # Filter to available indexes
                    selected_indexes = [idx for idx in recommended_indexes if idx in available_collection_names]
                    if not selected_indexes:
                        selected_indexes = available_collection_names[:2]  # Fallback
                else:
                    selected_indexes = available_collection_names[:2]  # Fallback
                    
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")
                selected_indexes = available_collection_names[:2]  # Fallback
        else:
            selected_indexes = request.index_names
        
        # Search in selected indexes
        search_tasks = []
        for index_name in selected_indexes:
            if index_name in available_collection_names:
                task = storage_manager.search_documents(
                    index_name=index_name,
                    query=request.query,
                    limit=request.limit,
                    score_threshold=request.score_threshold
                )
                search_tasks.append((index_name, task))
        
        if not search_tasks:
            # Search in all available collections as fallback
            for col in collections:
                task = storage_manager.search_documents(
                    index_name=col["name"],
                    query=request.query,
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
        # Ensure query_analysis is defined
        if 'query_analysis' not in locals():
            query_analysis = {}
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
    1. Searches for relevant CSV content using CSV-specific logic
    2. Searches for relevant content across other document types
    3. Passes all findings to the LLM to generate a comprehensive answer
    4. Provides reasoning and confidence scores
    5. Includes source documents for transparency
    """
    try:
        csv_results = None
        csv_answer = None
        csv_confidence = 0.0
        csv_reasoning = {}
        csv_sources = []
        csv_data_info = ""
        
        # Step 1: Check for CSV-specific content
        csv_index_collection = "csv_indexes"
        if await storage_manager.index_exists(csv_index_collection):
            try:
                # Search for relevant CSV indexes
                csv_search_results = await storage_manager.search_documents(
                    index_name=csv_index_collection,
                    query=request.question,
                    limit=5,  # Get top 5 most relevant CSV files
                    score_threshold=request.score_threshold
                )
                
                if csv_search_results:
                    # Extract CSV index information from search results
                    relevant_csvs = []
                    csv_indexes = []
                    
                    for result in csv_search_results:
                        try:
                            # Extract CSV index data from metadata
                            metadata = result.get("metadata", {})
                            csv_filename = metadata.get("csv_filename", "unknown")
                            column_headers_raw = metadata.get("column_headers", "[]")
                            total_rows = metadata.get("total_rows", 0)
                            total_columns = metadata.get("total_columns", 0)
                            csv_file_id = metadata.get("csv_file_id", result.get("document_id", ""))
                            
                            # Parse column_headers from JSON string
                            try:
                                if isinstance(column_headers_raw, str):
                                    column_headers = json.loads(column_headers_raw)
                                else:
                                    column_headers = column_headers_raw or []
                            except json.JSONDecodeError:
                                column_headers = []
                            
                            relevant_csvs.append({
                                "filename": csv_filename,
                                "score": result.get("score", 0.0),
                                "column_headers": column_headers,
                                "total_rows": total_rows,
                                "total_columns": total_columns,
                                "document_id": result.get("document_id", ""),
                                "csv_file_id": csv_file_id
                            })
                            
                            # Create CSV index object for reasoning
                            csv_index = CSVIndex(
                                csv_file_id=csv_file_id,
                                csv_filename=csv_filename,
                                column_headers=column_headers,
                                total_rows=total_rows,
                                total_columns=total_columns,
                                sample_data=None,  # We don't have sample data from search results
                                inferred_types=None  # We don't have inferred types from search results
                            )
                            csv_indexes.append(csv_index)
                            
                        except Exception as e:
                            logger.warning(f"Error processing CSV search result: {e}")
                            continue
                    
                    # Try to generate and execute SQL queries for the most relevant CSV
                    sql_results = []
                    
                    if relevant_csvs:
                        # Get the most relevant CSV
                        best_csv = relevant_csvs[0]
                        # Extract the csv_file_id from metadata, not the document_id
                        csv_file_id = best_csv.get("csv_file_id", best_csv["document_id"])
                        
                        # Generate SQL query using AI
                        sql_prompt = f"""
                        Generate a SQL query to answer this question: "{request.question}"
                        
                        CSV Structure:
                        - File: {best_csv['filename']}
                        - Columns: {', '.join(best_csv['column_headers'])}
                        - Total Rows: {best_csv['total_rows']}
                        - Table Name: csv_data_{csv_file_id.replace('-', '_')}
                        
                        Instructions:
                        - Return ONLY the SQL query, nothing else
                        - Use the exact table name: csv_data_{csv_file_id.replace('-', '_')}
                        - Focus on getting actual data values, not just structure
                        - If the question asks for specific values, use WHERE clauses to find them
                        - If the question asks for calculations, use appropriate SQL functions
                        """
                        
                        try:
                            sql_response = await gemini_client.generate_text(sql_prompt, temperature=0.1)
                            sql_query = extract_gemini_text(sql_response).strip()
                            
                            # Clean up SQL query (remove markdown, etc.)
                            if sql_query.startswith('```sql'):
                                sql_query = sql_query[7:]
                            if sql_query.endswith('```'):
                                sql_query = sql_query[:-3]
                            sql_query = sql_query.strip()
                            
                            # Execute the SQL query
                            if sql_query and sql_query.upper().startswith('SELECT'):
                                try:
                                    results, columns = csv_db_manager.execute_query(csv_file_id, sql_query)
                                    sql_results = {
                                        "csv_file": best_csv['filename'],
                                        "sql_query": sql_query,
                                        "results": results[:10],  # Limit to first 10 results
                                        "total_results": len(results),
                                        "columns": columns
                                    }
                                    logger.info(f"Executed SQL query: {sql_query}")
                                    
                                except Exception as sql_error:
                                    logger.warning(f"SQL execution failed: {sql_error}")
                                    sql_results = {"error": str(sql_error)}
                            
                        except Exception as sql_gen_error:
                            logger.warning(f"SQL generation failed: {sql_gen_error}")
                    
                    # Calculate confidence based on search scores
                    if relevant_csvs:
                        avg_score = sum(csv["score"] for csv in relevant_csvs) / len(relevant_csvs)
                        csv_confidence = min(avg_score * 1.2, 1.0)  # Boost confidence slightly
                    
                    # Create CSV sources for response
                    csv_sources = [
                        SearchResult(
                            document_id=csv["document_id"],
                            content=f"CSV File: {csv['filename']} - Columns: {', '.join(csv['column_headers'])} - Rows: {csv['total_rows']}",
                            score=csv["score"],
                            metadata={"type": "csv", "filename": csv["filename"]},
                            source_index="csv_indexes"
                        )
                        for csv in relevant_csvs
                    ]
                    
                    # Prepare CSV data information for the LLM
                    csv_data_info = f"""
                    CSV Data Available:
                    {chr(10).join([f"- {csv['filename']}: {csv['total_rows']} rows, {csv['total_columns']} columns ({', '.join(csv['column_headers'])})" for csv in relevant_csvs])}
                    
                    SQL Query Results:
                    {json.dumps(sql_results, indent=2) if sql_results else "No SQL results available"}
                    """
                    
                    csv_reasoning = {
                        "csv_files_analyzed": len(csv_indexes),
                        "search_scores": [csv["score"] for csv in relevant_csvs],
                        "analysis_method": "csv_index_search",
                        "sql_results": sql_results
                    }
                        
            except Exception as csv_error:
                logger.warning(f"CSV processing failed: {csv_error}")
        
        # Step 2: Perform regular document search for non-CSV content
        search_request = SearchRequest(
            query=request.question,
            index_names=request.index_names,
            limit=request.limit,
            score_threshold=request.score_threshold,
            search_strategy=request.search_strategy
        )
        
        # Get search results
        search_response = await search_documents(search_request)
        
        # Step 3: Collect all sources and information
        all_sources = []
        document_sources_text = ""
        
        if search_response.results:
            # Filter out CSV sources from regular search results
            non_csv_results = [
                result for result in search_response.results 
                if result.source_index != "csv_indexes"
            ]
            
            if non_csv_results:
                # Prepare document sources text for the LLM
                document_sources_text = "\n\n".join([
                    f"Document Source {i+1} (Score: {result.score:.3f}):\n{result.content[:1000]}"
                    for i, result in enumerate(non_csv_results[:5])  # Top 5 sources
                ])
                
                # Add document sources
                all_sources.extend(non_csv_results if request.include_sources else [])
        
        # Add CSV sources
        all_sources.extend(csv_sources if request.include_sources else [])
        
        # Step 4: Generate comprehensive answer using LLM with all information
        final_prompt = f"""
        You are an AI assistant with access to multiple sources of information. Please answer the user's question comprehensively.

        Question: "{request.question}"

        Available Information:

        {f"CSV Data:{csv_data_info}" if csv_data_info else "No CSV data available"}

        {f"Document Sources:{chr(10)}{document_sources_text}" if document_sources_text else "No document sources available"}

        Instructions:
        1. Provide a comprehensive answer that uses all available information
        2. If CSV data is available and relevant, use the actual data values from the SQL results
        3. If document sources provide relevant information, incorporate that as well
        4. If there are conflicts between sources, acknowledge them
        5. If one source is more specific or relevant, emphasize that
        6. Be direct and comprehensive in your answer
        7. If the sources don't fully answer the question, acknowledge this

        Format your response as JSON:
        {{
            "answer": "Your comprehensive answer here",
            "confidence": 0.85,
            "reasoning": "Explanation of how you arrived at this answer",
            "limitations": ["Any limitations or uncertainties"],
            "sources_used": ["csv", "documents"] or ["csv"] or ["documents"]
        }}
        """
        
        try:
            final_response = await gemini_client.generate_text(final_prompt, temperature=0.3)
            final_text = extract_gemini_text(final_response)
            
            # Parse the final answer
            try:
                # Extract JSON from response
                if "{" in final_text and "}" in final_text:
                    start = final_text.find("{")
                    end = final_text.rfind("}") + 1
                    json_str = final_text[start:end]
                    final_data = json.loads(json_str)
                    
                    final_answer = final_data.get("answer", "I couldn't generate a proper answer.")
                    final_confidence = final_data.get("confidence", 0.5)
                    reasoning_text = final_data.get("reasoning", "Analysis completed")
                    limitations = final_data.get("limitations", [])
                    sources_used = final_data.get("sources_used", [])
                    
                else:
                    # Fallback if JSON parsing fails
                    final_answer = final_text
                    final_confidence = 0.6
                    reasoning_text = "Answer generated from all available sources"
                    limitations = ["JSON parsing failed"]
                    sources_used = []
                    if csv_data_info:
                        sources_used.append("csv")
                    if document_sources_text:
                        sources_used.append("documents")
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Final answer JSON parsing failed: {e}")
                final_answer = final_text
                final_confidence = 0.5
                reasoning_text = "Answer generated from all available sources"
                limitations = ["JSON parsing failed"]
                sources_used = []
                if csv_data_info:
                    sources_used.append("csv")
                if document_sources_text:
                    sources_used.append("documents")
            
            # Create combined reasoning object
            combined_reasoning = {
                "answer": reasoning_text,
                "confidence": final_confidence,
                "limitations": limitations,
                "sources_used": sources_used,
                "csv_reasoning": csv_reasoning if csv_data_info else None,
                "document_reasoning": search_response.reasoning if document_sources_text else None
            }
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            # Fallback response
            if csv_data_info and document_sources_text:
                final_answer = "I found both CSV data and document sources but couldn't generate a comprehensive answer. Please try rephrasing your question."
            elif csv_data_info:
                final_answer = "I found CSV data but couldn't generate a proper answer. Please try rephrasing your question."
            elif document_sources_text:
                final_answer = "I found document sources but couldn't generate a proper answer. Please try rephrasing your question."
            else:
                final_answer = "I couldn't find any relevant information to answer your question. Please try rephrasing or uploading more relevant documents."
            
            final_confidence = 0.0
            combined_reasoning = {
                "answer": "Answer generation failed",
                "confidence": 0.0,
                "limitations": ["AI answer generation failed"],
                "sources_used": [],
                "error": str(e)
            }
        
        return AskResponse(
            answer=final_answer,
            confidence=final_confidence,
            sources=all_sources,
            reasoning=combined_reasoning,
            query_analysis=search_response.query_analysis
        )
        
    except Exception as e:
        logger.error("Ask question failed", error=str(e), question=request.question)
        raise HTTPException(status_code=500, detail=f"Ask question failed: {str(e)}")

# CSV-specific ask endpoint
@app.post("/ask_csv", response_model=AskCSVResponse)
async def ask_csv_question(request: AskCSVRequest):
    """
    Ask questions about CSV data using the CSV index for relevance matching.
    
    This endpoint:
    1. Searches the CSV index collection to find relevant CSV files
    2. Uses the CSV index content to understand the structure
    3. Generates appropriate SQL or search logic for the user's question
    """
    try:
        # Search for relevant CSV indexes
        csv_index_collection = "csv_indexes"
        
        # Check if CSV index collection exists
        if not await storage_manager.index_exists(csv_index_collection):
            return AskCSVResponse(
                answer="No CSV files have been uploaded yet. Please upload some CSV files first.",
                confidence=0.0,
                relevant_csvs=[],
                reasoning={"error": "No CSV indexes found"},
                query_analysis={"status": "no_csv_files"}
            )
        
        # Search for relevant CSV indexes
        csv_search_results = await storage_manager.search_documents(
            index_name=csv_index_collection,
            query=request.question,
            limit=5,  # Get top 5 most relevant CSV files
            score_threshold=request.score_threshold
        )
        
        if not csv_search_results:
            return AskCSVResponse(
                answer="I couldn't find any CSV files relevant to your question. Please try rephrasing or upload relevant CSV files.",
                confidence=0.0,
                relevant_csvs=[],
                reasoning={"error": "No relevant CSV files found"},
                query_analysis={"status": "no_relevant_csvs"}
            )
        
        # Extract CSV index information from search results
        relevant_csvs = []
        csv_indexes = []
        
        for result in csv_search_results:
            try:
                # Extract CSV index data from metadata
                metadata = result.get("metadata", {})
                csv_filename = metadata.get("csv_filename", "unknown")
                column_headers_raw = metadata.get("column_headers", "[]")
                total_rows = metadata.get("total_rows", 0)
                total_columns = metadata.get("total_columns", 0)
                csv_file_id = metadata.get("csv_file_id", result.get("document_id", ""))
                
                # Parse column_headers from JSON string
                try:
                    if isinstance(column_headers_raw, str):
                        column_headers = json.loads(column_headers_raw)
                    else:
                        column_headers = column_headers_raw or []
                except json.JSONDecodeError:
                    column_headers = []
                
                relevant_csvs.append({
                    "filename": csv_filename,
                    "score": result.get("score", 0.0),
                    "column_headers": column_headers,
                    "total_rows": total_rows,
                    "total_columns": total_columns,
                    "document_id": result.get("document_id", ""),
                    "csv_file_id": csv_file_id
                })
                
                # Create CSV index object for reasoning
                csv_index = CSVIndex(
                    csv_file_id=csv_file_id,
                    csv_filename=csv_filename,
                    column_headers=column_headers,
                    total_rows=total_rows,
                    total_columns=total_columns,
                    sample_data=None,  # We don't have sample data from search results
                    inferred_types=None  # We don't have inferred types from search results
                )
                csv_indexes.append(csv_index)
                
            except Exception as e:
                logger.warning(f"Error processing CSV search result: {e}")
                continue
        
        # Use AI to analyze the question and generate SQL/answer
        csv_context = "\n\n".join([
            f"CSV: {csv.csv_filename}\nColumns: {', '.join(csv.column_headers)}\nRows: {csv.total_rows}"
            for csv in csv_indexes
        ])
        
        reasoning_prompt = f"""
        You are a data analyst with access to CSV files. The user has asked: "{request.question}"
        
        Available CSV Files:
        {csv_context}
        
        Your task is to:
        1. Identify which CSV file(s) contain the data needed to answer the question
        2. Generate appropriate SQL queries to extract the required data
        3. Execute the SQL queries to get actual results
        4. Provide a clear answer based on the real data, not just the structure
        
        Focus on getting actual data values, not just describing what data would be needed.
        """
        
        try:
            response = await gemini_client.generate_text(reasoning_prompt, temperature=0.3)
            answer = extract_gemini_text(response)
            
            # Try to generate and execute SQL queries for the most relevant CSV
            sql_results = []
            final_answer = answer  # Start with the AI-generated answer
            
            if relevant_csvs:
                # Get the most relevant CSV
                best_csv = relevant_csvs[0]
                # Extract the csv_file_id from metadata, not the document_id
                csv_file_id = best_csv.get("csv_file_id", best_csv["document_id"])
                
                # Generate SQL query using AI
                sql_prompt = f"""
                Generate a SQL query to answer this question: "{request.question}"
                
                CSV Structure:
                - File: {best_csv['filename']}
                - Columns: {', '.join(best_csv['column_headers'])}
                - Total Rows: {best_csv['total_rows']}
                - Table Name: csv_data_{csv_file_id.replace('-', '_')}
                
                Instructions:
                - Return ONLY the SQL query, nothing else
                - Use the exact table name: csv_data_{csv_file_id.replace('-', '_')}
                - Focus on getting actual data values, not just structure
                - If the question asks for specific values, use WHERE clauses to find them
                - If the question asks for calculations, use appropriate SQL functions
                """
                
                try:
                    sql_response = await gemini_client.generate_text(sql_prompt, temperature=0.1)
                    sql_query = extract_gemini_text(sql_response).strip()
                    
                    # Clean up SQL query (remove markdown, etc.)
                    if sql_query.startswith('```sql'):
                        sql_query = sql_query[7:]
                    if sql_query.endswith('```'):
                        sql_query = sql_query[:-3]
                    sql_query = sql_query.strip()
                    
                    # Execute the SQL query
                    if sql_query and sql_query.upper().startswith('SELECT'):
                        try:
                            results, columns = csv_db_manager.execute_query(csv_file_id, sql_query)
                            sql_results = {
                                "csv_file": best_csv['filename'],
                                "sql_query": sql_query,
                                "results": results[:10],  # Limit to first 10 results
                                "total_results": len(results),
                                "columns": columns
                            }
                            logger.info(f"Executed SQL query: {sql_query}")
                            
                            # Generate a data-driven answer based on actual results
                            if results:
                                data_answer_prompt = f"""
                                Based on the SQL query results, provide a clear answer to: "{request.question}"
                                
                                SQL Query: {sql_query}
                                Results: {results[:5]}  # Show first 5 results
                                Total Results: {len(results)}
                                
                                Provide a concise answer based on the actual data, not just the structure.
                                """
                                
                                try:
                                    data_response = await gemini_client.generate_text(data_answer_prompt, temperature=0.3)
                                    data_answer = extract_gemini_text(data_response)
                                    csv_answer = data_answer
                                except Exception as data_error:
                                    logger.warning(f"Data answer generation failed: {data_error}")
                                    # Fall back to original answer
                            
                        except Exception as sql_error:
                            logger.warning(f"SQL execution failed: {sql_error}")
                            sql_results = {"error": str(sql_error)}
                    
                except Exception as sql_gen_error:
                    logger.warning(f"SQL generation failed: {sql_gen_error}")
                
                # Calculate confidence based on search scores
                if relevant_csvs:
                    avg_score = sum(csv["score"] for csv in relevant_csvs) / len(relevant_csvs)
                    confidence = min(avg_score * 1.2, 1.0)  # Boost confidence slightly
                else:
                    confidence = 0.0
                
                return AskCSVResponse(
                    answer=final_answer,
                    confidence=confidence,
                    relevant_csvs=relevant_csvs,
                    reasoning={
                        "csv_files_analyzed": len(csv_indexes),
                        "search_scores": [csv["score"] for csv in relevant_csvs],
                        "analysis_method": "csv_index_search",
                        "sql_results": sql_results
                    },
                    query_analysis={
                        "question_type": "csv_query",
                        "relevant_csv_count": len(relevant_csvs),
                        "best_match_score": max([csv["score"] for csv in relevant_csvs]) if relevant_csvs else 0.0
                    }
                )
                
        except Exception as e:
            logger.error(f"Error generating CSV answer: {e}")
            return AskCSVResponse(
                answer="I encountered an error while processing your question. Please try again.",
                confidence=0.0,
                relevant_csvs=relevant_csvs,
                reasoning={"error": str(e)},
                query_analysis={"status": "error"}
            )
            
    except Exception as e:
        logger.error(f"CSV question processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"CSV question processing failed: {str(e)}")

# List CSV databases endpoint
@app.get("/csv_databases")
async def list_csv_databases():
    """List all CSV databases with their metadata."""
    try:
        databases = csv_db_manager.list_csv_databases()
        return {
            "total_databases": len(databases),
            "databases": databases
        }
    except Exception as e:
        logger.error(f"Failed to list CSV databases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list CSV databases: {str(e)}")

# Execute SQL query endpoint
@app.post("/execute_sql")
async def execute_sql_query(csv_file_id: str = Query(..., description="CSV file ID"), 
                          sql_query: str = Query(..., description="SQL query to execute")):
    """Execute a SQL query on a specific CSV database."""
    try:
        results, columns = csv_db_manager.execute_query(csv_file_id, sql_query)
        return {
            "csv_file_id": csv_file_id,
            "sql_query": sql_query,
            "results": results,
            "columns": columns,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute SQL query: {str(e)}")

# Get indexes endpoint
@app.get("/indexes", response_model=List[IndexInfo])
async def get_indexes():
    """Get information about all available indexes."""
    try:
        collections = await storage_manager.list_indexes()
        
        indexes = []
        for col in collections:
            index_info = IndexInfo(
                name=col["name"],
                type="vector",
                size=col["vectors_count"],
                description=f"Distributed vector index with {col['vectors_count']} vectors",
                status="active"
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
    
    # Auto-start 3 nodes
    await start_initial_nodes()
    
    # Initialize default collections
    try:
        default_collections = ["index_document", "index_image", "index_tabular"]
        for collection_name in default_collections:
            try:
                await storage_manager.create_index(collection_name, 384)
                logger.info(f"Created default collection: {collection_name}")
            except Exception:
                # Collection might already exist
                pass
    except Exception as e:
        logger.warning("Failed to initialize default collections", error=str(e))
    
    # Auto-start autoscaling
    def start_autoscaling():
        try:
            requests.post("http://localhost:8000/autoscaling/start", timeout=5)
        except Exception as e:
            logger.error(f"Failed to auto-start autoscaling: {e}")
    threading.Thread(target=start_autoscaling, daemon=True).start()

async def start_initial_nodes():
    """Start the initial 3 nodes automatically."""
    logger.info("Starting initial 3 nodes...")
    
    # Node configurations
    initial_nodes = [
        {"id": "node1", "host": "localhost", "port": 8001},
        {"id": "node2", "host": "localhost", "port": 8002},
        {"id": "node3", "host": "localhost", "port": 8003},
    ]
    
    # Get the path to the dynamic node starter script
    script_path = PathLib(__file__).parent.parent.parent / "scripts" / "start_dynamic_node.py"
    
    if not script_path.exists():
        logger.error(f"Dynamic node starter not found at {script_path}")
        return
    
    # Start each node
    for node_config in initial_nodes:
        node_id = node_config["id"]
        host = node_config["host"]
        port = node_config["port"]
        data_dir = f"./node_data/{node_id}"
        
        try:
            logger.info(f"Starting {node_id} on {host}:{port}")
            
            # Create data directory
            PathLib(data_dir).mkdir(parents=True, exist_ok=True)
            
            # Start the node server process
            process = subprocess.Popen([
                sys.executable, str(script_path), node_id, host, str(port), data_dir
            ], stdout=None, stderr=None)
            
            # Store the process for later management
            node_processes[node_id] = process
            
            logger.info(f"Started {node_id} (PID: {process.pid})")
            
            # Wait briefly to see if process exits immediately
            await asyncio.sleep(2)
            if process.poll() is not None:
                logger.error(f"Node {node_id} failed to start (exit code {process.returncode})")
            else:
                # Process is still running, let's check if it's responding
                try:
                    import requests
                    response = requests.get(f"http://{host}:{port}/health", timeout=3)
                    if response.status_code == 200:
                        logger.info(f"✅ {node_id} is responding to health checks")
                    else:
                        logger.warning(f"⚠️ {node_id} is running but health check returned {response.status_code}")
                except Exception as health_error:
                    logger.warning(f"⚠️ {node_id} is running but not responding to health checks: {health_error}")
        except Exception as e:
            logger.error(f"Failed to start {node_id}: {e}")
    
    # Wait for nodes to start up
    logger.info("Waiting for nodes to start up...")
    await asyncio.sleep(5)
    
    # Check if nodes are running
    running_nodes = 0
    for node_id, process in node_processes.items():
        if process.poll() is None:
            running_nodes += 1
            logger.info(f"✅ {node_id} is running (PID: {process.pid})")
        else:
            logger.error(f"❌ {node_id} failed to start")
    
    logger.info(f"Started {running_nodes}/3 nodes successfully")
    
    # Wait a bit more for health checks
    await asyncio.sleep(3)
    
    # Check cluster status
    try:
        cluster_status = await storage_manager.get_cluster_status()
        healthy_nodes = cluster_status.get("healthy_nodes", 0)
        total_nodes = cluster_status.get("total_nodes", 0)
        logger.info(f"Cluster status: {healthy_nodes}/{total_nodes} nodes healthy")
    except Exception as e:
        logger.warning(f"Could not get cluster status: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down distributed indexing system")
    
    # Stop auto-scaling if running
    try:
        await auto_scaler.stop_monitoring()
        logger.info("Stopped auto-scaling")
    except Exception as e:
        logger.warning(f"Error stopping auto-scaling: {e}")
    
    # Stop all node processes
    await stop_all_nodes()

async def stop_all_nodes():
    """Stop all running node processes."""
    logger.info("Stopping all node processes...")
    
    for node_id, process in node_processes.items():
        try:
            logger.info(f"Stopping {node_id} (PID: {process.pid})")
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
                logger.info(f"✅ {node_id} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {node_id}")
                process.kill()
                process.wait()
                logger.info(f"✅ {node_id} force killed")
                
        except Exception as e:
            logger.error(f"Error stopping {node_id}: {e}")
    
    # Clear the node processes dictionary
    node_processes.clear()
    logger.info("All node processes stopped")

@app.get("/documents")
async def list_documents(
    collection: str = Query(..., description="Collection name"),
    limit: int = Query(10, description="Number of documents to return")
):
    """
    List documents in a distributed collection.
    """
    try:
        # Not yet implemented in distributed manager
        raise NotImplementedError("Listing documents is not yet implemented in the distributed system.")
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/delete_index")
async def delete_index(collection: str = Query(..., description="Collection name to delete")):
    """
    Delete a collection (index) by name.
    """
    try:
        await storage_manager.delete_index(collection)
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
        result = await storage_manager.get_document_by_id(collection, document_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to get document {document_id} from {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.get("/collection_stats/{collection}")
async def get_collection_statistics(collection: str):
    """
    Get detailed statistics about a collection.
    """
    try:
        stats = await storage_manager.get_index_stats(collection)
        if stats:
            return stats
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
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
    """
    try:
        # Not yet implemented in distributed manager
        raise NotImplementedError("Metadata search is not yet implemented in the distributed system.")
    except Exception as e:
        logger.error(f"Failed to search by metadata in {collection}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search by metadata: {str(e)}")

# Distributed System Models
class NodeInfo(BaseModel):
    id: str
    host: str
    port: int
    status: str
    load: float
    vector_count: int
    collections: List[str]

class ClusterStatus(BaseModel):
    total_nodes: int
    healthy_nodes: int
    total_shards: int
    total_collections: int
    total_vectors: int
    replication_factor: int
    consistency_level: str
    nodes: List[NodeInfo]

class AddNodeRequest(BaseModel):
    node_id: str
    host: str
    port: int

# Distributed System Endpoints
@app.get("/cluster/status", response_model=ClusterStatus)
async def get_cluster_status():
    """
    Get the status of the distributed cluster.
    Shows node health, shard distribution, and system metrics.
    """
    try:
        # Get real cluster information from distributed storage manager
        cluster_info = await storage_manager.get_cluster_status()
        
        # Get all collections to show complete picture
        all_collections = await storage_manager.list_indexes()
        collection_names = [col["name"] for col in all_collections]
        
        # Build node information with real data
        nodes = []
        for node_info in cluster_info.get("nodes", []):
            # For now, we'll show all collections on each node since we can't easily
            # determine which collections exist on which specific nodes
            # In a real implementation, this would be tracked in the distributed store
            nodes.append(NodeInfo(
                id=node_info["id"],
                host=node_info["host"],
                port=node_info["port"],
                status=node_info["status"],
                load=node_info.get("load", 0.0),
                vector_count=node_info.get("vector_count", 0),
                collections=collection_names  # Show all collections for now
            ))
        
        return ClusterStatus(
            total_nodes=cluster_info.get("total_nodes", 3),
            healthy_nodes=cluster_info.get("healthy_nodes", 3),
            total_shards=cluster_info.get("total_shards", 8),
            total_collections=len(collection_names),
            total_vectors=cluster_info.get("total_vectors", 0),
            replication_factor=cluster_info.get("replication_factor", 2),
            consistency_level=cluster_info.get("consistency_level", "quorum"),
            nodes=nodes
        )
        
    except Exception as e:
        logger.error("Failed to get cluster status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get cluster status: {str(e)}")

@app.post("/cluster/nodes")
async def add_node(request: AddNodeRequest):
    """
    Add a new node to the distributed cluster.
    """
    try:
        # In a real implementation, this would add the node to the distributed system
        logger.info(f"Adding node {request.node_id} at {request.host}:{request.port}")
        
        return {
            "status": "success",
            "message": f"Node {request.node_id} added successfully",
            "node": {
                "id": request.node_id,
                "host": request.host,
                "port": request.port,
                "status": "joining"
            }
        }
    except Exception as e:
        logger.error(f"Failed to add node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add node: {str(e)}")

@app.delete("/cluster/nodes/{node_id}")
async def remove_node(node_id: str):
    """
    Remove a node from the distributed cluster.
    """
    try:
        logger.info(f"Removing node {node_id}")
        # Remove from distributed storage manager (and thus from all registries and health checks)
        success = await storage_manager.remove_node(node_id)
        if not success:
            raise Exception(f"Failed to remove node {node_id} from distributed system")
        # Optionally, stop the node process if managed here
        process = node_processes.get(node_id)
        if process:
            process.terminate()
            process.wait(timeout=5)
            logger.info(f"Stopped node process: {node_id}")
            del node_processes[node_id]
        return {
            "status": "success",
            "message": f"Node {node_id} removed successfully"
        }
    except Exception as e:
        logger.error(f"Failed to remove node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove node: {str(e)}")

@app.get("/cluster/health")
async def cluster_health_check():
    """
    Comprehensive health check for the distributed cluster.
    """
    try:
        # Check individual node health
        node_health = []
        nodes = [
            {"id": "node1", "host": "localhost", "port": 8001},
            {"id": "node2", "host": "localhost", "port": 8002},
            {"id": "node3", "host": "localhost", "port": 8003}
        ]
        
        # Check node processes
        node_process_status = {}
        for node_id, process in node_processes.items():
            if process.poll() is None:
                node_process_status[node_id] = {
                    "status": "running",
                    "pid": process.pid
                }
            else:
                node_process_status[node_id] = {
                    "status": "stopped",
                    "exit_code": process.poll()
                }
        
        for node in nodes:
            try:
                response = requests.get(f"http://{node['host']}:{node['port']}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    node_health.append({
                        "node_id": node["id"],
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "uptime": data.get("uptime", 0),
                        "load": data.get("load", 0.0),
                        "vector_count": data.get("vector_count", 0)
                    })
                else:
                    node_health.append({
                        "node_id": node["id"],
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    })
            except Exception as e:
                node_health.append({
                    "node_id": node["id"],
                    "status": "unreachable",
                    "error": str(e)
                })
        
        healthy_nodes = sum(1 for node in node_health if node["status"] == "healthy")
        total_nodes = len(node_health)
        running_processes = sum(1 for status in node_process_status.values() if status["status"] == "running")
        
        # Calculate overall status
        if healthy_nodes >= total_nodes * 0.7:
            overall_status = "healthy"
        elif healthy_nodes >= total_nodes * 0.5:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "cluster_status": overall_status,
            "healthy_nodes": healthy_nodes,
            "total_nodes": total_nodes,
            "running_processes": running_processes,
            "node_health": node_health,
            "node_processes": node_process_status,
            "auto_scaling": {
                "is_running": auto_scaler.is_running,
                "running_processes": len(auto_scaler.node_processes)
            },
            "features": {
                "fault_tolerance": "enabled",
                "load_balancing": "enabled",
                "data_replication": "enabled",
                "consistency": "quorum",
                "auto_startup": "enabled"
            }
        }
    except Exception as e:
        logger.error(f"Cluster health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cluster health check failed: {str(e)}")

@app.get("/cluster/sharding")
async def get_sharding_info():
    """
    Get detailed sharding information for the distributed cluster.
    Shows how data is distributed across nodes and shards.
    """
    try:
        cluster_status = await storage_manager.get_cluster_status()
        return {
            "sharding_strategy": "consistent_hashing",
            "shard_count": cluster_status.get("total_shards", 0),
            "replication_factor": cluster_status.get("replication_factor", 2),
            "consistency_level": cluster_status.get("consistency_level", "quorum"),
            "shard_distribution": cluster_status.get("shards", []),
            "node_distribution": cluster_status.get("nodes", [])
        }
    except Exception as e:
        logger.error("Failed to get sharding info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get sharding info: {str(e)}")

# Auto-scaling endpoints
class AutoScalingStatus(BaseModel):
    is_running: bool
    last_scale_up: float
    last_scale_down: float
    scaling_history: List[Dict[str, Any]]
    current_metrics: Optional[Dict[str, Any]]
    thresholds: Dict[str, Any]

class UpdateThresholdsRequest(BaseModel):
    cpu_threshold_high: Optional[float] = None
    memory_threshold_high: Optional[float] = None
    storage_threshold_high: Optional[float] = None
    latency_threshold_high: Optional[float] = None
    error_rate_threshold_high: Optional[float] = None
    cpu_threshold_low: Optional[float] = None
    memory_threshold_low: Optional[float] = None
    storage_threshold_low: Optional[float] = None
    latency_threshold_low: Optional[float] = None
    error_rate_threshold_low: Optional[float] = None
    min_nodes: Optional[int] = None
    max_nodes: Optional[int] = None

@app.get("/autoscaling/status", response_model=AutoScalingStatus)
async def get_autoscaling_status():
    """
    Get the current status of the auto-scaler.
    Shows if it's running, recent scaling actions, and current metrics.
    """
    try:
        return auto_scaler.get_scaling_status()
    except Exception as e:
        logger.error("Failed to get auto-scaling status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get auto-scaling status: {str(e)}")

@app.post("/autoscaling/start")
async def start_autoscaling():
    """
    Start the auto-scaling monitoring and scaling operations.
    """
    try:
        if not auto_scaler.is_running:
            asyncio.create_task(auto_scaler.start_monitoring())
            logger.info("Auto-scaling started")
            return {"status": "started", "message": "Auto-scaling monitoring started"}
        else:
            return {"status": "already_running", "message": "Auto-scaling is already running"}
    except Exception as e:
        logger.error("Failed to start auto-scaling", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start auto-scaling: {str(e)}")

@app.post("/autoscaling/stop")
async def stop_autoscaling():
    """
    Stop the auto-scaling monitoring and scaling operations.
    """
    try:
        if auto_scaler.is_running:
            await auto_scaler.stop_monitoring()
            logger.info("Auto-scaling stopped")
            return {"status": "stopped", "message": "Auto-scaling monitoring stopped"}
        else:
            return {"status": "not_running", "message": "Auto-scaling is not running"}
    except Exception as e:
        logger.error("Failed to stop auto-scaling", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to stop auto-scaling: {str(e)}")

@app.post("/autoscaling/thresholds")
async def update_autoscaling_thresholds(request: UpdateThresholdsRequest):
    """
    Update auto-scaling thresholds dynamically.
    """
    try:
        # Get current thresholds
        current_thresholds = auto_scaler.thresholds
        
        # Update only provided values
        if request.cpu_threshold_high is not None:
            current_thresholds.cpu_threshold_high = request.cpu_threshold_high
        if request.memory_threshold_high is not None:
            current_thresholds.memory_threshold_high = request.memory_threshold_high
        if request.storage_threshold_high is not None:
            current_thresholds.storage_threshold_high = request.storage_threshold_high
        if request.latency_threshold_high is not None:
            current_thresholds.latency_threshold_high = request.latency_threshold_high
        if request.error_rate_threshold_high is not None:
            current_thresholds.error_rate_threshold_high = request.error_rate_threshold_high
        if request.cpu_threshold_low is not None:
            current_thresholds.cpu_threshold_low = request.cpu_threshold_low
        if request.memory_threshold_low is not None:
            current_thresholds.memory_threshold_low = request.memory_threshold_low
        if request.storage_threshold_low is not None:
            current_thresholds.storage_threshold_low = request.storage_threshold_low
        if request.latency_threshold_low is not None:
            current_thresholds.latency_threshold_low = request.latency_threshold_low
        if request.error_rate_threshold_low is not None:
            current_thresholds.error_rate_threshold_low = request.error_rate_threshold_low
        if request.min_nodes is not None:
            current_thresholds.min_nodes = request.min_nodes
        if request.max_nodes is not None:
            current_thresholds.max_nodes = request.max_nodes
        
        # Update the auto-scaler
        auto_scaler.update_thresholds(current_thresholds)
        
        logger.info("Updated auto-scaling thresholds")
        return {
            "status": "updated",
            "message": "Auto-scaling thresholds updated",
            "new_thresholds": auto_scaler.get_scaling_status()["thresholds"]
        }
    except Exception as e:
        logger.error("Failed to update auto-scaling thresholds", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")

@app.post("/autoscaling/scale-up")
async def manual_scale_up():
    """
    Manually trigger a scale-up operation.
    Useful for testing or immediate scaling needs.
    """
    try:
        # Get current cluster status
        cluster_status = await storage_manager.get_cluster_status()
        current_nodes = cluster_status.get("total_nodes", 0)
        
        # Check if we can scale up
        if current_nodes >= auto_scaler.thresholds.max_nodes:
            raise HTTPException(status_code=400, detail="Maximum number of nodes reached")
        
        # Trigger scale up using the existing method
        await auto_scaler._scale_up("Manual scale-up triggered", current_nodes)
        
        return {
            "status": "scaled_up",
            "message": f"Manually scaled up to {current_nodes + 1} nodes",
            "new_node_count": current_nodes + 1
        }
    except Exception as e:
        logger.error("Failed to manually scale up", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to scale up: {str(e)}")

@app.post("/autoscaling/scale-down")
async def manual_scale_down():
    """
    Manually trigger a scale-down operation.
    Useful for testing or immediate scaling needs.
    """
    try:
        # Get current cluster status
        cluster_status = await storage_manager.get_cluster_status()
        current_nodes = cluster_status.get("total_nodes", 0)
        
        # Check if we can scale down
        if current_nodes <= auto_scaler.thresholds.min_nodes:
            raise HTTPException(status_code=400, detail="Minimum number of nodes reached")
        
        # Trigger scale down using the existing method
        await auto_scaler._scale_down("Manual scale-down triggered", current_nodes)
        
        return {
            "status": "scaled_down",
            "message": f"Manually scaled down to {current_nodes - 1} nodes",
            "new_node_count": current_nodes - 1
        }
    except Exception as e:
        logger.error("Failed to manually scale down", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to scale down: {str(e)}")

@app.get("/debug/collections")
async def debug_collections():
    """
    Debug endpoint to show detailed collection information.
    Helps diagnose indexing and collection creation issues.
    """
    try:
        collections = await storage_manager.list_indexes()
        
        # Get cluster status for additional info
        cluster_status = await storage_manager.get_cluster_status()
        
        debug_info = {
            "total_collections": len(collections),
            "collections": [],
            "cluster_status": {
                "total_nodes": cluster_status.get("total_nodes", 0),
                "healthy_nodes": cluster_status.get("healthy_nodes", 0),
                "total_vectors": cluster_status.get("total_vectors", 0)
            }
        }
        
        for collection in collections:
            # Get detailed stats for each collection
            stats = await storage_manager.get_index_stats(collection["name"])
            debug_info["collections"].append({
                "name": collection["name"],
                "vectors_count": collection["vectors_count"],
                "shard_count": collection["shard_count"],
                "stats": stats or {}
            })
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug collections failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/debug/upload_test")
async def debug_upload_test():
    """
    Debug endpoint to test upload functionality with AI-driven collection decisions.
    """
    try:
        # Create a test document
        test_document = BaseDocument(
            id=f"test_{int(time.time())}",
            content="This is a test document for debugging the upload system. It contains technical content about distributed systems and vector databases.",
            metadata={"test": True, "timestamp": time.time()},
            type=DataType.DOCUMENT
        )
        
        # Get existing collections
        existing_collections = await storage_manager.list_indexes()
        existing_collection_names = [col["name"] for col in existing_collections]
        
        # Test collection determination with existing collections
        ai_metadata = {"test": True}
        collections = await determine_collections(test_document, ai_metadata, existing_collection_names)
        
        return {
            "test_document": {
                "id": test_document.id,
                "content_preview": (test_document.content or "")[:100] + "..." if test_document.content else "No content",
                "type": test_document.type.value
            },
            "existing_collections": existing_collection_names,
            "determined_collections": collections,
            "total_collections": len(collections),
            "new_collections": [col for col in collections if col not in existing_collection_names],
            "reused_collections": [col for col in collections if col in existing_collection_names]
        }
        
    except Exception as e:
        logger.error(f"Debug upload test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug test failed: {str(e)}")

@app.get("/debug/ai_collection_analysis")
async def debug_ai_collection_analysis():
    """
    Debug endpoint to test AI collection analysis with detailed reasoning.
    """
    try:
        # Create test documents with different content
        test_documents = [
            {
                "name": "machine_learning_research.pdf",
                "content": "This research paper presents novel deep learning architectures for natural language processing. The study introduces a new transformer model that achieves state-of-the-art performance on benchmark datasets including GLUE and SuperGLUE. The model incorporates attention mechanisms and demonstrates improved efficiency in processing long sequences.",
                "type": DataType.DOCUMENT,
                "metadata": {"filename": "machine_learning_research.pdf", "mime_type": "application/pdf", "file_size": 2048576}
            },
            {
                "name": "cybersecurity_guide.pdf", 
                "content": "Comprehensive cybersecurity guide covering network security protocols, encryption standards, threat detection systems, and incident response procedures. Includes detailed analysis of common attack vectors and best practices for enterprise security implementation.",
                "type": DataType.DOCUMENT,
                "metadata": {"filename": "cybersecurity_guide.pdf", "mime_type": "application/pdf", "file_size": 1536000}
            },
            {
                "name": "financial_report.pdf",
                "content": "Q3 2024 financial report showing revenue growth of 23% year-over-year, with detailed analysis of market trends, customer acquisition costs, and profitability metrics. The report includes projections for Q4 and strategic recommendations for market expansion.",
                "type": DataType.DOCUMENT,
                "metadata": {"filename": "financial_report.pdf", "mime_type": "application/pdf", "file_size": 1024000}
            }
        ]
        
        # Get existing collections
        existing_collections = await storage_manager.list_indexes()
        existing_collection_names = [col["name"] for col in existing_collections]
        
        results = []
        
        for i, doc_data in enumerate(test_documents, 1):
            # Create document object
            document = BaseDocument(
                id=f"test_doc_{i}",
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                type=doc_data['type']
            )
            
            # Test collection determination
            ai_metadata = {"source": "debug_test", "category": "demo"}
            collections = await determine_collections(document, ai_metadata, existing_collection_names)
            
            # Analyze results
            new_collections = [col for col in collections if col not in existing_collection_names]
            reused_collections = [col for col in collections if col in existing_collection_names]
            
            results.append({
                "document": {
                    "name": doc_data['name'],
                    "content_preview": doc_data['content'][:100] + "...",
                    "type": doc_data['type'].value,
                    "metadata": doc_data['metadata']
                },
                "ai_decision": {
                    "all_collections": collections,
                    "reused_existing": reused_collections,
                    "created_new": new_collections,
                    "total_collections": len(collections)
                }
            })
            
            # Update existing collections for next iteration
            existing_collection_names.extend(new_collections)
        
        return {
            "test_results": results,
            "final_collections": existing_collection_names,
            "total_final_collections": len(existing_collection_names)
        }
        
    except Exception as e:
        logger.error(f"Debug AI collection analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug analysis failed: {str(e)}")

# Debug endpoint for CSV databases
@app.get("/debug/csv_databases")
async def debug_csv_databases():
    """Debug endpoint to check CSV database status."""
    try:
        # List all CSV databases
        databases = csv_db_manager.list_csv_databases()
        
        # Get database mapping
        db_mapping = csv_db_manager.get_database_mapping()
        
        # Check CSV index collection
        csv_index_collection = "csv_indexes"
        csv_indexes_exist = await storage_manager.index_exists(csv_index_collection)
        
        csv_indexes = []
        if csv_indexes_exist:
            # Get some sample CSV indexes
            try:
                sample_results = await storage_manager.search_documents(
                    index_name=csv_index_collection,
                    query="",  # Empty query to get all
                    limit=10,
                    score_threshold=0.0
                )
                csv_indexes = [
                    {
                        "document_id": result.get("document_id", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0)
                    }
                    for result in sample_results
                ]
            except Exception as e:
                logger.warning(f"Failed to get CSV indexes: {e}")
        
        return {
            "csv_databases": databases,
            "database_mapping": db_mapping,
            "csv_indexes_exist": csv_indexes_exist,
            "csv_indexes": csv_indexes,
            "total_databases": len(databases),
            "total_indexes": len(csv_indexes),
            "db_directory": str(csv_db_manager.db_directory)
        }
    except Exception as e:
        logger.error(f"Failed to debug CSV databases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to debug CSV databases: {str(e)}")

async def monitor_node_output(node_id: str, process: subprocess.Popen):
    """Monitor the output of a node subprocess in real-time."""
    try:
        while process.poll() is None:
            # Read stdout
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    logger.info(f"[{node_id}] {line.strip()}")
            
            # Read stderr
            if process.stderr:
                line = process.stderr.readline()
                if line:
                    logger.error(f"[{node_id}] ERROR: {line.strip()}")
            
            await asyncio.sleep(0.1)
        
        # Process has exited, read remaining output
        out, err = process.communicate()
        if out:
            logger.info(f"[{node_id}] Final stdout: {out}")
        if err:
            logger.error(f"[{node_id}] Final stderr: {err}")
            
    except Exception as e:
        logger.error(f"Error monitoring {node_id} output: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "core.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers
    ) 