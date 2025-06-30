#!/usr/bin/env python3
"""
Example usage of the distributed indexing system.

This script demonstrates the complete flow:
1. Upload files (PDFs, CSVs, images)
2. Process and chunk documents
3. Generate embeddings using Gemini AI
4. Store in distributed Qdrant indexes
5. Search across multiple indexes with AI reasoning
"""

import asyncio
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our system components
from core.types.document_processor import DocumentProcessor
from core.types.image_processor import ImageProcessor
from core.types.tabular_processor import TabularProcessor
from core.services.ingestion.chunker import TextChunker
from core.services.inference.gemini_client import GeminiClient
from data.storage.qdrant_client import QdrantClient
from core.utils.logging import setup_logging, get_logger


class DistributedIndexingExample:
    """Example class demonstrating the distributed indexing system."""
    
    def __init__(self):
        """Initialize the example system."""
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
        self.tabular_processor = TabularProcessor()
        self.text_chunker = TextChunker()
        self.gemini_client = GeminiClient()
        self.qdrant_client = QdrantClient()
        
        self.logger.info("Distributed indexing example initialized")
    
    async def create_sample_files(self) -> List[str]:
        """Create sample files for demonstration."""
        sample_files = []
        
        # Create sample text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            Artificial Intelligence and Machine Learning
            
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that work and react like humans. Machine Learning (ML) is a 
            subset of AI that enables computers to learn and improve from experience without 
            being explicitly programmed.
            
            Key concepts in AI include:
            - Natural Language Processing (NLP)
            - Computer Vision
            - Robotics
            - Expert Systems
            
            Machine Learning algorithms can be categorized into:
            1. Supervised Learning
            2. Unsupervised Learning
            3. Reinforcement Learning
            
            Recent advances in AI have led to breakthroughs in areas such as:
            - Large Language Models (LLMs)
            - Computer Vision
            - Autonomous Vehicles
            - Healthcare Diagnostics
            """)
            sample_files.append(f.name)
        
        # Create sample CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("""name,age,city,occupation
John Doe,30,New York,Software Engineer
Jane Smith,25,San Francisco,Data Scientist
Bob Johnson,35,Chicago,Product Manager
Alice Brown,28,Boston,UX Designer
Charlie Wilson,32,Seattle,DevOps Engineer
""")
            sample_files.append(f.name)
        
        # Create sample JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "company": "TechCorp",
                "employees": [
                    {"id": 1, "name": "John Doe", "department": "Engineering", "salary": 80000},
                    {"id": 2, "name": "Jane Smith", "department": "Data Science", "salary": 90000},
                    {"id": 3, "name": "Bob Johnson", "department": "Product", "salary": 85000}
                ],
                "departments": ["Engineering", "Data Science", "Product", "Marketing"],
                "total_employees": 3
            }, f, indent=2)
            sample_files.append(f.name)
        
        self.logger.info(f"Created {len(sample_files)} sample files")
        return sample_files
    
    async def process_and_index_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process files and store them in distributed indexes."""
        results = {
            "processed_files": 0,
            "total_chunks": 0,
            "indexes_created": [],
            "errors": []
        }
        
        for file_path in file_paths:
            try:
                self.logger.info(f"Processing file: {file_path}")
                
                # Determine file type and process
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                    document = await self.image_processor.process_image(file_path)
                elif file_extension in ['.csv', '.xlsx', '.xls', '.parquet', '.json']:
                    document = await self.tabular_processor.process_tabular(file_path)
                else:
                    document = await self.document_processor.process_document(file_path)
                
                # Extract metadata using Gemini AI
                if document.content:
                    ai_metadata = await self.gemini_client.extract_metadata(
                        document.content, 
                        document.type.value
                    )
                    document.metadata.update(ai_metadata)
                    self.logger.info(f"Extracted metadata: {ai_metadata.get('topics', [])}")
                
                # Chunk document
                chunked_docs = await self.text_chunker.chunk_document(document, "auto")
                self.logger.info(f"Created {len(chunked_docs)} chunks")
                
                # Generate embeddings
                chunk_texts = [doc.content for doc in chunked_docs if doc.content]
                if chunk_texts:
                    embeddings = await self.gemini_client.generate_embeddings(chunk_texts)
                    
                    # Store in distributed Qdrant index
                    collection_name = f"index_{document.type.value}"
                    
                    # Create collection with distributed configuration
                    try:
                        await self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vector_size=len(embeddings[0]) if embeddings else 768,
                            shard_number=3,  # Distribute across 3 shards
                            replication_factor=2  # 2 replicas for fault tolerance
                        )
                        results["indexes_created"].append(collection_name)
                        self.logger.info(f"Created distributed collection: {collection_name}")
                    except Exception as e:
                        self.logger.warning(f"Collection {collection_name} might already exist: {e}")
                    
                    # Upsert vectors
                    await self.qdrant_client.upsert_vectors(
                        collection_name=collection_name,
                        vectors=embeddings,
                        documents=chunked_docs
                    )
                    
                    results["processed_files"] += 1
                    results["total_chunks"] += len(chunked_docs)
                    
                    self.logger.info(f"Successfully indexed {len(chunked_docs)} chunks in {collection_name}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                results["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        return results
    
    async def search_across_indexes(self, query: str) -> Dict[str, Any]:
        """Search across multiple distributed indexes with AI reasoning."""
        self.logger.info(f"Searching for: {query}")
        
        # Get available indexes
        collections = await self.qdrant_client.list_collections()
        available_indexes = [
            {
                "name": col["name"],
                "description": f"Index for {col['name']}",
                "type": "vector",
                "size": col.get("vectors_count", 0)
            }
            for col in collections
        ]
        
        self.logger.info(f"Available indexes: {[idx['name'] for idx in available_indexes]}")
        
        # Use Gemini AI to analyze query and select relevant indexes
        query_analysis = await self.gemini_client.analyze_query(query, available_indexes)
        selected_indexes = query_analysis.get("recommended_indexes", [])
        
        self.logger.info(f"AI selected indexes: {selected_indexes}")
        self.logger.info(f"Search strategy: {query_analysis.get('search_strategy')}")
        self.logger.info(f"Confidence: {query_analysis.get('confidence')}")
        
        # Generate query embedding
        query_embedding = await self.gemini_client.generate_embeddings([query])
        
        # Search across selected indexes in parallel
        search_tasks = []
        for index_name in selected_indexes:
            if index_name in [col["name"] for col in collections]:
                task = self.qdrant_client.search_vectors(
                    collection_name=index_name,
                    query_vector=query_embedding[0],
                    limit=5,
                    score_threshold=0.0
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
                self.logger.info(f"Found {len(results)} results in {index_name}")
            except Exception as e:
                self.logger.warning(f"Search failed for index {index_name}: {e}")
        
        # Sort results by score
        search_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Use Gemini AI to reason about results
        reasoning = await self.gemini_client.reason_about_results(query, search_results)
        
        return {
            "query": query,
            "query_analysis": query_analysis,
            "search_results": search_results,
            "reasoning": reasoning,
            "total_results": len(search_results)
        }
    
    async def demonstrate_scalability(self):
        """Demonstrate the scalability features."""
        self.logger.info("Demonstrating scalability features...")
        
        # Get collection information
        collections = await self.qdrant_client.list_collections()
        
        for collection in collections:
            info = await self.qdrant_client.get_collection_info(collection["name"])
            self.logger.info(f"Collection: {info['name']}")
            self.logger.info(f"  - Vectors: {info['vectors_count']}")
            self.logger.info(f"  - Points: {info['points_count']}")
            self.logger.info(f"  - Segments: {info['segments_count']}")
            self.logger.info(f"  - Vector size: {info['config']['vector_size']}")
            self.logger.info(f"  - Distance metric: {info['config']['distance']}")
    
    async def run_full_demo(self):
        """Run the complete demonstration."""
        self.logger.info("Starting distributed indexing demonstration...")
        
        try:
            # Step 1: Create sample files
            self.logger.info("Step 1: Creating sample files...")
            sample_files = await self.create_sample_files()
            
            # Step 2: Process and index files
            self.logger.info("Step 2: Processing and indexing files...")
            indexing_results = await self.process_and_index_files(sample_files)
            
            self.logger.info("Indexing Results:")
            self.logger.info(f"  - Processed files: {indexing_results['processed_files']}")
            self.logger.info(f"  - Total chunks: {indexing_results['total_chunks']}")
            self.logger.info(f"  - Indexes created: {indexing_results['indexes_created']}")
            if indexing_results['errors']:
                self.logger.warning(f"  - Errors: {len(indexing_results['errors'])}")
            
            # Step 3: Demonstrate scalability
            self.logger.info("Step 3: Demonstrating scalability...")
            await self.demonstrate_scalability()
            
            # Step 4: Search across indexes
            self.logger.info("Step 4: Searching across distributed indexes...")
            
            # Test queries
            test_queries = [
                "What is artificial intelligence?",
                "Tell me about machine learning algorithms",
                "What are the different types of AI?",
                "How does natural language processing work?",
                "What are the recent advances in AI?"
            ]
            
            for query in test_queries:
                self.logger.info(f"\n--- Testing Query: {query} ---")
                search_results = await self.search_across_indexes(query)
                
                self.logger.info(f"Query Analysis: {search_results['query_analysis']['reasoning']}")
                self.logger.info(f"Total Results: {search_results['total_results']}")
                
                if search_results['search_results']:
                    top_result = search_results['search_results'][0]
                    self.logger.info(f"Top Result (Score: {top_result['score']:.3f}):")
                    self.logger.info(f"  Content: {top_result['content'][:200]}...")
                    self.logger.info(f"  Source Index: {top_result['source_index']}")
                
                self.logger.info(f"AI Reasoning: {search_results['reasoning'].get('answer', 'No reasoning available')}")
            
            # Cleanup
            for file_path in sample_files:
                try:
                    os.unlink(file_path)
                except:
                    pass
            
            self.logger.info("Demonstration completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}")
            raise


async def main():
    """Main function to run the demonstration."""
    example = DistributedIndexingExample()
    await example.run_full_demo()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 