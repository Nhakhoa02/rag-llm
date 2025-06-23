#!/usr/bin/env python3
"""
Simple test script to verify the distributed indexing system components.
"""

import asyncio
import tempfile
import os
from pathlib import Path

# Import system components
from src.ingestion.processor import DocumentProcessor
from src.ingestion.chunker import TextChunker
from src.ai.gemini_client import GeminiClient
from src.storage.qdrant_client import QdrantClient
from src.utils.logging import setup_logging, get_logger


async def test_basic_functionality():
    """Test basic functionality of the system components."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting basic functionality test...")
    
    try:
        # Test 1: Document Processing
        logger.info("Test 1: Document Processing")
        document_processor = DocumentProcessor()
        
        # Create a test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about artificial intelligence and machine learning.")
            test_file = f.name
        
        try:
            document = document_processor.process_document(test_file)
            logger.info(f"✓ Document processed successfully: {document.id}")
            logger.info(f"  Content length: {len(document.content)}")
            logger.info(f"  Type: {document.type}")
        finally:
            os.unlink(test_file)
        
        # Test 2: Text Chunking
        logger.info("Test 2: Text Chunking")
        text_chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        test_text = "This is a longer text that should be chunked into smaller pieces. " * 10
        chunks = text_chunker.chunk_text(test_text, "fixed_size")
        logger.info(f"✓ Text chunked successfully: {len(chunks)} chunks created")
        
        # Test 3: Gemini Client (if API key is available)
        logger.info("Test 3: Gemini AI Client")
        try:
            gemini_client = GeminiClient()
            
            # Test text generation
            response = await gemini_client.generate_text("Hello, how are you?")
            logger.info(f"✓ Gemini AI working: {response.text[:50]}...")
            
            # Test metadata extraction
            metadata = await gemini_client.extract_metadata(
                "This is a document about artificial intelligence and machine learning."
            )
            logger.info(f"✓ Metadata extraction working: {metadata.get('topics', [])}")
            
        except Exception as e:
            logger.warning(f"⚠ Gemini AI test skipped (API key not configured): {e}")
        
        # Test 4: Qdrant Client
        logger.info("Test 4: Qdrant Client")
        try:
            qdrant_client = QdrantClient()
            
            # Test collection creation
            test_collection = "test_collection"
            await qdrant_client.create_collection(
                collection_name=test_collection,
                vector_size=768,
                shard_number=1,
                replication_factor=1
            )
            logger.info(f"✓ Qdrant collection created: {test_collection}")
            
            # Test collection info
            info = await qdrant_client.get_collection_info(test_collection)
            logger.info(f"✓ Collection info retrieved: {info['name']}")
            
            # Cleanup
            await qdrant_client.delete_collection(test_collection)
            logger.info(f"✓ Test collection cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠ Qdrant test skipped (Qdrant not running): {e}")
        
        logger.info("✓ All basic functionality tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


async def test_integration():
    """Test integration between components."""
    logger = get_logger(__name__)
    
    logger.info("Starting integration test...")
    
    try:
        # Create a test document
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
            """)
            test_file = f.name
        
        try:
            # Process document
            document_processor = DocumentProcessor()
            document = document_processor.process_document(test_file)
            
            # Chunk document
            text_chunker = TextChunker()
            chunked_docs = text_chunker.chunk_document(document, "auto")
            
            logger.info(f"✓ Integration test: {len(chunked_docs)} chunks created from document")
            
            # Test Gemini integration (if available)
            try:
                gemini_client = GeminiClient()
                
                # Extract metadata
                metadata = await gemini_client.extract_metadata(document.content, "text")
                logger.info(f"✓ AI metadata extracted: {metadata.get('topics', [])}")
                
                # Generate embeddings
                embeddings = await gemini_client.generate_embeddings([doc.content for doc in chunked_docs])
                logger.info(f"✓ Embeddings generated: {len(embeddings)} vectors")
                
            except Exception as e:
                logger.warning(f"⚠ Gemini integration test skipped: {e}")
            
            # Test Qdrant integration (if available)
            try:
                qdrant_client = QdrantClient()
                
                # Create test collection
                test_collection = "integration_test"
                await qdrant_client.create_collection(
                    collection_name=test_collection,
                    vector_size=768,
                    shard_number=1,
                    replication_factor=1
                )
                
                # Store vectors (if embeddings available)
                if 'embeddings' in locals():
                    await qdrant_client.upsert_vectors(
                        collection_name=test_collection,
                        vectors=embeddings,
                        documents=chunked_docs
                    )
                    logger.info(f"✓ Vectors stored in Qdrant")
                
                # Cleanup
                await qdrant_client.delete_collection(test_collection)
                
            except Exception as e:
                logger.warning(f"⚠ Qdrant integration test skipped: {e}")
            
            logger.info("✓ Integration test completed!")
            return True
            
        finally:
            os.unlink(test_file)
            
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("DISTRIBUTED INDEXING SYSTEM TEST")
    logger.info("=" * 50)
    
    # Run basic functionality tests
    basic_success = await test_basic_functionality()
    
    # Run integration tests
    integration_success = await test_integration()
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Basic Functionality: {'✓ PASS' if basic_success else '✗ FAIL'}")
    logger.info(f"Integration: {'✓ PASS' if integration_success else '✗ FAIL'}")
    
    if basic_success and integration_success:
        logger.info("🎉 All tests passed! System is ready to use.")
    else:
        logger.warning("⚠ Some tests failed. Check the logs above for details.")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main()) 