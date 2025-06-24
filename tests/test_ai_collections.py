#!/usr/bin/env python3
"""
Test script for AI-driven collection decision system.
This demonstrates how the AI decides whether to create new collections or use existing ones.
"""

import asyncio
import json
from src.models.base import BaseDocument, DataType
from src.ai.gemini_client import GeminiClient, extract_gemini_text

async def test_ai_collection_decisions():
    """Test the AI collection decision system with different document types."""
    
    # Sample documents with different content
    test_documents = [
        {
            "name": "AI Research Paper",
            "content": "This research paper discusses advances in machine learning algorithms, specifically focusing on deep neural networks and their applications in natural language processing. The study presents novel approaches to transformer architectures and demonstrates improved performance on benchmark datasets.",
            "type": DataType.DOCUMENT
        },
        {
            "name": "Cybersecurity Guide",
            "content": "This comprehensive guide covers cybersecurity best practices for enterprise environments. Topics include network security, encryption protocols, threat detection, incident response procedures, and compliance with security standards like ISO 27001.",
            "type": DataType.DOCUMENT
        },
        {
            "name": "Financial Report",
            "content": "Quarterly financial report showing revenue growth of 15% year-over-year, with detailed analysis of market trends, customer acquisition costs, and profitability metrics. The report includes projections for the next quarter and strategic recommendations.",
            "type": DataType.DOCUMENT
        },
        {
            "name": "Technical Manual",
            "content": "Technical manual for distributed systems architecture, covering topics like load balancing, fault tolerance, data replication, and scalability patterns. Includes implementation examples and best practices for microservices.",
            "type": DataType.DOCUMENT
        }
    ]
    
    # Simulate existing collections
    existing_collections = [
        "index_document",
        "index_technology", 
        "index_research",
        "index_ai",
        "index_business"
    ]
    
    print("🤖 AI-Driven Collection Decision System Test")
    print("=" * 60)
    print(f"Existing collections: {existing_collections}")
    print()
    
    for i, doc_data in enumerate(test_documents, 1):
        print(f"📄 Document {i}: {doc_data['name']}")
        print(f"Content preview: {doc_data['content'][:100]}...")
        
        # Create document object
        document = BaseDocument(
            id=f"test_doc_{i}",
            content=doc_data['content'],
            metadata={"source": "test", "category": "demo"},
            type=doc_data['type']
        )
        
        # Simulate AI analysis (simplified version)
        ai_metadata = {"source": "test", "category": "demo"}
        
        # Call the collection determination function
        from src.api.main import determine_collections
        collections = await determine_collections(document, ai_metadata, existing_collections)
        
        # Analyze results
        new_collections = [col for col in collections if col not in existing_collections]
        reused_collections = [col for col in collections if col in existing_collections]
        
        print(f"✅ AI Decision Results:")
        print(f"   - Reusing existing: {reused_collections}")
        print(f"   - Creating new: {new_collections}")
        print(f"   - Total collections: {len(collections)}")
        print()
        
        # Update existing collections for next iteration
        existing_collections.extend(new_collections)
    
    print("🎯 Final Collection State:")
    print(f"Total collections: {len(existing_collections)}")
    print(f"Collections: {existing_collections}")

if __name__ == "__main__":
    asyncio.run(test_ai_collection_decisions()) 