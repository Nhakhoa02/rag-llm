#!/usr/bin/env python3
"""
Debug script to test CSV processing directly.
"""

import asyncio
import pandas as pd
import tempfile
import os
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.append('src')

from src.ingestion.csv_processor import CSVProcessor
from src.ai.gemini_client import GeminiClient

async def debug_csv_processing():
    """Debug CSV processing step by step."""
    print("🔍 Debugging CSV Processing...")
    print("=" * 60)
    
    # Create test CSV
    data = {
        'name': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown'],
        'age': [30, 25, 35, 28],
        'salary': [50000.0, 60000.0, 70000.0, 55000.0],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR']
    }
    df = pd.DataFrame(data)
    
    print("📊 Test CSV data:")
    print(df)
    print()
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Initialize CSV processor
        csv_processor = CSVProcessor()
        print("✅ CSV processor initialized")
        
        # Test row-level document creation
        print("\n🔧 Testing row-level document creation...")
        metadata = {
            'filename': 'test_employees.csv',
            'description': 'Test employee data',
            'source': 'debug_test'
        }
        
        row_docs = await csv_processor.create_row_level_documents(df, 'test_employees.csv', metadata)
        
        if row_docs and isinstance(row_docs, list) and len(row_docs) > 0:
            print(f"✅ Created {len(row_docs)} row-level documents")
            print(f"   First document content: {row_docs[0].content[:100]}...")
            print(f"   First document metadata keys: {list(row_docs[0].metadata.keys())}")
        else:
            print("❌ No row-level documents created")
            print(f"   row_docs type: {type(row_docs)}")
            print(f"   row_docs value: {row_docs}")
            return
        
        # Test embedding generation
        print("\n🔧 Testing embedding generation...")
        gemini_client = GeminiClient()
        
        embeddings_created = 0
        for i, doc in enumerate(row_docs[:2]):  # Test first 2 documents
            print(f"   Generating embedding for document {i+1}...")
            embedding = await gemini_client.generate_embeddings(doc.content)
            if embedding:
                embeddings_created += 1
                print(f"   ✅ Embedding generated (length: {len(embedding)})")
            else:
                print(f"   ❌ Failed to generate embedding")
        
        print(f"✅ Generated embeddings for {embeddings_created}/{min(2, len(row_docs))} documents")
        
        # Test collection creation
        print("\n🔧 Testing collection creation...")
        csv_collection_name = f"csv_rows_test_employees"
        print(f"   Collection name: {csv_collection_name}")
        
        # Test the full CSV processing
        print("\n🔧 Testing full CSV processing...")
        csv_document = await csv_processor.process_csv_file(temp_path, metadata, 'test_employees.csv')
        
        if csv_document:
            print(f"✅ CSV document created: {csv_document.id}")
            print(f"   Schema: {csv_document.csv_schema is not None}")
            print(f"   Summary: {csv_document.csv_summary is not None}")
            print(f"   SQLite DB: {csv_document.sqlite_db_path}")
        else:
            print("❌ CSV document creation failed")
        
    except Exception as e:
        print(f"❌ Error during CSV processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print("\n🎉 CSV processing debug completed!")

async def main():
    """Main debug function."""
    print("🚀 CSV Processing Debug")
    print("=" * 60)
    print("This debug script tests:")
    print("1. Row-level document creation")
    print("2. Embedding generation")
    print("3. Collection creation")
    print("4. Full CSV processing")
    print()
    
    await debug_csv_processing()

if __name__ == "__main__":
    asyncio.run(main()) 