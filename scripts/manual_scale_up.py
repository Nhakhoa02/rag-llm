#!/usr/bin/env python3
"""
Manual Scale Up Demo

This script demonstrates manually scaling up by starting new node servers.
"""

import asyncio
import time
import sys
import subprocess
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.distributed_storage_manager import create_distributed_storage_manager
from data.storage.distributed_vector_store import VectorNode
from data.storage.vector_node_server import create_node_server
from core.models.document import Document

def start_node_server(node_id: str, host: str, port: int, data_dir: str):
    """Start a node server in a separate process."""
    print(f"🚀 Starting node server: {node_id} on {host}:{port}")
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Start the node server in a separate process
    process = subprocess.Popen([
        sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.getcwd()}')
from data.storage.vector_node_server import create_node_server
server = create_node_server('{node_id}', '{host}', {port}, '{data_dir}')
server.run()
"""
    ])
    
    print(f"✅ Started {node_id} (PID: {process.pid})")
    return process

async def manual_scale_up_demo():
    """Demonstrate manual scaling up."""
    print("🚀 Manual Scale Up Demo")
    print("=" * 40)
    
    # Start with initial nodes
    initial_nodes = [
        VectorNode("node1", "localhost", 8001),
        VectorNode("node2", "localhost", 8002),
    ]
    
    print(f"📋 Initial nodes: {[n.id for n in initial_nodes]}")
    
    # Create storage manager
    storage_manager = create_distributed_storage_manager(
        nodes=initial_nodes,
        replication_factor=2,
        consistency_level="quorum",
        shard_count=4,
        vector_size=384
    )
    
    # Wait for initial nodes to be healthy
    print("\n⏳ Waiting for initial nodes to become healthy...")
    await asyncio.sleep(5)
    
    # Check initial status
    status = await storage_manager.get_cluster_status()
    print(f"✅ Initial status: {status['healthy_nodes']}/{status['total_nodes']} nodes healthy")
    
    # Manually start a new node
    print("\n🆕 Manually starting new node...")
    new_node_id = "node3"
    new_port = 8003
    data_dir = f"./node_data/{new_node_id}"
    
    # Start the node server
    process = start_node_server(new_node_id, "localhost", new_port, data_dir)
    
    # Wait for the server to start
    print("⏳ Waiting for node server to start...")
    await asyncio.sleep(3)
    
    # Add the node to the distributed system
    print("🔗 Adding node to distributed system...")
    new_node = VectorNode(id=new_node_id, host="localhost", port=new_port)
    success = await storage_manager.add_node(new_node)
    
    if success:
        print(f"✅ Successfully added {new_node_id} to distributed system")
        
        # Wait a bit for health checks
        await asyncio.sleep(5)
        
        # Check updated status
        status = await storage_manager.get_cluster_status()
        print(f"✅ Updated status: {status['healthy_nodes']}/{status['total_nodes']} nodes healthy")
        
        # Test data distribution
        print("\n📚 Testing data distribution...")
        collection_name = "scale_test"
        
        # Create collection
        await storage_manager.create_index(collection_name, 384)
        
        # Add test documents
        test_docs = [
            Document(id="doc1", content="This document tests scaling functionality", type="document"),
            Document(id="doc2", content="Another document to verify distribution works", type="document"),
        ]
        
        success = await storage_manager.upsert_documents(collection_name, test_docs)
        if success:
            print("✅ Successfully added documents")
            
            # Search to verify
            results = await storage_manager.search_documents(collection_name, "scaling", limit=5)
            print(f"✅ Search returned {len(results)} results")
        else:
            print("❌ Failed to add documents")
    else:
        print(f"❌ Failed to add {new_node_id} to distributed system")
    
    # Clean up
    print(f"\n🧹 Stopping node {new_node_id}...")
    process.terminate()
    process.wait(timeout=5)
    print(f"✅ Stopped {new_node_id}")
    
    print("\n🎉 Manual scale up demo completed!")

if __name__ == "__main__":
    asyncio.run(manual_scale_up_demo()) 