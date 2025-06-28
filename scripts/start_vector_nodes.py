#!/usr/bin/env python3
"""
Start Vector Storage Nodes

Script to start multiple vector storage nodes for the distributed system.
This demonstrates true distributed architecture with fault tolerance.
"""

import asyncio
import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from typing import List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.vector_node_server import create_node_server

def start_node(node_id: str, host: str, port: int, data_dir: str = None):
    """Start a vector storage node."""
    if data_dir is None:
        data_dir = f"./node_data/{node_id}"
    
    print(f"Starting vector node {node_id} on {host}:{port}")
    print(f"Data directory: {data_dir}")
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Start the node server
    server = create_node_server(node_id, host, port, data_dir)
    server.run()

async def start_node_server(node_id: int, port: int, host: str = "localhost") -> None:
    """Start a vector node server."""
    try:
        # Import here to avoid circular imports
        from data.storage.vector_node_server import create_node_server
        
        server = await create_node_server(node_id, port, host)
        print(f"✅ Node {node_id} started on {host}:{port}")
        return server
    except Exception as e:
        print(f"❌ Failed to start Node {node_id}: {e}")
        raise

def main():
    """Main function to start multiple nodes."""
    print("🚀 Starting Distributed Vector Storage Nodes")
    print("=" * 50)
    
    # Node configurations
    nodes = [
        {"id": "node1", "host": "localhost", "port": 8001},
        {"id": "node2", "host": "localhost", "port": 8002},
        {"id": "node3", "host": "localhost", "port": 8003},
    ]
    
    # Start nodes in separate processes
    processes = []
    
    try:
        for node in nodes:
            # Start each node in a separate process
            process = subprocess.Popen([
                sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.getcwd()}')
from data.storage.vector_node_server import create_node_server
server = create_node_server('{node['id']}', '{node['host']}', {node['port']}', './node_data/{node['id']}')
server.run()
"""
            ])
            processes.append(process)
            print(f"✅ Started {node['id']} (PID: {process.pid})")
            time.sleep(1)  # Small delay between starts
        
        print("\n🎉 All nodes started successfully!")
        print("📊 Node Status:")
        print("   - node1: http://localhost:8001")
        print("   - node2: http://localhost:8002")
        print("   - node3: http://localhost:8003")
        print("\n🔍 Health Check URLs:")
        print("   - http://localhost:8001/health")
        print("   - http://localhost:8002/health")
        print("   - http://localhost:8003/health")
        print("\n⏹️  Press Ctrl+C to stop all nodes")
        
        # Wait for all processes
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all nodes...")
        for process in processes:
            process.terminate()
            process.wait()
        print("✅ All nodes stopped")

if __name__ == "__main__":
    main() 