#!/usr/bin/env python3
"""
Dynamic Node Starter

This script can start any vector node with given parameters.
Usage: python start_dynamic_node.py <node_id> <host> <port> [data_dir]
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.vector_node_server import create_node_server

def start_node(node_id: str, host: str = "localhost", port: int = 8001, data_dir: str = ""):
    """Start a vector node server."""
    if not data_dir:
        data_dir = f"./node_data/{node_id}"
    
    print(f"Starting Vector Node {node_id} on {host}:{port}")
    print(f"Data directory: {data_dir}")
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create and start the server
    server = create_node_server(node_id, host, port, data_dir)
    server.run()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python start_dynamic_node.py <node_id> [host] [port] [data_dir]")
        print("Example: python start_dynamic_node.py node4 localhost 8004")
        sys.exit(1)
    
    node_id = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8001
    data_dir = sys.argv[4] if len(sys.argv) > 4 else f"./node_data/{node_id}"
    
    try:
        start_node(node_id, host, port, data_dir)
    except KeyboardInterrupt:
        print(f"\nStopping node {node_id}...")
    except Exception as e:
        print(f"Error starting node {node_id}: {e}")
        sys.exit(1) 