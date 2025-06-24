#!/usr/bin/env python3
"""Start Node 3 for distributed vector storage."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.storage.vector_node_server import create_node_server

if __name__ == "__main__":
    print("🚀 Starting Vector Node 3 on localhost:8003")
    server = create_node_server("node3", "localhost", 8003)
    server.run() 