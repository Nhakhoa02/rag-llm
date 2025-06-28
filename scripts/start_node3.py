#!/usr/bin/env python3
"""
Start script for Node 3 of the distributed vector store.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.vector_node_server import create_node_server

if __name__ == "__main__":
    print("🚀 Starting Vector Node 3 on localhost:8003")
    server = create_node_server("node3", "localhost", 8003)
    server.run() 