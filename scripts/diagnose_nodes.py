#!/usr/bin/env python3
"""
Node Diagnosis Script

This script helps diagnose issues with node startup and connectivity.
"""

import asyncio
import time
import sys
import subprocess
import os
import requests
import psutil
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.vector_node_server import create_node_server

def check_port_availability(port: int) -> bool:
    """Check if a port is available."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_processes_on_port(port: int) -> list:
    """Find processes using a specific port."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    processes.append({
                        'pid': proc.pid,
                        'name': proc.name(),
                        'cmdline': ' '.join(proc.cmdline())
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def check_node_health(node_id: str, host: str, port: int) -> dict:
    """Check if a node is healthy."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "healthy",
                "node_id": node_id,
                "load": data.get("load", 0.0),
                "vector_count": data.get("vector_count", 0),
                "uptime": data.get("uptime", 0)
            }
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

def start_test_node(node_id: str, host: str, port: int, data_dir: str) -> subprocess.Popen:
    """Start a test node server."""
    print(f"🚀 Starting test node: {node_id} on {host}:{port}")
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Start the node server
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

async def diagnose_nodes():
    """Run comprehensive node diagnosis."""
    print("🔍 Node Diagnosis Tool")
    print("=" * 50)
    
    # Check common ports
    print("\n📋 Checking common node ports...")
    for port in range(8001, 8010):
        available = check_port_availability(port)
        processes = find_processes_on_port(port)
        
        if available and not processes:
            print(f"   Port {port}: ✅ Available")
        elif processes:
            print(f"   Port {port}: ❌ In use by {len(processes)} process(es)")
            for proc in processes:
                print(f"      PID {proc['pid']}: {proc['name']} - {proc['cmdline'][:50]}...")
        else:
            print(f"   Port {port}: ❌ Not available")
    
    # Test node startup
    print("\n🧪 Testing node startup...")
    test_node_id = "test_node"
    test_port = 8009  # Use a port that's likely available
    test_data_dir = f"./node_data/{test_node_id}"
    
    # Check if test port is available
    if not check_port_availability(test_port):
        print(f"❌ Port {test_port} is not available for testing")
        return
    
    # Start test node
    process = start_test_node(test_node_id, "localhost", test_port, test_data_dir)
    
    # Wait for startup
    print("⏳ Waiting for node to start...")
    await asyncio.sleep(3)
    
    # Check if process is still running
    if process.poll() is not None:
        print(f"❌ Node process exited with code {process.poll()}")
        return
    
    # Check health
    print("🔍 Checking node health...")
    for i in range(5):  # Try for 5 seconds
        health = check_node_health(test_node_id, "localhost", test_port)
        print(f"   Attempt {i+1}: {health['status']}")
        
        if health['status'] == 'healthy':
            print(f"   ✅ Node is healthy!")
            print(f"      Load: {health['load']:.2f}")
            print(f"      Vectors: {health['vector_count']}")
            print(f"      Uptime: {health['uptime']:.1f}s")
            break
        elif i < 4:  # Don't sleep on last iteration
            await asyncio.sleep(1)
    else:
        print(f"   ❌ Node did not become healthy: {health['error']}")
    
    # Clean up
    print(f"\n🧹 Stopping test node...")
    process.terminate()
    try:
        process.wait(timeout=5)
        print("✅ Test node stopped")
    except subprocess.TimeoutExpired:
        process.kill()
        print("⚠️  Force killed test node")
    
    # Check for any remaining processes
    print("\n🔍 Checking for remaining node processes...")
    for port in range(8001, 8010):
        processes = find_processes_on_port(port)
        if processes:
            print(f"   Port {port}: {len(processes)} process(es) still running")
            for proc in processes:
                print(f"      PID {proc['pid']}: {proc['name']}")

if __name__ == "__main__":
    asyncio.run(diagnose_nodes()) 