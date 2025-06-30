#!/usr/bin/env python3
"""
Troubleshoot Scaling Issues

This script helps troubleshoot the specific scaling issue where node4 is failing to connect.
"""

import asyncio
import time
import sys
import subprocess
import os
import requests
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_port(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            s.close()
            return False  # Port is available
    except OSError:
        return True  # Port is in use

def check_node_health(port: int) -> dict:
    """Check if a node is responding on a port."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

def start_node_manually(node_id: str, port: int) -> subprocess.Popen:
    """Start a node manually and return the process."""
    print(f"🚀 Starting {node_id} on port {port}...")
    
    data_dir = f"./node_data/{node_id}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Start the node server
    process = subprocess.Popen([
        sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{os.getcwd()}')
from data.storage.vector_node_server import create_node_server
server = create_node_server('{node_id}', 'localhost', {port}, '{data_dir}')
server.run()
"""
    ])
    
    print(f"✅ Started {node_id} (PID: {process.pid})")
    return process

async def troubleshoot_scaling():
    """Troubleshoot the scaling issue."""
    print("🔍 Troubleshooting Scaling Issues")
    print("=" * 50)
    
    # Check the specific port that's failing (8004)
    print("\n📋 Checking port 8004 (node4)...")
    port_8004_in_use = check_port(8004)
    print(f"   Port 8004 in use: {port_8004_in_use}")
    
    if port_8004_in_use:
        print("   ❌ Port 8004 is already in use!")
        print("   💡 This might be why node4 can't start")
    else:
        print("   ✅ Port 8004 is available")
    
    # Check other common ports
    print("\n📋 Checking other node ports...")
    for port in [8001, 8002, 8003, 8004, 8005]:
        in_use = check_port(port)
        health = check_node_health(port)
        status = "🟢 Available" if not in_use else "🔴 In use"
        health_status = health["status"]
        
        print(f"   Port {port}: {status} | Health: {health_status}")
        if health["status"] == "healthy":
            print(f"      Node data: {health['data']}")
        elif health["status"] == "unreachable":
            print(f"      Error: {health['error']}")
    
    # Try to start node4 manually
    print("\n🧪 Testing manual node4 startup...")
    
    # Check if we can start node4
    if not check_port(8004):
        print("   Starting node4 manually...")
        process = start_node_manually("node4", 8004)
        
        # Wait for startup
        print("   ⏳ Waiting for node4 to start...")
        await asyncio.sleep(5)
        
        # Check if process is still running
        if process.poll() is not None:
            print(f"   ❌ Node4 process exited with code {process.poll()}")
        else:
            print("   ✅ Node4 process is still running")
            
            # Check health
            health = check_node_health(8004)
            print(f"   Health check: {health['status']}")
            
            if health['status'] == 'healthy':
                print("   🎉 Node4 is working!")
                print(f"      Load: {health['data'].get('load', 0):.2f}")
                print(f"      Vectors: {health['data'].get('vector_count', 0)}")
            else:
                print(f"   ⚠️  Node4 not healthy: {health['error']}")
            
            # Clean up
            print("   🧹 Stopping test node4...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print("   ✅ Node4 stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print("   ⚠️  Force killed node4")
    else:
        print("   ❌ Port 8004 is in use, can't test startup")
    
    # Check for any hanging processes
    print("\n🔍 Checking for hanging processes...")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.cmdline())
                if 'vector_node_server' in cmdline or 'create_node_server' in cmdline:
                    print(f"   Found node process: PID {proc.pid} - {proc.name()}")
                    print(f"      Cmd: {cmdline[:100]}...")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        print("   ⚠️  psutil not available, can't check processes")
    
    print("\n💡 Troubleshooting Tips:")
    print("   1. Make sure no other processes are using ports 8001-8005")
    print("   2. Check if you have enough system resources")
    print("   3. Verify the node_data directory is writable")
    print("   4. Check the logs for any Python errors")
    print("   5. Try running the diagnosis script: python scripts/diagnose_nodes.py")

if __name__ == "__main__":
    asyncio.run(troubleshoot_scaling()) 