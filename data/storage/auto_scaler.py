"""
Simple Auto-Scaling System for Distributed RAG
"""

import asyncio
import time
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .distributed_storage_manager import DistributedStorageManager
from .distributed_vector_store import VectorNode
from .vector_node_server import create_node_server
from core.models.base import BaseDocument
from core.models.document import Document
from core.utils.logging import get_logger
from core.utils.metrics import monitor_function
from config.config import settings

logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ScalingThresholds:
    """Configurable thresholds for scaling decisions."""
    cpu_threshold_high: float = 0.75
    memory_threshold_high: float = 0.80
    storage_threshold_high: float = 0.85
    latency_threshold_high: float = 1000.0
    error_rate_threshold_high: float = 0.05
    cpu_threshold_low: float = 0.30
    memory_threshold_low: float = 0.40
    storage_threshold_low: float = 0.50
    latency_threshold_low: float = 100.0
    error_rate_threshold_low: float = 0.01
    min_nodes: int = 3
    max_nodes: int = 10
    scale_up_cooldown: int = 300
    scale_down_cooldown: int = 600
    health_check_interval: int = 30

class AutoScaler:
    """Simple auto-scaler for distributed vector storage."""
    
    def __init__(self, storage_manager: DistributedStorageManager, thresholds: Optional[ScalingThresholds] = None):
        self.storage_manager = storage_manager
        self.thresholds = thresholds or ScalingThresholds()
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.scaling_history: List[Dict] = []
        self.is_running = False
        self.node_processes: Dict[str, subprocess.Popen] = {}  # Track running node processes
        
    async def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        self.is_running = True
        logger.info("Starting auto-scaler monitoring")
        
        while self.is_running:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.thresholds.health_check_interval)
            except Exception as e:
                logger.error(f"Auto-scaler monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def stop_monitoring(self):
        """Stop the auto-scaling monitoring loop."""
        self.is_running = False
        logger.info("Stopped auto-scaler monitoring")
        
        # Stop all running node processes
        for node_id, process in self.node_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Stopped node process: {node_id}")
            except Exception as e:
                logger.error(f"Error stopping node {node_id}: {e}")
    
    async def _monitoring_cycle(self):
        """Single monitoring cycle."""
        try:
            # Get cluster status
            cluster_status = await self.storage_manager.get_cluster_status()
            node_count = cluster_status.get("total_nodes", 0)
            healthy_nodes = cluster_status.get("healthy_nodes", 0)
            
            # Simple scaling logic
            current_time = time.time()
            
            # Scale up if healthy nodes are low
            if (healthy_nodes < self.thresholds.min_nodes and 
                node_count < self.thresholds.max_nodes and
                current_time - self.last_scale_up > self.thresholds.scale_up_cooldown):
                await self._scale_up("Low healthy nodes", node_count)
            
            # Scale down if we have excess nodes
            elif (node_count > self.thresholds.min_nodes and
                  current_time - self.last_scale_down > self.thresholds.scale_down_cooldown):
                await self._scale_down("Excess capacity", node_count)
                
        except Exception as e:
            logger.error(f"Monitoring cycle error: {e}")
    
    async def _scale_up(self, reason: str, current_nodes: int):
        """Start a new node server and add it to the cluster."""
        try:
            new_node_id = f"node{current_nodes + 1}"
            new_port = 8000 + current_nodes + 1
            data_dir = f"./node_data/{new_node_id}"
            
            logger.info(f"Starting new node server: {new_node_id} on port {new_port}")
            
            # Create data directory
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            
            # Start the node server using the dynamic node starter script
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            script_path = os.path.join(project_root, "scripts", "start_dynamic_node.py")
            
            # Start the node server process
            process = subprocess.Popen([
                sys.executable, str(script_path), new_node_id, "localhost", str(new_port), data_dir
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Store the process for later management
            self.node_processes[new_node_id] = process
            
            logger.info(f"Started node server process (PID: {process.pid}) for {new_node_id}")
            
            # Wait longer for the server to start and verify it's running
            logger.info(f"Waiting for {new_node_id} to start up...")
            await asyncio.sleep(5)  # Increased from 3 to 5 seconds
            
            # Verify the process is still running
            if process.poll() is not None:
                # Process exited, get the output to see what went wrong
                stdout, stderr = process.communicate()
                logger.error(f"Node server process for {new_node_id} exited with code {process.poll()}")
                if stdout:
                    logger.error(f"Node {new_node_id} stdout: {stdout}")
                if stderr:
                    logger.error(f"Node {new_node_id} stderr: {stderr}")
                del self.node_processes[new_node_id]
                return
            
            # Try to connect to the node to verify it's ready
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{new_port}/health", timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"Node {new_node_id} is responding to health checks")
                        else:
                            logger.warning(f"Node {new_node_id} returned status {response.status}")
            except Exception as e:
                logger.warning(f"Node {new_node_id} not yet ready: {e}")
                # Continue anyway - the health check system will handle it
            
            # Create VectorNode and add to distributed system
            new_node = VectorNode(id=new_node_id, host="localhost", port=new_port)
            success = await self.storage_manager.add_node(new_node)
            
            if success:
                self.last_scale_up = time.time()
                logger.info(f"Successfully scaled UP: Started and added node {new_node_id}")
                
                self.scaling_history.append({
                    "timestamp": time.time(),
                    "action": "scale_up",
                    "reason": reason,
                    "node_id": new_node_id,
                    "port": new_port,
                    "process_id": process.pid
                })
            else:
                logger.error(f"Failed to scale UP: Could not add node {new_node_id} to distributed system")
                # Clean up the process if adding to distributed system failed
                process.terminate()
                del self.node_processes[new_node_id]
                
        except Exception as e:
            logger.error(f"Error during scale UP: {e}")
            # Clean up any partially created processes
            if 'process' in locals() and new_node_id in self.node_processes:
                process.terminate()
                del self.node_processes[new_node_id]
    
    async def _scale_down(self, reason: str, current_nodes: int):
        """Stop a node server and remove it from the cluster."""
        try:
            cluster_status = await self.storage_manager.get_cluster_status()
            nodes = cluster_status.get("nodes", [])
            
            if nodes:
                # Remove the last node
                node_to_remove = nodes[-1]["id"]
                
                # Stop the node process if we're managing it
                if node_to_remove in self.node_processes:
                    process = self.node_processes[node_to_remove]
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                        logger.info(f"Stopped node process: {node_to_remove}")
                        del self.node_processes[node_to_remove]
                    except Exception as e:
                        logger.error(f"Error stopping node process {node_to_remove}: {e}")
                
                # Remove from distributed system
                success = await self.storage_manager.remove_node(node_to_remove)
                
                if success:
                    self.last_scale_down = time.time()
                    logger.info(f"Successfully scaled DOWN: Stopped and removed node {node_to_remove}")
                    
                    self.scaling_history.append({
                        "timestamp": time.time(),
                        "action": "scale_down",
                        "reason": reason,
                        "node_id": node_to_remove
                    })
                else:
                    logger.error(f"Failed to scale DOWN: Could not remove node {node_to_remove}")
                    
        except Exception as e:
            logger.error(f"Error during scale DOWN: {e}")
    
    def get_scaling_status(self) -> Dict:
        """Get current scaling status and history."""
        return {
            "is_running": self.is_running,
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "scaling_history": self.scaling_history[-10:],
            "running_processes": {node_id: process.pid for node_id, process in self.node_processes.items()},
            "current_metrics": None,
            "thresholds": {
                "cpu_threshold_high": self.thresholds.cpu_threshold_high,
                "memory_threshold_high": self.thresholds.memory_threshold_high,
                "storage_threshold_high": self.thresholds.storage_threshold_high,
                "latency_threshold_high": self.thresholds.latency_threshold_high,
                "error_rate_threshold_high": self.thresholds.error_rate_threshold_high,
                "cpu_threshold_low": self.thresholds.cpu_threshold_low,
                "memory_threshold_low": self.thresholds.memory_threshold_low,
                "storage_threshold_low": self.thresholds.storage_threshold_low,
                "latency_threshold_low": self.thresholds.latency_threshold_low,
                "error_rate_threshold_low": self.thresholds.error_rate_threshold_low,
                "min_nodes": self.thresholds.min_nodes,
                "max_nodes": self.thresholds.max_nodes
            }
        }
    
    def update_thresholds(self, new_thresholds: ScalingThresholds):
        """Update scaling thresholds dynamically."""
        self.thresholds = new_thresholds
        logger.info("Updated auto-scaling thresholds")

def create_auto_scaler(storage_manager: DistributedStorageManager, 
                      custom_thresholds: Optional[ScalingThresholds] = None) -> AutoScaler:
    """Create an auto-scaler with the specified configuration."""
    return AutoScaler(storage_manager, custom_thresholds) 