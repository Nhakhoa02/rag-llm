# Distributed RAG System

## Overview

This project implements a scalable, fault-tolerant distributed system for Retrieval-Augmented Generation (RAG) using a peer-to-peer distributed vector store. The system supports document ingestion, AI-driven collection management, distributed vector storage with sharding and replication, autoscaling, and robust search capabilities.

## Key Features

- **Peer-to-Peer Distributed Architecture:** All nodes are equal peers; no single point of failure or master node.
- **Sharding & Replication:** Data is partitioned into shards and each shard is replicated to multiple nodes for scalability and fault tolerance.
- **AI-Driven Collection Management:** Documents are automatically assigned to semantic collections (indexes) using an AI model.
- **Autoscaling:** The system automatically adds or removes nodes based on health and load.
- **Health Monitoring:** Continuous health checks ensure only healthy nodes are used.
- **FastAPI Coordinator:** Central API for uploads, search, and cluster management.

## Architecture Diagram

```
flowchart TD
    subgraph Cluster[Distributed Vector Store Cluster]
        Node1["Node 1\n(shard 1, shard 2, ...)"]
        Node2["Node 2\n(shard 2, shard 3, ...)"]
        Node3["Node 3\n(shard 1, shard 3, ...)"]
        NodeN["Node N\n(shard X, shard Y, ...)"]
    end
    API["FastAPI Coordinator\n(Upload, Search, Autoscaling)"]
    User["User\n(Client/Web UI)"]
    API <--> Cluster
    User --> API
    Node1 <--> Node2
    Node2 <--> Node3
    Node3 <--> NodeN
    NodeN <--> Node1
    note1["Each node stores a subset of data (shards)\nEach shard is replicated to multiple nodes"]
    Cluster --- note1
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start vector node servers:**
   ```bash
   python start_node1.py
   python start_node2.py
   python start_node3.py
   # (Add more nodes as needed)
   ```

3. **Start the FastAPI coordinator:**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

4. **Upload documents and search via the API or web UI.**

## Usage

- **Upload documents:**
  - Use the `/upload` endpoint to add new documents. The system will process, chunk, embed, and index them.
- **Search:**
  - Use the `/search` endpoint to query for relevant documents by semantic similarity.
- **Cluster management:**
  - Use `/cluster/status` and `/autoscaling/status` to monitor the system.
  - Use `/autoscaling/scale-up` and `/autoscaling/scale-down` to manually trigger scaling events.

## Notes

- **No Cassandra, MinIO, or Elasticsearch:**
  - The current system does not use these backends. All distributed storage is handled by the custom vector store logic.
- **Extensible:**
  - The architecture can be extended to support additional storage backends or features as needed.

## License

MIT 