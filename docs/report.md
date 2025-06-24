# Distributed RAG System: Project Report

## 1. Problem Statement

The objective of this project is to design and implement a scalable, fault-tolerant distributed system for Retrieval-Augmented Generation (RAG). The system must efficiently store, index, and search large volumes of vectorized data (such as document embeddings) across multiple nodes, supporting dynamic scaling, robust data management, and high availability—even in the presence of node failures.

---

## 2. Required Functionality

- **Document Ingestion & Indexing:**
  - Users can upload documents (PDFs, images, etc.), which are processed, chunked, and embedded into vectors.
- **AI-Driven Collection Management:**
  - An AI model dynamically determines which semantic collections (indexes) each document should belong to, supporting both broad and highly specific topics.
- **Distributed Vector Storage:**
  - Data is distributed across multiple nodes using sharding and replication, ensuring both scalability and redundancy.
- **Search & Retrieval:**
  - Users can query the system to retrieve relevant documents from one or more distributed indexes, leveraging vector similarity search.
- **Autoscaling:**
  - The system automatically adds or removes nodes based on health and load, maintaining optimal performance and resource usage.
- **Health Monitoring:**
  - Continuous health checks ensure that only healthy nodes are used for storage and retrieval, and unhealthy nodes are replaced as needed.

---

## 3. System Architecture

- **Peer-to-Peer Distributed Architecture:**
  - All nodes are equal peers; there is no single point of failure or master node.
- **Sharding:**
  - The dataset is partitioned into multiple shards, each stored on different nodes to distribute load and enable horizontal scaling.
- **Replication:**
  - Each shard is replicated to multiple nodes, so data remains available even if some nodes fail.
- **FastAPI Coordinator:**
  - A central API service handles user requests, orchestrates uploads, queries, and manages the cluster (but does not store data itself).
- **Autoscaler:**
  - Monitors cluster health and automatically scales the number of nodes up or down as needed.

### **Architecture Diagram**

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

---

## 4. Achieving Distributed System Characteristics

### **Fault Tolerance**
- **Replication:**
  - Each piece of data is stored on multiple nodes. If a node fails, its data is still available from replicas.
- **Health Checks:**
  - The system continuously monitors node health and reroutes requests away from unhealthy nodes.
- **Auto-Recovery:**
  - The autoscaler can automatically add new nodes to replace failed ones, ensuring the system remains operational.

### **Scalability**
- **Sharding:**
  - Data is partitioned across nodes, allowing the system to handle more data and queries as more nodes are added.
- **Autoscaling:**
  - The system can dynamically add or remove nodes in response to load, ensuring efficient resource usage and performance.
- **Peer-to-Peer Design:**
  - Any node can join or leave the cluster, and the system will rebalance shards and replicas as needed.

---

## 5. Methodology: Sharding and Replication

- **Sharding:**
  - The dataset is split into shards, each assigned to different nodes for parallelism and load balancing.
- **Replication:**
  - Each shard is stored on multiple nodes, providing redundancy and high availability.
- **Peer-to-Peer:**
  - All nodes are equal; there is no master/slave distinction. Nodes communicate and coordinate to maintain data consistency and availability.

---

## 6. Bonus: Solving Other Distributed Challenges

- **Dynamic Index Assignment:**
  - The AI-driven approach allows for flexible, semantic organization of data, supporting evolving and overlapping topics.
- **Manual and Automatic Scaling:**
  - Both user-triggered and automatic scaling are supported for operational flexibility.
- **Debugging and Observability:**
  - The system provides endpoints and logs for monitoring cluster state, node health, and data distribution.

---

## 7. Project Presentation & Results

- **Demonstration:**
  - Show document upload, distributed indexing, search, node failure and recovery, and autoscaling in action.
- **Metrics:**
  - Highlight system scalability (adding nodes increases capacity) and fault tolerance (data remains available after node failures).

---

## 8. Conclusion

This project demonstrates a robust, scalable, and fault-tolerant distributed RAG system using a peer-to-peer architecture with sharding and replication. The system supports dynamic scaling, AI-driven collection management, and high availability, making it suitable for real-world large-scale vector search and retrieval applications. 