# Auto-Scaling System for Distributed RAG

## Overview

The auto-scaling system provides intelligent, automatic scaling capabilities for the distributed vector storage system. It monitors system performance metrics in real-time and automatically adds or removes nodes to maintain optimal performance, availability, and cost efficiency.

## 🎯 Key Features

- **Real-time Monitoring**: Continuously tracks CPU, memory, storage, latency, and error rates
- **Intelligent Scaling**: Automatically scales up when performance degrades and scales down when resources are underutilized
- **Configurable Thresholds**: Customizable scaling triggers for different environments
- **Fault Tolerance**: Maintains minimum node requirements and handles node failures gracefully
- **Manual Controls**: Override automatic scaling with manual scale up/down operations
- **Scaling History**: Complete audit trail of all scaling decisions and actions
- **Performance Analytics**: Detailed metrics and insights for capacity planning

## 🏗️ Architecture

### Components

1. **AutoScaler**: Main orchestrator that monitors metrics and makes scaling decisions
2. **ScalingMetrics**: Data structure containing current system performance metrics
3. **ScalingThresholds**: Configurable parameters that determine when to scale
4. **DistributedStorageManager**: Interface to the distributed vector storage system
5. **Performance Monitoring**: Real-time collection of system metrics

### Scaling Decision Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitor       │───▶│   Analyze        │───▶│   Execute       │
│   Metrics       │    │   Conditions     │    │   Scaling       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ • CPU Usage     │    │ • Scale Up       │    │ • Add Node      │
│ • Memory Usage  │    │ • Scale Down     │    │ • Remove Node   │
│ • Storage Usage │    │ • Maintain       │    │ • Rebalance     │
│ • Latency       │    │ • Cooldown Check │    │ • Update State  │
│ • Error Rate    │    │ • Thresholds     │    │ • Log Actions   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Scaling Metrics

### Primary Metrics

| Metric | Description | Scale Up Trigger | Scale Down Trigger |
|--------|-------------|------------------|-------------------|
| **CPU Usage** | Average CPU utilization across nodes | > 75% | < 30% |
| **Memory Usage** | Average memory utilization across nodes | > 80% | < 40% |
| **Storage Usage** | Average storage utilization across nodes | > 85% | < 50% |
| **Search Latency** | Average search response time | > 500ms | < 100ms |
| **Error Rate** | Percentage of failed requests | > 5% | < 1% |
| **Healthy Nodes** | Ratio of healthy to total nodes | < 80% | N/A |

### Secondary Metrics

- **Query Throughput**: Requests per second
- **Average Vectors per Node**: Load distribution
- **Node Load**: Individual node performance
- **Shard Distribution**: Data distribution across nodes

## ⚙️ Configuration

### Default Thresholds

```python
ScalingThresholds(
    # Scale UP thresholds
    cpu_threshold_high=0.75,        # 75% CPU usage
    memory_threshold_high=0.80,     # 80% memory usage
    storage_threshold_high=0.85,    # 85% storage usage
    latency_threshold_high=500.0,   # 500ms search latency
    error_rate_threshold_high=0.05, # 5% error rate
    
    # Scale DOWN thresholds
    cpu_threshold_low=0.30,         # 30% CPU usage
    memory_threshold_low=0.40,      # 40% memory usage
    storage_threshold_low=0.50,     # 50% storage usage
    latency_threshold_low=100.0,    # 100ms search latency
    error_rate_threshold_low=0.01,  # 1% error rate
    
    # Node management
    min_nodes=3,                    # Minimum nodes
    max_nodes=10,                   # Maximum nodes
    scale_up_cooldown=300,          # 5 minutes between scale ups
    scale_down_cooldown=600,        # 10 minutes between scale downs
    health_check_interval=30        # 30 seconds monitoring interval
)
```

### Environment-Specific Configurations

#### Development Environment
```python
# Aggressive scaling for testing
ScalingThresholds(
    cpu_threshold_high=0.3,         # Scale up quickly
    memory_threshold_high=0.4,
    latency_threshold_high=200.0,
    min_nodes=2,
    max_nodes=5,
    scale_up_cooldown=30,           # Short cooldown
    scale_down_cooldown=60
)
```

#### Production Environment
```python
# Conservative scaling for stability
ScalingThresholds(
    cpu_threshold_high=0.85,        # Higher thresholds
    memory_threshold_high=0.90,
    latency_threshold_high=1000.0,
    min_nodes=5,
    max_nodes=20,
    scale_up_cooldown=600,          # Longer cooldown
    scale_down_cooldown=1800
)
```

## 🔄 Scaling Triggers

### When to Scale UP

The system automatically scales up when **any** of the following conditions are met:

1. **High Resource Usage**
   - CPU usage > threshold for 2 consecutive checks
   - Memory usage > threshold for 2 consecutive checks
   - Storage usage > threshold for 2 consecutive checks

2. **Performance Degradation**
   - Search latency > threshold for 3 consecutive checks
   - Error rate > threshold for 2 consecutive checks

3. **Availability Issues**
   - Healthy nodes ratio < 80%
   - Node failures detected

4. **Capacity Constraints**
   - Average vectors per node > 1,000,000
   - Storage approaching limits

### When to Scale DOWN

The system scales down when **multiple** conditions are met simultaneously:

1. **Low Resource Usage**
   - CPU usage < threshold for 5 consecutive checks
   - Memory usage < threshold for 5 consecutive checks
   - Storage usage < threshold for 5 consecutive checks

2. **Good Performance**
   - Search latency < threshold for 5 consecutive checks
   - Error rate < threshold for 5 consecutive checks

3. **Safety Checks**
   - At least 3 low indicators present
   - Minimum node count maintained
   - Cooldown period elapsed

## 🚀 Scaling Process

### Scale UP Process

1. **Detection**: Monitor detects scaling conditions
2. **Validation**: Verify scaling is allowed (cooldown, max nodes)
3. **Node Creation**: 
   - Generate new node ID and port
   - Create VectorNode instance
   - Add to distributed storage manager
4. **Health Check**: Wait for new node to become healthy
5. **Data Rebalancing**: Redistribute shards to new node
6. **State Update**: Update cluster state and metrics
7. **Logging**: Record scaling action and reason

### Scale DOWN Process

1. **Detection**: Monitor detects low utilization
2. **Validation**: Verify scaling is safe (min nodes, cooldown)
3. **Node Selection**: Choose least loaded node for removal
4. **Data Migration**: Move shards from target node to others
5. **Node Removal**: Remove node from distributed system
6. **State Update**: Update cluster state and metrics
7. **Logging**: Record scaling action and reason

## 📈 Performance Monitoring

### Real-time Metrics Collection

```python
# Metrics collected every 30 seconds
ScalingMetrics(
    cpu_usage=0.65,           # Average across all nodes
    memory_usage=0.72,        # Average across all nodes
    storage_usage=0.45,       # Average across all nodes
    search_latency=125.5,     # Average search time in ms
    query_throughput=45.2,    # Queries per second
    error_rate=0.002,         # 0.2% error rate
    node_count=4,             # Current node count
    healthy_nodes=4,          # Healthy node count
    total_vectors=1500000,    # Total vectors in system
    avg_vectors_per_node=375000  # Average per node
)
```

### Historical Data

- **Metrics History**: Last 100 data points stored
- **Scaling History**: Complete audit trail of scaling actions
- **Performance Trends**: Analysis of system behavior over time

## 🛠️ API Endpoints

### Auto-Scaling Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/autoscaling/status` | GET | Get current auto-scaling status |
| `/autoscaling/start` | POST | Start auto-scaling monitoring |
| `/autoscaling/stop` | POST | Stop auto-scaling monitoring |
| `/autoscaling/thresholds` | POST | Update scaling thresholds |
| `/autoscaling/scale-up` | POST | Manually trigger scale up |
| `/autoscaling/scale-down` | POST | Manually trigger scale down |

### Example Usage

```bash
# Start auto-scaling
curl -X POST http://localhost:8000/autoscaling/start

# Update thresholds
curl -X POST http://localhost:8000/autoscaling/thresholds \
  -H "Content-Type: application/json" \
  -d '{"cpu_threshold_high": 0.8, "max_nodes": 15}'

# Check status
curl http://localhost:8000/autoscaling/status

# Manual scale up
curl -X POST http://localhost:8000/autoscaling/scale-up
```

## 🎯 Best Practices

### Configuration Guidelines

1. **Start Conservative**: Begin with higher thresholds and adjust based on performance
2. **Monitor Cooldowns**: Set appropriate cooldown periods to prevent thrashing
3. **Set Realistic Limits**: Configure min/max nodes based on your infrastructure
4. **Test Scaling**: Use the demo script to test scaling behavior

### Production Considerations

1. **Resource Monitoring**: Monitor actual resource usage vs. thresholds
2. **Scaling History**: Review scaling events to optimize thresholds
3. **Cost Management**: Balance performance with infrastructure costs
4. **Disaster Recovery**: Ensure minimum nodes for fault tolerance

### Troubleshooting

1. **Scaling Not Triggering**: Check if cooldown periods are active
2. **Frequent Scaling**: Adjust thresholds or increase cooldown periods
3. **Node Failures**: Verify node health and network connectivity
4. **Performance Issues**: Review metrics and adjust thresholds accordingly

## 🔍 Monitoring and Analytics

### Scaling Dashboard

The system provides comprehensive monitoring through:

- **Real-time Status**: Current scaling state and metrics
- **Historical Data**: Performance trends over time
- **Scaling Events**: Complete audit trail of all actions
- **Threshold Analysis**: Effectiveness of current settings

### Key Performance Indicators

1. **Scaling Frequency**: How often scaling occurs
2. **Response Time**: Time from trigger to completion
3. **Success Rate**: Percentage of successful scaling operations
4. **Cost Impact**: Resource utilization before/after scaling

## 🚀 Getting Started

### Quick Start

1. **Start the System**:
   ```bash
   # Start vector nodes
   python start_node1.py
   python start_node2.py
   python start_node3.py
   
   # Start main API server
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Start Auto-Scaling**:
   ```bash
   curl -X POST http://localhost:8000/autoscaling/start
   ```

3. **Run Demo**:
   ```bash
   python demo_autoscaling.py
   ```

### Customization

1. **Adjust Thresholds**: Modify scaling triggers for your environment
2. **Set Node Limits**: Configure min/max nodes based on infrastructure
3. **Monitor Performance**: Use the provided endpoints to track system behavior
4. **Optimize Settings**: Fine-tune based on actual usage patterns

## 🔮 Future Enhancements

### Planned Features

1. **Predictive Scaling**: ML-based prediction of scaling needs
2. **Cost Optimization**: Automatic cost-aware scaling decisions
3. **Multi-Region Support**: Cross-region scaling and load balancing
4. **Advanced Analytics**: Detailed performance insights and recommendations
5. **Integration APIs**: Connect with external monitoring systems

### Advanced Scenarios

1. **Scheduled Scaling**: Scale based on time-based patterns
2. **Event-Driven Scaling**: Scale based on external events
3. **Custom Metrics**: Support for application-specific metrics
4. **Policy-Based Scaling**: Complex scaling rules and policies

---

## 📚 Additional Resources

- [Distributed Architecture Documentation](architecture.md)
- [API Reference](../README.md#api-endpoints)
- [Performance Tuning Guide](performance.md)
- [Troubleshooting Guide](troubleshooting.md)

For questions and support, please refer to the project documentation or create an issue in the repository. 