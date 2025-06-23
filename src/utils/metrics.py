"""
Metrics collection and monitoring utilities.
"""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import structlog

from .logging import get_logger


class MetricsCollector:
    """Collector for system metrics and performance monitoring."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Request metrics
        self.request_counter = Counter(
            'rag_requests_total',
            'Total number of requests',
            ['service', 'endpoint', 'method', 'status']
        )
        
        self.request_duration = Histogram(
            'rag_request_duration_seconds',
            'Request duration in seconds',
            ['service', 'endpoint', 'method']
        )
        
        # Processing metrics
        self.processing_duration = Histogram(
            'rag_processing_duration_seconds',
            'Data processing duration in seconds',
            ['service', 'operation', 'data_type']
        )
        
        self.processing_counter = Counter(
            'rag_processing_total',
            'Total number of processing operations',
            ['service', 'operation', 'data_type', 'status']
        )
        
        # Storage metrics
        self.storage_operations = Counter(
            'rag_storage_operations_total',
            'Total number of storage operations',
            ['service', 'operation', 'storage_type', 'status']
        )
        
        self.storage_duration = Histogram(
            'rag_storage_duration_seconds',
            'Storage operation duration in seconds',
            ['service', 'operation', 'storage_type']
        )
        
        # Indexing metrics
        self.indexing_operations = Counter(
            'rag_indexing_operations_total',
            'Total number of indexing operations',
            ['service', 'operation', 'index_type', 'status']
        )
        
        self.indexing_duration = Histogram(
            'rag_indexing_duration_seconds',
            'Indexing operation duration in seconds',
            ['service', 'operation', 'index_type']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'rag_active_connections',
            'Number of active connections',
            ['service', 'connection_type']
        )
        
        self.queue_size = Gauge(
            'rag_queue_size',
            'Current queue size',
            ['service', 'queue_name']
        )
        
        self.memory_usage = Gauge(
            'rag_memory_usage_bytes',
            'Memory usage in bytes',
            ['service']
        )
        
        # Error metrics
        self.error_counter = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['service', 'error_type', 'operation']
        )
        
        # Custom metrics storage
        self._custom_metrics: Dict[str, Any] = {}
    
    def record_request(self, service: str, endpoint: str, method: str, 
                      status: int, duration: float) -> None:
        """Record request metrics."""
        self.request_counter.labels(
            service=service,
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
        
        self.request_duration.labels(
            service=service,
            endpoint=endpoint,
            method=method
        ).observe(duration)
        
        self.logger.debug(
            "Request recorded",
            service=service,
            endpoint=endpoint,
            method=method,
            status=status,
            duration=duration
        )
    
    def record_processing(self, service: str, operation: str, data_type: str,
                         duration: float, status: str = "success") -> None:
        """Record processing metrics."""
        self.processing_duration.labels(
            service=service,
            operation=operation,
            data_type=data_type
        ).observe(duration)
        
        self.processing_counter.labels(
            service=service,
            operation=operation,
            data_type=data_type,
            status=status
        ).inc()
    
    def record_storage_operation(self, service: str, operation: str, 
                               storage_type: str, duration: float, 
                               status: str = "success") -> None:
        """Record storage operation metrics."""
        self.storage_duration.labels(
            service=service,
            operation=operation,
            storage_type=storage_type
        ).observe(duration)
        
        self.storage_operations.labels(
            service=service,
            operation=operation,
            storage_type=storage_type,
            status=status
        ).inc()
    
    def record_indexing_operation(self, service: str, operation: str,
                                index_type: str, duration: float,
                                status: str = "success") -> None:
        """Record indexing operation metrics."""
        self.indexing_duration.labels(
            service=service,
            operation=operation,
            index_type=index_type
        ).observe(duration)
        
        self.indexing_operations.labels(
            service=service,
            operation=operation,
            index_type=index_type,
            status=status
        ).inc()
    
    def set_active_connections(self, service: str, connection_type: str, count: int) -> None:
        """Set active connections count."""
        self.active_connections.labels(
            service=service,
            connection_type=connection_type
        ).set(count)
    
    def set_queue_size(self, service: str, queue_name: str, size: int) -> None:
        """Set queue size."""
        self.queue_size.labels(
            service=service,
            queue_name=queue_name
        ).set(size)
    
    def set_memory_usage(self, service: str, usage_bytes: int) -> None:
        """Set memory usage."""
        self.memory_usage.labels(service=service).set(usage_bytes)
    
    def record_error(self, service: str, error_type: str, operation: str) -> None:
        """Record error metrics."""
        self.error_counter.labels(
            service=service,
            error_type=error_type,
            operation=operation
        ).inc()
    
    def add_custom_metric(self, name: str, metric: Any) -> None:
        """Add custom metric."""
        self._custom_metrics[name] = metric
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest()
    
    def get_custom_metric(self, name: str) -> Optional[Any]:
        """Get custom metric by name."""
        return self._custom_metrics.get(name)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def monitor_function(service: str, operation: str, data_type: str = "unknown"):
    """Decorator to monitor function execution time and success/failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_processing(
                    service=service,
                    operation=operation,
                    data_type=data_type,
                    duration=duration,
                    status="success"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_processing(
                    service=service,
                    operation=operation,
                    data_type=data_type,
                    duration=duration,
                    status="error"
                )
                metrics_collector.record_error(
                    service=service,
                    error_type=type(e).__name__,
                    operation=operation
                )
                raise
        return wrapper
    return decorator


def monitor_request(service: str):
    """Decorator to monitor API request metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_request(
                    service=service,
                    endpoint=func.__name__,
                    method="POST",  # Default, should be extracted from request
                    status=200,
                    duration=duration
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_request(
                    service=service,
                    endpoint=func.__name__,
                    method="POST",
                    status=500,
                    duration=duration
                )
                metrics_collector.record_error(
                    service=service,
                    error_type=type(e).__name__,
                    operation=func.__name__
                )
                raise
        return wrapper
    return decorator 