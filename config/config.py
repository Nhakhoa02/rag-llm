"""
Configuration management for the distributed indexing system.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
print(env_path)
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading from env.example if .env doesn't exist
    example_env_path = Path(__file__).parent.parent / "env.example"
    if example_env_path.exists():
        load_dotenv(example_env_path)


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = ConfigDict(extra="allow")
    
    # Qdrant settings
    qdrant_host: str = Field(default="localhost", validation_alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    
    # Cassandra settings (optional)
    cassandra_hosts: List[str] = Field(default=["localhost:9042"], validation_alias="CASSANDRA_HOSTS")
    cassandra_keyspace: str = Field(default="rag_llm", validation_alias="CASSANDRA_KEYSPACE")
    cassandra_username: Optional[str] = Field(default=None, validation_alias="CASSANDRA_USERNAME")
    cassandra_password: Optional[str] = Field(default=None, validation_alias="CASSANDRA_PASSWORD")
    
    # Redis settings (optional)
    redis_host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, validation_alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    
    # Elasticsearch settings (optional)
    elasticsearch_host: str = Field(default="localhost", validation_alias="ELASTICSEARCH_HOST")
    elasticsearch_port: int = Field(default=9200, validation_alias="ELASTICSEARCH_PORT")
    elasticsearch_username: Optional[str] = Field(default=None, validation_alias="ELASTICSEARCH_USERNAME")
    elasticsearch_password: Optional[str] = Field(default=None, validation_alias="ELASTICSEARCH_PASSWORD")
    
    # MinIO settings (optional)
    minio_endpoint: str = Field(default="localhost:9000", validation_alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", validation_alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", validation_alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="rag-llm", validation_alias="MINIO_BUCKET")
    
    @validator("cassandra_hosts", pre=True)
    def parse_cassandra_hosts(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [host.strip() for host in v.split(",")]
        return v


class APISettings(BaseSettings):
    """API server configuration settings."""
    
    model_config = ConfigDict(extra="allow")
    
    host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    port: int = Field(default=8000, validation_alias="API_PORT")
    workers: int = Field(default=4, validation_alias="API_WORKERS")
    reload: bool = Field(default=True, validation_alias="API_RELOAD")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        validation_alias="CORS_ORIGINS"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        validation_alias="CORS_METHODS"
    )
    cors_headers: List[str] = Field(
        default=["*"],
        validation_alias="CORS_HEADERS"
    )
    
    @validator("cors_origins", "cors_methods", "cors_headers", pre=True)
    def parse_list_fields(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [item.strip() for item in v.split(",")]
        return v


class ProcessingSettings(BaseSettings):
    """Data processing configuration settings."""
    
    model_config = ConfigDict(extra="allow")
    
    vector_dimension: int = Field(default=768, validation_alias="VECTOR_DIMENSION")
    chunk_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    max_file_size: int = Field(default=104857600, validation_alias="MAX_FILE_SIZE")  # 100MB
    
    # Gemini AI settings
    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-pro", validation_alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, validation_alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=1000, validation_alias="GEMINI_MAX_TOKENS")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    model_config = ConfigDict(extra="allow")
    
    # Prometheus
    prometheus_host: str = Field(default="localhost", validation_alias="PROMETHEUS_HOST")
    prometheus_port: int = Field(default=9090, validation_alias="PROMETHEUS_PORT")
    
    # Grafana
    grafana_host: str = Field(default="localhost", validation_alias="GRAFANA_HOST")
    grafana_port: int = Field(default=3000, validation_alias="GRAFANA_PORT")
    
    # Jaeger
    jaeger_host: str = Field(default="localhost", validation_alias="JAEGER_HOST")
    jaeger_port: int = Field(default=16686, validation_alias="JAEGER_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = Field(default="json", validation_alias="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, validation_alias="LOG_FILE")


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    model_config = ConfigDict(extra="allow")
    
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        validation_alias="SECRET_KEY"
    )
    encryption_key: str = Field(
        default="your-encryption-key-here-change-in-production",
        validation_alias="ENCRYPTION_KEY"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, validation_alias="RATE_LIMIT_WINDOW")  # seconds


class DeploymentSettings(BaseSettings):
    """Deployment configuration settings."""
    
    model_config = ConfigDict(extra="allow")
    
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    
    # Kubernetes
    k8s_namespace: str = Field(default="rag-llm", validation_alias="K8S_NAMESPACE")
    
    # Docker
    docker_registry: str = Field(default="your-registry.com", validation_alias="DOCKER_REGISTRY")
    docker_image: str = Field(default="rag-llm", validation_alias="DOCKER_IMAGE")
    docker_tag: str = Field(default="latest", validation_alias="DOCKER_TAG")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = ConfigDict(extra="allow")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    processing: ProcessingSettings = ProcessingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    deployment: DeploymentSettings = DeploymentSettings()


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = [
        "GEMINI_API_KEY",
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        return False
    
    print("✅ Environment validation passed!")
    return True


def print_configuration():
    """Print the current configuration (without sensitive data)."""
    print("🔧 Current Configuration:")
    print(f"  Environment: {settings.deployment.environment}")
    print(f"  Debug Mode: {settings.deployment.debug}")
    print(f"  API Host: {settings.api.host}:{settings.api.port}")
    print(f"  Qdrant: {settings.database.qdrant_host}:{settings.database.qdrant_port}")
    print(f"  Vector Dimension: {settings.processing.vector_dimension}")
    print(f"  Chunk Size: {settings.processing.chunk_size}")
    print(f"  Gemini API Key: {'✅ Set' if settings.processing.gemini_api_key else '❌ Not Set'}")


if __name__ == "__main__":
    # Print configuration when run directly
    print_configuration()
    validate_environment() 