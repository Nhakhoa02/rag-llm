"""
Distributed Multidimensional Data Indexing & Storage for RAG Systems

A scalable, fault-tolerant, and consistent distributed system for indexing
and storing multidimensional data optimized for Retrieval-Augmented Generation (RAG) applications.
"""

__version__ = "1.0.0"
__author__ = "RAG LLM Team"
__email__ = "team@rag-llm.com"

from .config import settings
from .utils.logging import setup_logging

# Setup logging configuration
setup_logging()

__all__ = [
    "settings",
    "setup_logging",
] 