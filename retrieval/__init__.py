"""
Multi-Retrieval: Unified召回系统
支持向量召回、BM25召回、分片召回的多路召回框架

Architecture:
    Query -> [Vector Retrieval] -> Top-K vectors
          -> [BM25 Retrieval] -> Top-K keywords
          -> [Shard Retrieval] -> Top-K from each shard
    
    Combine -> Rerank -> Final Top-K
"""

from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever
from .shard_retriever import ShardRetriever
from .unified_retriever import UnifiedRetriever, RetrievalConfig
from .reranker import Reranker

__all__ = [
    "VectorRetriever",
    "BM25Retriever",
    "ShardRetriever",
    "UnifiedRetriever",
    "RetrievalConfig",
    "Reranker",
]
