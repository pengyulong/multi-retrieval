"""
向量召回模块 - Vector Retrieval
支持多种向量数据库后端
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import json
import os


@dataclass
class RetrievalResult:
    """单条召回结果"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "vector"  # vector, bm25, shard


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    backend: str = "memory"  # memory, faiss, milvus, pinecone, weaviate, qdrant
    dimension: int = 768
    metric: str = "cosine"  # cosine, l2, ip
    index_type: str = "flat"  # flat, ivf, hnsw
    
    # FAISS specific
    faiss_index_path: Optional[str] = None
    
    # Milvus specific
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "default"
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west-1"
    pinecone_index: str = "default"
    
    # Qdrant specific
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "default"


class BaseVectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add(self, ids: List[str], vectors: np.ndarray, texts: List[str], 
            metadata: Optional[List[Dict]] = None) -> None:
        """添加向量到存储"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               filter_dict: Optional[Dict] = None) -> List[RetrievalResult]:
        """搜索最相似的向量"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存索引到磁盘"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载索引"""
        pass


class MemoryVectorStore(BaseVectorStore):
    """内存向量存储 (简单实现，用于测试)"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.vectors: Dict[str, np.ndarray] = {}
        self.texts: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add(self, ids: List[str], vectors: np.ndarray, texts: List[str],
            metadata: Optional[List[Dict]] = None) -> None:
        for i, doc_id in enumerate(ids):
            self.vectors[doc_id] = vectors[i]
            self.texts[doc_id] = texts[i]
            self.metadata[doc_id] = metadata[i] if metadata else {}
    
    def search(self, query_vector: np.ndarray, top_k: int = 10,
               filter_dict: Optional[Dict] = None) -> List[RetrievalResult]:
        if not self.vectors:
            return []
        
        # 计算所有相似度
        scores = []
        ids = list(self.vectors.keys())
        for doc_id in ids:
            vec = self.vectors[doc_id]
            if self.config.metric == "cosine":
                score = self._cosine_sim(query_vector, vec)
            elif self.config.metric == "l2":
                score = -self._l2_dist(query_vector, vec)  # 负距离作为分数
            else:  # ip (inner product)
                score = np.dot(query_vector, vec)
            scores.append((doc_id, score))
        
        # 排序并返回top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in scores[:top_k]:
            results.append(RetrievalResult(
                id=doc_id,
                text=self.texts[doc_id],
                score=float(score),
                metadata=self.metadata[doc_id],
                source="vector"
            ))
        return results
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))
    
    def _l2_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算L2距离"""
        return float(np.linalg.norm(a - b))
    
    def delete(self, ids: List[str]) -> None:
        for doc_id in ids:
            self.vectors.pop(doc_id, None)
            self.texts.pop(doc_id, None)
            self.metadata.pop(doc_id, None)
    
    def save(self, path: str) -> None:
        data = {
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "texts": self.texts,
            "metadata": self.metadata,
            "config": {
                "dimension": self.config.dimension,
                "metric": self.config.metric
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.vectors = {k: np.array(v) for k, v in data["vectors"].items()}
        self.texts = data["texts"]
        self.metadata = data["metadata"]


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量存储"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._index = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self.texts: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        
        self._init_index()
    
    def _init_index(self):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        
        dimension = self.config.dimension
        
        if self.config.index_type == "flat":
            if self.config.metric == "cosine":
                # 归一化向量使用内积
                self._index = faiss.IndexFlatIP(dimension)
            elif self.config.metric == "l2":
                self._index = faiss.IndexFlatL2(dimension)
            else:
                self._index = faiss.IndexFlatIP(dimension)
        elif self.config.index_type == "hnsw":
            # HNSW索引
            self._index = faiss.IndexHNSWFlat(dimension, 32)
        elif self.config.index_type == "ivf":
            # IVF索引需要训练
            quantizer = faiss.IndexFlatL2(dimension)
            self._index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            self._index = faiss.IndexFlatL2(dimension)
    
    def add(self, ids: List[str], vectors: np.ndarray, texts: List[str],
            metadata: Optional[List[Dict]] = None) -> None:
        import faiss
        
        vectors = vectors.astype('float32')
        
        # 如果是cosine度量，先归一化
        if self.config.metric == "cosine":
            faiss.normalize_L2(vectors)
        
        # 添加到索引
        start_idx = len(self._idx_to_id)
        self._index.add(vectors)
        
        for i, doc_id in enumerate(ids):
            idx = start_idx + i
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id
            self.texts[doc_id] = texts[i]
            self.metadata[doc_id] = metadata[i] if metadata else {}
    
    def search(self, query_vector: np.ndarray, top_k: int = 10,
               filter_dict: Optional[Dict] = None) -> List[RetrievalResult]:
        import faiss
        
        if self._index.ntotal == 0:
            return []
        
        query = query_vector.astype('float32').reshape(1, -1)
        
        if self.config.metric == "cosine":
            faiss.normalize_L2(query)
        
        # 搜索
        scores, indices = self._index.search(query, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # FAISS返回-1表示无效
                continue
            doc_id = self._idx_to_id.get(int(idx))
            if doc_id:
                results.append(RetrievalResult(
                    id=doc_id,
                    text=self.texts[doc_id],
                    score=float(score),
                    metadata=self.metadata[doc_id],
                    source="vector"
                ))
        return results
    
    def delete(self, ids: List[str]) -> None:
        # FAISS不支持高效删除，需要重建索引
        raise NotImplementedError("Delete not supported for FAISS. Use rebuild_index().")
    
    def save(self, path: str) -> None:
        import faiss
        faiss.write_index(self._index, path)
        
        meta_path = path + ".meta"
        with open(meta_path, "w") as f:
            json.dump({
                "id_to_idx": self._id_to_idx,
                "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                "texts": self.texts,
                "metadata": self.metadata
            }, f)
    
    def load(self, path: str) -> None:
        import faiss
        self._index = faiss.read_index(path)
        
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
            self._id_to_idx = data["id_to_idx"]
            self._idx_to_id = {int(k): v for k, v in data["idx_to_id"].items()}
            self.texts = data["texts"]
            self.metadata = data["metadata"]


class VectorRetriever:
    """向量召回器"""
    
    def __init__(
        self,
        embedder,  # embedding模型，需要有encode(texts) -> np.ndarray方法
        config: Optional[VectorStoreConfig] = None,
        vector_store: Optional[BaseVectorStore] = None,
        batch_size: int = 100
    ):
        self.embedder = embedder
        self.config = config or VectorStoreConfig()
        self.batch_size = batch_size
        
        # 使用提供的vector store或根据config创建
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = self._create_vector_store()
    
    def _create_vector_store(self) -> BaseVectorStore:
        if self.config.backend == "memory":
            return MemoryVectorStore(self.config)
        elif self.config.backend == "faiss":
            return FAISSVectorStore(self.config)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def index(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """
        构建向量索引
        
        Args:
            documents: 文档列表，每条文档是dict
            id_field: 文档ID字段名
            text_field: 文档文本字段名
            metadata_fields: 需要保存的其他字段
        """
        texts = []
        ids = []
        metadata_list = []
        
        for doc in documents:
            texts.append(doc[text_field])
            ids.append(str(doc[id_field]))
            
            if metadata_fields:
                metadata = {f: doc.get(f) for f in metadata_fields}
            else:
                # 默认保留除text外的所有字段
                metadata = {k: v for k, v in doc.items() if k != text_field}
            metadata_list.append(metadata)
        
        # 批量编码
        all_vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_vectors = self.embedder.encode(batch_texts)
            all_vectors.append(batch_vectors)
        
        vectors = np.vstack(all_vectors)
        
        self.vector_store.add(ids, vectors, texts, metadata_list)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        return_raw_vectors: bool = False
    ) -> List[RetrievalResult]:
        """
        召回最相似的文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_dict: 元数据过滤条件
            return_raw_vectors: 是否返回原始向量
            
        Returns:
            RetrievalResult列表
        """
        # 编码查询
        query_vector = self.embedder.encode([query])[0]
        
        # 搜索
        results = self.vector_store.search(query_vector, top_k, filter_dict)
        
        if return_raw_vectors:
            # 可以附加原始向量用于调试
            pass
        
        return results
    
    def save(self, path: str) -> None:
        """保存索引"""
        self.vector_store.save(path)
    
    def load(self, path: str) -> None:
        """加载索引"""
        self.vector_store.load(path)
