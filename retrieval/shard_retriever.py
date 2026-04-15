"""
分片召回模块 - Shard Retrieval
支持大规模数据的分布式分片召回
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .vector_retriever import RetrievalResult


@dataclass
class ShardConfig:
    """分片配置"""
    num_shards: int = 4  # 分片数量
    shard_key: str = "id"  # 用于分片的key
    routing_strategy: str = "hash"  # hash, round_robin, range
    
    # 哈希分片配置
    hash_modulo: int = 100  # 取模基数
    
    # 范围分片配置
    range_boundaries: Optional[List[Any]] = None


class ShardRouter:
    """分片路由"""
    
    def __init__(self, config: ShardConfig):
        self.config = config
    
    def get_shard_id(self, key_value: Any) -> int:
        """根据key值确定分片ID"""
        if self.config.routing_strategy == "hash":
            return self._hash_route(key_value)
        elif self.config.routing_strategy == "range":
            return self._range_route(key_value)
        elif self.config.routing_strategy == "round_robin":
            return self._round_robin_route(key_value)
        else:
            return self._hash_route(key_value)
    
    def _hash_route(self, key_value: Any) -> int:
        """哈希路由"""
        key_str = str(key_value)
        hash_val = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
        return hash_val % self.config.num_shards
    
    def _range_route(self, key_value: Any) -> int:
        """范围路由"""
        boundaries = self.config.range_boundaries or []
        for i, boundary in enumerate(boundaries):
            if key_value < boundary:
                return i
        return len(boundaries)
    
    def _round_robin_route(self, key_value: Any) -> int:
        """轮询路由（需要外部状态）"""
        # 简单的哈希实现作为round_robin的近似
        return hash(key_value) % self.config.num_shards


class Shard:
    """单个分片"""
    
    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.documents: Dict[str, Dict] = {}  # doc_id -> document
        self.lock = threading.Lock()
    
    def add(self, doc_id: str, document: Dict) -> None:
        with self.lock:
            self.documents[doc_id] = document
    
    def add_batch(self, documents: List[Dict], id_field: str) -> None:
        with self.lock:
            for doc in documents:
                doc_id = str(doc[id_field])
                self.documents[doc_id] = doc
    
    def get(self, doc_id: str) -> Optional[Dict]:
        with self.lock:
            return self.documents.get(doc_id)
    
    def search(
        self,
        query: str,
        search_func: Callable[[Dict, str], float],
        top_k: int = 10
    ) -> List[Dict]:
        """在分片内搜索"""
        results = []
        for doc_id, doc in self.documents.items():
            text = doc.get("text", "")
            score = search_func(doc, query)
            if score > 0:
                results.append({
                    "id": doc_id,
                    "score": score,
                    "document": doc
                })
        
        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def filter(self, filter_dict: Dict) -> List[Dict]:
        """在分片内过滤"""
        results = []
        for doc in self.documents.values():
            match = True
            for key, value in filter_dict.items():
                if doc.get(key) != value:
                    match = False
                    break
            if match:
                results.append(doc)
        return results
    
    def count(self) -> int:
        with self.lock:
            return len(self.documents)
    
    def save(self, path: str) -> None:
        with self.lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump({
                    "shard_id": self.shard_id,
                    "documents": self.documents
                }, f)
    
    def load(self, path: str) -> None:
        with self.lock:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                self.documents = data["documents"]


class ShardRetriever:
    """
    分片召回器
    
    将数据分散到多个分片，支持并行检索
    """
    
    def __init__(
        self,
        config: Optional[ShardConfig] = None,
        shards: Optional[List[Shard]] = None,
        search_func: Optional[Callable] = None,
        num_workers: int = 4
    ):
        self.config = config or ShardConfig()
        self.router = ShardRouter(self.config)
        self.num_workers = num_workers
        
        # 初始化分片
        if shards:
            self.shards = shards
        else:
            self.shards = [Shard(i) for i in range(self.config.num_shards)]
        
        # 默认搜索函数（基于关键词匹配）
        self.search_func = search_func or self._default_search_func
    
    @staticmethod
    def _default_search_func(doc: Dict, query: str) -> float:
        """默认搜索函数：简单的词匹配"""
        text = doc.get("text", "").lower()
        query_lower = query.lower()
        
        # 简单的TF匹配
        query_words = set(query_lower.split())
        text_words = set(text.split())
        
        if not query_words:
            return 0.0
        
        # Jaccard相似度
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)
        
        return intersection / union if union > 0 else 0.0
    
    def index(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
        parallel: bool = True
    ) -> None:
        """
        构建分片索引
        
        Args:
            documents: 文档列表
            id_field: ID字段名
            text_field: 文本字段名
            metadata_fields: 元数据字段
            parallel: 是否并行添加
        """
        # 预处理文档
        processed_docs = []
        for doc in documents:
            processed_doc = {
                text_field: doc.get(text_field, ""),
            }
            # 保留所有字段
            processed_doc.update(doc)
            processed_docs.append(processed_doc)
        
        def add_to_shard(shard_id: int, shard_docs: List[Dict]) -> None:
            for doc in shard_docs:
                doc_id = str(doc[id_field])
                self.shards[shard_id].add(doc_id, doc)
        
        # 按分片分组文档
        shard_docs: Dict[int, List[Dict]] = defaultdict(list)
        
        for doc in processed_docs:
            doc_id = str(doc[id_field])
            shard_id = self.router.get_shard_id(doc_id)
            shard_docs[shard_id].append(doc)
        
        # 添加到分片
        if parallel and len(processed_docs) > 100:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for shard_id, docs in shard_docs.items():
                    future = executor.submit(add_to_shard, shard_id, docs)
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
        else:
            for shard_id, docs in shard_docs.items():
                for doc in docs:
                    doc_id = str(doc[id_field])
                    self.shards[shard_id].add(doc_id, doc)
        
        # 统计信息
        total = sum(shard.count() for shard in self.shards)
        print(f"Indexed {total} documents into {self.config.num_shards} shards")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        shard_ids: Optional[List[int]] = None,
        parallel: bool = True
    ) -> List[RetrievalResult]:
        """
        分片召回
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_dict: 元数据过滤条件
            shard_ids: 指定查询哪些分片（None表示全部）
            parallel: 是否并行查询
            
        Returns:
            RetrievalResult列表
        """
        target_shards = (
            [self.shards[i] for i in shard_ids]
            if shard_ids
            else self.shards
        )
        
        if parallel and len(target_shards) > 1:
            results = self._parallel_search(query, target_shards, top_k)
        else:
            results = self._sequential_search(query, target_shards, top_k)
        
        # 应用过滤
        if filter_dict:
            filtered = []
            for r in results:
                doc = r.metadata
                match = all(doc.get(k) == v for k, v in filter_dict.items())
                if match:
                    filtered.append(r)
            results = filtered
        
        # 排序并截断
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _sequential_search(
        self,
        query: str,
        shards: List[Shard],
        top_k: int
    ) -> List[RetrievalResult]:
        """顺序搜索所有分片"""
        all_results = []
        
        for shard in shards:
            shard_results = shard.search(query, self.search_func, top_k)
            for r in shard_results:
                all_results.append(RetrievalResult(
                    id=r["id"],
                    text=r["document"].get("text", ""),
                    score=r["score"],
                    metadata=r["document"],
                    source=f"shard_{shard.shard_id}"
                ))
        
        return all_results
    
    def _parallel_search(
        self,
        query: str,
        shards: List[Shard],
        top_k: int
    ) -> List[RetrievalResult]:
        """并行搜索所有分片"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(shard.search, query, self.search_func, top_k): shard
                for shard in shards
            }
            
            for future in as_completed(futures):
                shard = futures[future]
                try:
                    shard_results = future.result()
                    for r in shard_results:
                        all_results.append(RetrievalResult(
                            id=r["id"],
                            text=r["document"].get("text", ""),
                            score=r["score"],
                            metadata=r["document"],
                            source=f"shard_{shard.shard_id}"
                        ))
                except Exception as e:
                    print(f"Shard {shard.shard_id} search error: {e}")
        
        return all_results
    
    def get_shard_stats(self) -> Dict[int, int]:
        """获取每个分片的文档数量"""
        return {i: shard.count() for i, shard in enumerate(self.shards)}
    
    def save(self, base_path: str) -> None:
        """保存所有分片"""
        os.makedirs(base_path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(base_path, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "num_shards": self.config.num_shards,
                "shard_key": self.config.shard_key,
                "routing_strategy": self.config.routing_strategy
            }, f)
        
        # 保存每个分片
        for i, shard in enumerate(self.shards):
            shard_path = os.path.join(base_path, f"shard_{i}.json")
            shard.save(shard_path)
    
    def load(self, base_path: str) -> None:
        """加载所有分片"""
        # 加载配置
        config_path = os.path.join(base_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            self.config.num_shards = config_data["num_shards"]
            self.config.shard_key = config_data["shard_key"]
            self.config.routing_strategy = config_data["routing_strategy"]
            self.router = ShardRouter(self.config)
        
        # 加载每个分片
        for i in range(self.config.num_shards):
            shard_path = os.path.join(base_path, f"shard_{i}.json")
            shard = Shard(i)
            shard.load(shard_path)
            self.shards[i] = shard
    
    def add_shard(self, shard: Shard) -> None:
        """添加新的分片"""
        self.shards.append(shard)
        self.config.num_shards = len(self.shards)
    
    def merge_shards(self, shard_ids: List[int]) -> Shard:
        """合并多个分片"""
        merged = Shard(-1)
        for shard_id in shard_ids:
            shard = self.shards[shard_id]
            for doc_id, doc in shard.documents.items():
                merged.add(doc_id, doc)
        return merged
