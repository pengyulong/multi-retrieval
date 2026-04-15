"""
统一召回器 - Unified Retriever
整合向量召回、BM25召回、分片召回的多路召回系统
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .vector_retriever import VectorRetriever, RetrievalResult as VectorResult
from .bm25_retriever import BM25Retriever, BM25Result
from .shard_retriever import ShardRetriever, RetrievalResult as ShardResult


# 统一的结果类型
RetrievalResult = VectorResult


@dataclass
class RetrievalConfig:
    """召回配置"""
    # 各召回方式是否启用
    enable_vector: bool = True
    enable_bm25: bool = True
    enable_shard: bool = False
    
    # 各召回方式的top_k
    vector_top_k: int = 20
    bm25_top_k: int = 20
    shard_top_k: int = 20
    
    # 最终返回数量
    final_top_k: int = 10
    
    # 融合权重
    vector_weight: float = 0.4
    bm25_weight: float = 0.4
    shard_weight: float = 0.2
    
    # 是否启用并行召回
    parallel: bool = True
    num_workers: int = 4
    
    # 分数归一化方法
    normalization: str = "minmax"  # minmax, l2, none
    
    # RRF (Reciprocal Rank Fusion) 参数
    use_rrf: bool = False
    rrf_k: int = 60  # RRF公式中的常数


class ScoreNormalizer:
    """分数归一化"""
    
    @staticmethod
    def minmax_normalize(scores: List[float]) -> List[float]:
        """Min-Max归一化到[0,1]"""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s - min_s < 1e-8:
            return [0.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    @staticmethod
    def l2_normalize(scores: List[float]) -> List[float]:
        """L2归一化"""
        if not scores:
            return []
        norm = np.sqrt(sum(s * s for s in scores))
        if norm < 1e-8:
            return [0.0] * len(scores)
        return [s / norm for s in scores]
    
    @staticmethod
    def normalize(scores: List[float], method: str = "minmax") -> List[float]:
        if method == "minmax":
            return ScoreNormalizer.minmax_normalize(scores)
        elif method == "l2":
            return ScoreNormalizer.l2_normalize(scores)
        else:
            return scores


@dataclass 
class QueryConfig:
    """查询配置"""
    text: str
    vector_top_k: Optional[int] = None
    bm25_top_k: Optional[int] = None
    shard_top_k: Optional[int] = None
    final_top_k: Optional[int] = None
    filter_dict: Optional[Dict] = None
    shard_ids: Optional[List[int]] = None
    enable_vector: Optional[bool] = None
    enable_bm25: Optional[bool] = None
    enable_shard: Optional[bool] = None


class UnifiedRetriever:
    """
    统一召回器
    
    将向量召回、BM25召回、分片召回的结果进行融合
    """
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        shard_retriever: Optional[ShardRetriever] = None,
        reranker: Optional[Any] = None  # 可选的重排序器
    ):
        self.config = config or RetrievalConfig()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.shard_retriever = shard_retriever
        self.reranker = reranker  # 预留接口
    
    def retrieve(
        self,
        query: str,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        shard_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        shard_ids: Optional[List[int]] = None,
        return_all_scores: bool = False,
        query_config: Optional[QueryConfig] = None
    ) -> Union[List[RetrievalResult], Dict[str, List[RetrievalResult]]]:
        """
        多路召回
        
        Args:
            query: 查询文本
            vector_top_k: 向量召回数量
            bm25_top_k: BM25召回数量
            shard_top_k: 分片召回数量
            final_top_k: 最终返回数量
            filter_dict: 过滤条件
            shard_ids: 指定查询的分片
            return_all_scores: 是否返回各路单独的召回结果
            query_config: 查询配置对象（优先级更高）
            
        Returns:
            如果return_all_scores=False，返回融合后的结果
            否则返回各路单独的召回结果
        """
        # 合并配置
        if query_config:
            query = query_config.text
            vector_top_k = query_config.vector_top_k or vector_top_k
            bm25_top_k = query_config.bm25_top_k or bm25_top_k
            shard_top_k = query_config.shard_top_k or shard_top_k
            final_top_k = query_config.final_top_k or final_top_k
            filter_dict = query_config.filter_dict or filter_dict
            shard_ids = query_config.shard_ids or shard_ids
        
        # 使用默认配置
        vector_top_k = vector_top_k or self.config.vector_top_k
        bm25_top_k = bm25_top_k or self.config.bm25_top_k
        shard_top_k = shard_top_k or self.config.shard_top_k
        final_top_k = final_top_k or self.config.final_top_k
        
        # 多路召回
        if self.config.parallel:
            results = self._parallel_retrieve(
                query, vector_top_k, bm25_top_k, shard_top_k, filter_dict, shard_ids
            )
        else:
            results = self._sequential_retrieve(
                query, vector_top_k, bm25_top_k, shard_top_k, filter_dict, shard_ids
            )
        
        vector_results, bm25_results, shard_results = results
        
        if return_all_scores:
            return {
                "vector": vector_results,
                "bm25": bm25_results,
                "shard": shard_results
            }
        
        # 融合结果
        fused_results = self.fuse_results(
            vector_results, bm25_results, shard_results,
            vector_top_k, bm25_top_k, shard_top_k
        )
        
        # 截断到final_top_k
        return fused_results[:final_top_k]
    
    def _parallel_retrieve(
        self,
        query: str,
        vector_top_k: int,
        bm25_top_k: int,
        shard_top_k: int,
        filter_dict: Optional[Dict],
        shard_ids: Optional[List[int]]
    ) -> tuple:
        """并行执行多路召回"""
        vector_results = []
        bm25_results = []
        shard_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {}
            
            if self.config.enable_vector and self.vector_retriever:
                f = executor.submit(
                    self.vector_retriever.retrieve,
                    query, vector_top_k, filter_dict
                )
                futures[f] = "vector"
            
            if self.config.enable_bm25 and self.bm25_retriever:
                f = executor.submit(
                    self.bm25_retriever.retrieve,
                    query, bm25_top_k, filter_dict
                )
                futures[f] = "bm25"
            
            if self.config.enable_shard and self.shard_retriever:
                f = executor.submit(
                    self.shard_retriever.retrieve,
                    query, shard_top_k, filter_dict, shard_ids
                )
                futures[f] = "shard"
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    if source == "vector":
                        vector_results = result
                    elif source == "bm25":
                        bm25_results = result
                    elif source == "shard":
                        shard_results = result
                except Exception as e:
                    print(f"{source} retrieval error: {e}")
        
        return vector_results, bm25_results, shard_results
    
    def _sequential_retrieve(
        self,
        query: str,
        vector_top_k: int,
        bm25_top_k: int,
        shard_top_k: int,
        filter_dict: Optional[Dict],
        shard_ids: Optional[List[int]]
    ) -> tuple:
        """顺序执行多路召回"""
        vector_results = []
        bm25_results = []
        shard_results = []
        
        if self.config.enable_vector and self.vector_retriever:
            try:
                vector_results = self.vector_retriever.retrieve(query, vector_top_k, filter_dict)
            except Exception as e:
                print(f"Vector retrieval error: {e}")
        
        if self.config.enable_bm25 and self.bm25_retriever:
            try:
                bm25_results = self.bm25_retriever.retrieve(query, bm25_top_k, filter_dict)
            except Exception as e:
                print(f"BM25 retrieval error: {e}")
        
        if self.config.enable_shard and self.shard_retriever:
            try:
                shard_results = self.shard_retriever.retrieve(query, shard_top_k, filter_dict, shard_ids)
            except Exception as e:
                print(f"Shard retrieval error: {e}")
        
        return vector_results, bm25_results, shard_results
    
    def fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        shard_results: List[RetrievalResult],
        vector_top_k: int,
        bm25_top_k: int,
        shard_top_k: int
    ) -> List[RetrievalResult]:
        """
        融合多路召回结果
        
        支持两种融合方式:
        1. 加权分数融合 - 各路分数加权求和
        2. RRF融合 - Reciprocal Rank Fusion
        """
        if self.config.use_rrf:
            return self._rrf_fusion(vector_results, bm25_results, shard_results)
        else:
            return self._weighted_fusion(vector_results, bm25_results, shard_results)
    
    def _weighted_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        shard_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """加权分数融合"""
        # 构建doc_id -> 分数的映射
        doc_scores: Dict[str, Dict] = {}
        
        # 向量召回分数
        if vector_results and self.config.enable_vector:
            scores = [r.score for r in vector_results]
            normalized = ScoreNormalizer.normalize(scores, self.config.normalization)
            for r, norm_s in zip(vector_results, normalized):
                if r.id not in doc_scores:
                    doc_scores[r.id] = {"text": r.text, "metadata": r.metadata, "vector_score": 0, "bm25_score": 0, "shard_score": 0}
                doc_scores[r.id]["vector_score"] = norm_s * self.config.vector_weight
        
        # BM25召回分数
        if bm25_results and self.config.enable_bm25:
            scores = [r.score for r in bm25_results]
            normalized = ScoreNormalizer.normalize(scores, self.config.normalization)
            for r, norm_s in zip(bm25_results, normalized):
                if r.id not in doc_scores:
                    doc_scores[r.id] = {"text": r.text, "metadata": r.metadata, "vector_score": 0, "bm25_score": 0, "shard_score": 0}
                doc_scores[r.id]["bm25_score"] = norm_s * self.config.bm25_weight
        
        # 分片召回分数
        if shard_results and self.config.enable_shard:
            scores = [r.score for r in shard_results]
            normalized = ScoreNormalizer.normalize(scores, self.config.normalization)
            for r, norm_s in zip(shard_results, normalized):
                if r.id not in doc_scores:
                    doc_scores[r.id] = {"text": r.text, "metadata": r.metadata, "vector_score": 0, "bm25_score": 0, "shard_score": 0}
                doc_scores[r.id]["shard_score"] = norm_s * self.config.shard_weight
        
        # 计算加权总分
        fused = []
        for doc_id, scores in doc_scores.items():
            total_score = scores["vector_score"] + scores["bm25_score"] + scores["shard_score"]
            # 确定来源
            sources = []
            if scores["vector_score"] > 0:
                sources.append("vector")
            if scores["bm25_score"] > 0:
                sources.append("bm25")
            if scores["shard_score"] > 0:
                sources.append("shard")
            
            fused.append(RetrievalResult(
                id=doc_id,
                text=scores["text"],
                score=total_score,
                metadata=scores["metadata"],
                source=",".join(sources)
            ))
        
        # 排序
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused
    
    def _rrf_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        shard_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF)
        
        RRF = Σ 1 / (k + rank)
        其中k是常数(通常60)，rank是文档在该路召回中的排名(从1开始)
        """
        k = self.config.rrf_k
        doc_rrf_scores: Dict[str, float] = {}
        doc_info: Dict[str, Dict] = {}
        
        # 向量召回的RRF
        for rank, r in enumerate(vector_results, 1):
            doc_rrf_scores[r.id] = doc_rrf_scores.get(r.id, 0) + 1 / (k + rank)
            if r.id not in doc_info:
                doc_info[r.id] = {"text": r.text, "metadata": r.metadata}
        
        # BM25召回的RRF
        for rank, r in enumerate(bm25_results, 1):
            doc_rrf_scores[r.id] = doc_rrf_scores.get(r.id, 0) + 1 / (k + rank)
            if r.id not in doc_info:
                doc_info[r.id] = {"text": r.text, "metadata": r.metadata}
        
        # 分片召回的RRF
        for rank, r in enumerate(shard_results, 1):
            doc_rrf_scores[r.id] = doc_rrf_scores.get(r.id, 0) + 1 / (k + rank)
            if r.id not in doc_info:
                doc_info[r.id] = {"text": r.text, "metadata": r.metadata}
        
        # 构建结果
        fused = [
            RetrievalResult(
                id=doc_id,
                text=doc_info[doc_id]["text"],
                score=score,
                metadata=doc_info[doc_id]["metadata"],
                source="rrf"
            )
            for doc_id, score in doc_rrf_scores.items()
        ]
        
        # 排序
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused
    
    def index(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """构建所有索引"""
        # 向量索引
        if self.vector_retriever and self.config.enable_vector:
            print("Building vector index...")
            self.vector_retriever.index(
                documents, id_field, text_field, metadata_fields
            )
        
        # BM25索引
        if self.bm25_retriever and self.config.enable_bm25:
            print("Building BM25 index...")
            self.bm25_retriever.index(
                documents, id_field, text_field, metadata_fields
            )
        
        # 分片索引
        if self.shard_retriever and self.config.enable_shard:
            print("Building shard index...")
            self.shard_retriever.index(
                documents, id_field, text_field, metadata_fields
            )
    
    def save(self, path: str) -> None:
        """保存所有索引"""
        import json
        import os
        
        os.makedirs(path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(path, "config.json")
        config_data = {
            "enable_vector": self.config.enable_vector,
            "enable_bm25": self.config.enable_bm25,
            "enable_shard": self.config.enable_shard,
            "vector_top_k": self.config.vector_top_k,
            "bm25_top_k": self.config.bm25_top_k,
            "shard_top_k": self.config.shard_top_k,
            "final_top_k": self.config.final_top_k,
            "vector_weight": self.config.vector_weight,
            "bm25_weight": self.config.bm25_weight,
            "shard_weight": self.config.shard_weight,
            "normalization": self.config.normalization,
            "use_rrf": self.config.use_rrf,
            "rrf_k": self.config.rrf_k
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        
        # 保存各路索引
        if self.vector_retriever:
            self.vector_retriever.save(os.path.join(path, "vector.index"))
        
        if self.bm25_retriever:
            self.bm25_retriever.save(os.path.join(path, "bm25.index"))
        
        if self.shard_retriever:
            self.shard_retriever.save(os.path.join(path, "shards"))
    
    def load(self, path: str) -> None:
        """加载所有索引"""
        import json
        import os
        
        # 加载配置
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 加载各路索引
        if self.vector_retriever and os.path.exists(os.path.join(path, "vector.index")):
            self.vector_retriever.load(os.path.join(path, "vector.index"))
        
        if self.bm25_retriever and os.path.exists(os.path.join(path, "bm25.index")):
            self.bm25_retriever.load(os.path.join(path, "bm25.index"))
        
        if self.shard_retriever and os.path.exists(os.path.join(path, "shards")):
            self.shard_retriever.load(os.path.join(path, "shards"))
