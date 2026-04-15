"""
重排序模块 - Reranker
支持基于机器学习模型的重排序
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from .vector_retriever import RetrievalResult


@dataclass
class RerankConfig:
    """重排序配置"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"  # cpu, cuda


class BaseReranker:
    """重排序基类"""
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        重排序结果
        
        Args:
            query: 查询文本
            results: 候选结果
            top_k: 返回数量
            
        Returns:
            重排序后的结果
        """
        raise NotImplementedError
    
    def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        计算query和document的相关性分数
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            分数列表
        """
        raise NotImplementedError


class SimilarityReranker(BaseReranker):
    """基于相似度的重排序（简单实现）"""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """使用embedding相似度重排序"""
        if not results:
            return []
        
        # 编码query和documents
        query_vec = self.embedder.encode([query])[0]
        doc_texts = [r.text for r in results]
        doc_vecs = self.embedder.encode(doc_texts)
        
        # 计算相似度
        scores = []
        for vec in doc_vecs:
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            scores.append(float(sim))
        
        # 排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 重新排列结果
        reranked = []
        for idx, score in indexed_scores[:top_k or len(results)]:
            r = results[idx]
            reranked.append(RetrievalResult(
                id=r.id,
                text=r.text,
                score=score,
                metadata=r.metadata,
                source=r.source
            ))
        
        return reranked
    
    def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """计算相似度分数"""
        query_vec = self.embedder.encode([query])[0]
        doc_vecs = self.embedder.encode(documents)
        
        scores = []
        for vec in doc_vecs:
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            scores.append(float(sim))
        
        return scores


class CrossEncoderReranker(BaseReranker):
    """基于Cross-Encoder的重排序"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError("transformers and torch are required for CrossEncoderReranker")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name
        )
        self.model.to(self.config.device)
        self.model.eval()
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """使用Cross-Encoder重排序"""
        if not results:
            return []
        
        doc_texts = [r.text for r in results]
        scores = self.score(query, doc_texts)
        
        # 排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 重新排列结果
        reranked = []
        for idx, score in indexed_scores[:top_k or len(results)]:
            r = results[idx]
            reranked.append(RetrievalResult(
                id=r.id,
                text=r.text,
                score=score,
                metadata=r.metadata,
                source=r.source
            ))
        
        return reranked
    
    def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """使用Cross-Encoder计算相关性分数"""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        if not documents:
            return []
        
        all_scores = []
        
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用logits作为分数
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            all_scores.extend(batch_scores if isinstance(batch_scores, list) else [batch_scores])
        
        return all_scores


class CohereReranker(BaseReranker):
    """Cohere Rerank API"""
    
    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.ai"
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """使用Cohere重排序"""
        import requests
        
        if not results:
            return []
        
        doc_texts = [r.text for r in results]
        top_k = top_k or len(results)
        
        response = requests.post(
            f"{self.base_url}/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "documents": doc_texts,
                "model": self.model,
                "top_n": top_k
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Cohere rerank failed: {response.text}")
        
        data = response.json()
        
        # 构建rerank结果
        rerank_dict = {r["index"]: r["relevance_score"] for r in data["results"]}
        
        reranked = []
        for idx, score in rerank_dict.items():
            r = results[idx]
            reranked.append(RetrievalResult(
                id=r.id,
                text=r.text,
                score=score,
                metadata=r.metadata,
                source=r.source
            ))
        
        # 按分数排序
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
    
    def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """调用Cohere rerank获取分数"""
        reranked = self.rerank(query, [
            RetrievalResult(id=str(i), text=doc, score=0) 
            for i, doc in enumerate(documents)
        ], top_k=len(documents))
        
        # 按原始顺序返回分数
        scores = [0.0] * len(documents)
        for r in reranked:
            idx = int(r.id)
            scores[idx] = r.score
        
        return scores


class Reranker:
    """重排序器统一接口"""
    
    SUPPORTED_TYPES = {
        "similarity": SimilarityReranker,
        "cross-encoder": CrossEncoderReranker,
        "cohere": CohereReranker
    }
    
    def __init__(
        self,
        reranker_type: str = "similarity",
        **kwargs
    ):
        self.reranker_type = reranker_type
        
        if reranker_type == "similarity":
            if "embedder" not in kwargs:
                raise ValueError("embedder is required for SimilarityReranker")
            self.reranker = SimilarityReranker(kwargs["embedder"])
        elif reranker_type == "cross-encoder":
            config = kwargs.get("config", RerankConfig())
            self.reranker = CrossEncoderReranker(config)
        elif reranker_type == "cohere":
            if "api_key" not in kwargs:
                raise ValueError("api_key is required for CohereReranker")
            self.reranker = CohereReranker(kwargs["api_key"], kwargs.get("model", "rerank-english-v2.0"))
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        return self.reranker.rerank(query, results, top_k)
    
    def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        return self.reranker.score(query, documents)
