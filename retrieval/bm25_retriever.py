"""
BM25召回模块 - BM25 Retrieval
基于倒排索引的稀疏检索
"""

from __future__ import annotations

import math
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import re

from .vector_retriever import RetrievalResult


@dataclass
class BM25Result:
    """单条BM25召回结果"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Indexer:
    """BM25索引器"""
    
    def __init__(
        self,
        k1: float = 1.5,      # BM25 term frequency saturation parameter
        b: float = 0.75,       # BM25 length normalization parameter
        avg_doc_len: Optional[float] = None
    ):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len
        
        # 索引结构
        self.doc_ids: List[str] = []          # doc_id列表
        self.doc_texts: Dict[str, str] = {}    # doc_id -> text
        self.doc_metadata: Dict[str, Dict] = {} # doc_id -> metadata
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> token数
        
        # 倒排索引: term -> [(doc_id, term_freq)]
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        # 文档频率: term -> doc_freq
        self.doc_freq: Dict[str, int] = defaultdict(int)
        
        # IDF值缓存
        self.idf_cache: Dict[str, float] = {}
        
        self.num_docs = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词（可替换为更复杂的分词器）"""
        # 转为小写，按非字母数字分割
        tokens = re.findall(r'\w+', text.lower())
        if tokens and len(tokens) == 1 and not tokens[0].isascii():
            # 可能是中文单字，尝试用jieba
            try:
                import jieba
                tokens = list(jieba.cut(text.lower()))
            except ImportError:
                pass
        return tokens
    
    def _compute_avg_doc_len(self) -> float:
        """计算平均文档长度"""
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """添加单个文档"""
        tokens = self._tokenize(text)
        
        self.doc_ids.append(doc_id)
        self.doc_texts[doc_id] = text
        self.doc_metadata[doc_id] = metadata or {}
        self.doc_lengths[doc_id] = len(tokens)
        
        # 更新倒排索引
        term_freqs = Counter(tokens)
        for term, freq in term_freqs.items():
            self.inverted_index[term].append((doc_id, freq))
            self.doc_freq[term] += 1
        
        self.num_docs += 1
        self.avg_doc_len = self._compute_avg_doc_len()
        
        # 清除IDF缓存
        self.idf_cache.clear()
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """批量添加文档"""
        for doc in documents:
            doc_id = str(doc[id_field])
            text = doc[text_field]
            
            if metadata_fields:
                metadata = {f: doc.get(f) for f in metadata_fields}
            else:
                metadata = {k: v for k, v in doc.items() if k != text_field}
            
            self.add_document(doc_id, text, metadata)
    
    def _compute_idf(self, term: str) -> float:
        """计算IDF"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        df = self.doc_freq.get(term, 0)
        
        if df == 0:
            idf = 0.0
        else:
            # 标准BM25 IDF公式
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
        
        self.idf_cache[term] = idf
        return idf
    
    def compute_bm25_score(
        self,
        query_terms: List[str],
        doc_id: str
    ) -> float:
        """计算单个文档的BM25分数"""
        if doc_id not in self.doc_lengths:
            return 0.0
        
        doc_len = self.doc_lengths[doc_id]
        avg_len = self.avg_doc_len if self.avg_doc_len else 1
        
        # 统计该文档中查询词的出现频率
        doc_term_freqs = Counter()
        for term in self._tokenize(self.doc_texts[doc_id]):
            if term in query_terms:
                doc_term_freqs[term] += 1
        
        score = 0.0
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            idf = self._compute_idf(term)
            tf = doc_term_freqs.get(term, 0)
            
            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            
            score += idf * numerator / (denominator + 1e-8)
        
        return score
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_func: Optional[callable] = None
    ) -> List[BM25Result]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_func: 过滤函数，接受doc_id返回bool
            
        Returns:
            BM25Result列表，按分数降序排列
        """
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # 计算所有候选文档的分数
        candidate_doc_ids = set()
        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, _ in self.inverted_index[term]:
                    if filter_func is None or filter_func(doc_id):
                        candidate_doc_ids.add(doc_id)
        
        # 计算分数
        scores = []
        for doc_id in candidate_doc_ids:
            score = self.compute_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k
        results = []
        for doc_id, score in scores[:top_k]:
            results.append(BM25Result(
                id=doc_id,
                text=self.doc_texts[doc_id],
                score=score,
                metadata=self.doc_metadata[doc_id]
            ))
        
        return results
    
    def get_inverted_index(self, term: str) -> List[Tuple[str, int]]:
        """获取词的倒排列表"""
        return self.inverted_index.get(term, [])
    
    def save(self, path: str) -> None:
        """保存索引"""
        data = {
            "k1": self.k1,
            "b": self.b,
            "avg_doc_len": self.avg_doc_len,
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "doc_metadata": self.doc_metadata,
            "doc_lengths": self.doc_lengths,
            "inverted_index": {k: list(v) for k, v in self.inverted_index.items()},
            "doc_freq": dict(self.doc_freq),
            "num_docs": self.num_docs
        }
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        """加载索引"""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.k1 = data["k1"]
        self.b = data["b"]
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        self.doc_metadata = data["doc_metadata"]
        self.doc_lengths = data["doc_lengths"]
        self.inverted_index = defaultdict(list, {k: list(v) for k, v in data["inverted_index"].items()})
        self.doc_freq = defaultdict(int, data["doc_freq"])
        self.num_docs = data["num_docs"]


class BM25Retriever:
    """BM25召回器"""
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_path: Optional[str] = None
    ):
        self.indexer = BM25Indexer(k1=k1, b=b)
        self.index_path = index_path
    
    def index(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = "id",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """构建BM25索引"""
        self.indexer.add_documents(
            documents,
            id_field=id_field,
            text_field=text_field,
            metadata_fields=metadata_fields
        )
        
        if self.index_path:
            self.indexer.save(self.index_path)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        BM25召回
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_dict: 元数据过滤条件
            
        Returns:
            RetrievalResult列表
        """
        def filter_func(doc_id: str) -> bool:
            if filter_dict is None:
                return True
            metadata = self.indexer.doc_metadata.get(doc_id, {})
            for key, value in filter_dict.items():
                if metadata.get(key) != value:
                    return False
            return True
        
        results = self.indexer.search(query, top_k, filter_func)
        
        # 转换为统一的RetrievalResult格式
        return [
            RetrievalResult(
                id=r.id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                source="bm25"
            )
            for r in results
        ]
    
    def save(self, path: str) -> None:
        """保存索引"""
        self.indexer.save(path)
    
    def load(self, path: str) -> None:
        """加载索引"""
        self.indexer.load(path)
