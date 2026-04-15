# Multi-Retrieval 多路召回框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

统一的多路召回框架，支持向量召回、BM25召回、分片召回的融合。

## 特性

- 🚀 **向量召回** - 基于语义向量相似度，支持 FAISS / 内存存储
- 🔍 **BM25召回** - 基于倒排索引的关键词检索，内置 jieba 中文分词
- 📦 **分片召回** - 大规模数据分布式分片，支持并行检索
- ⚖️ **分数融合** - 加权融合 / RRF (Reciprocal Rank Fusion)
- 🎯 **重排序** - 支持 Cross-Encoder、Cohere API

## 安装

```bash
pip install numpy faiss-cpu jieba

# 可选依赖
pip install sentence-transformers  # Embedding
pip install torch transformers     # Cross-Encoder
pip install requests              # Cohere API
```

## 快速开始

```python
from retrieval import UnifiedRetriever, RetrievalConfig
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever

# 准备文档
documents = [
    {"id": "1", "text": "Python编程语言教程", "category": "programming"},
    {"id": "2", "text": "机器学习入门", "category": "ai"},
    {"id": "3", "text": "深度学习实战", "category": "ai"},
]

# 初始化召回器
vector_retriever = VectorRetriever(embedder=your_embedder)
bm25_retriever = BM25Retriever()

# 构建索引
vector_retriever.index(documents)
bm25_retriever.index(documents)

# 配置统一召回器
config = RetrievalConfig(
    enable_vector=True,
    enable_bm25=True,
    vector_weight=0.4,
    bm25_weight=0.6,
    final_top_k=10
)

unified = UnifiedRetriever(
    config=config,
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever
)

# 执行召回
results = unified.retrieve("机器学习相关内容")
```

## 架构

```
Query
  │
  ├──► [Vector Retriever] ──► Top-K vectors ──┐
  │                                          │
  ├──► [BM25 Retriever] ───► Top-K keywords ─┤
  │                                          │ Merge
  └──► [Shard Retriever] ──► Top-K shards ───┘
                                                      │
                                              ┌───────▼───────┐
                                              │  Score Fusion │
                                              └───────┬───────┘
                                                      │
                                              ┌───────▼───────┐
                                              │  Final Top-K  │
                                              └───────────────┘
```

## 项目结构

```
multi_retrieval/
├── retrieval/
│   ├── __init__.py              # 统一导出
│   ├── vector_retriever.py      # 向量召回
│   ├── bm25_retriever.py        # BM25 召回
│   ├── shard_retriever.py       # 分片召回
│   ├── unified_retriever.py     # 统一召回器
│   └── reranker.py              # 重排序
├── examples/
│   └── usage.py                 # 使用示例
├── docs/
│   └── 技术文档-多路召回框架设计.md
├── pyproject.toml
└── requirements.txt
```

## 配置参考

```python
config = RetrievalConfig(
    # 启用控制
    enable_vector=True,
    enable_bm25=True,
    enable_shard=False,

    # Top-K 配置
    vector_top_k=20,
    bm25_top_k=20,
    shard_top_k=20,
    final_top_k=10,

    # 融合权重
    vector_weight=0.4,
    bm25_weight=0.4,
    shard_weight=0.2,

    # RRF 融合
    use_rrf=False,
    rrf_k=60,

    # 性能配置
    parallel=True,
    num_workers=4,
)
```

## 场景化推荐

| 场景 | 配置建议 |
|------|---------|
| 语义为主 | vector_weight=0.7, bm25_weight=0.3 |
| 关键词为主 | vector_weight=0.3, bm25_weight=0.7 |
| 均衡召回 | use_rrf=True |
| 大规模数据 | enable_shard=True, parallel=True |

## License

MIT
