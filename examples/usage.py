"""
多路召回使用示例
"""

from retrieval import UnifiedRetriever, RetrievalConfig
from retrieval.vector_retriever import VectorRetriever, VectorStoreConfig, MemoryVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.shard_retriever import ShardRetriever, ShardConfig


def demo_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("Demo 1: 基本使用")
    print("=" * 60)
    
    # 准备测试数据
    documents = [
        {
            "id": "doc1",
            "text": "Python是一种高级编程语言，由Guido van Rossum创建。",
            "category": "programming",
            "lang": "zh"
        },
        {
            "id": "doc2", 
            "text": "JavaScript是一种脚本语言，用于Web开发。",
            "category": "programming",
            "lang": "zh"
        },
        {
            "id": "doc3",
            "text": "机器学习是人工智能的一个分支。",
            "category": "ai",
            "lang": "zh"
        },
        {
            "id": "doc4",
            "text": "深度学习是机器学习的子集，使用神经网络。",
            "category": "ai",
            "lang": "zh"
        },
        {
            "id": "doc5",
            "text": "自然语言处理用于处理文本数据。",
            "category": "nlp",
            "lang": "zh"
        },
    ]
    
    # 方式1: 使用内存向量存储 + BM25
    print("\n1.1 使用内存向量存储 + BM25")
    
    # 模拟embedding模型
    class DummyEmbedder:
        def encode(self, texts):
            import numpy as np
            # 简单的词袋embedding模拟
            vectors = []
            for text in texts:
                words = set(text.lower().split())
                vec = np.random.randn(128)
                # 使相关文本有相似向量
                if "机器学习" in text or "深度学习" in text:
                    vec[0] = 1.0
                elif "python" in text.lower() or "javascript" in text.lower():
                    vec[1] = 1.0
                vectors.append(vec)
            return np.array(vectors)
    
    embedder = DummyEmbedder()
    
    # 创建配置
    vector_config = VectorStoreConfig(backend="memory", dimension=128)
    bm25_retriever = BM25Retriever()
    
    # 构建向量索引
    vector_retriever = VectorRetriever(embedder=embedder, config=vector_config)
    vector_retriever.index(documents)
    print(f"  向量索引构建完成: {len(documents)} 文档")
    
    # 构建BM25索引
    bm25_retriever.index(documents)
    print(f"  BM25索引构建完成: {len(documents)} 文档")
    
    # 创建统一召回器
    config = RetrievalConfig(
        enable_vector=True,
        enable_bm25=True,
        enable_shard=False,
        vector_weight=0.5,
        bm25_weight=0.5,
        final_top_k=3
    )
    
    unified = UnifiedRetriever(
        config=config,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )
    
    # 执行召回
    query = "机器学习相关内容"
    print(f"\n查询: '{query}'")
    
    # 查看各路召回结果
    all_results = unified.retrieve(query, return_all_scores=True)
    print("\n各路召回结果:")
    for source, results in all_results.items():
        if results:
            print(f"  {source}:")
            for r in results[:3]:
                print(f"    - {r.id}: {r.text[:30]}... (score={r.score:.4f})")
    
    # 获取融合结果
    fused_results = unified.retrieve(query)
    print("\n融合后结果:")
    for i, r in enumerate(fused_results, 1):
        print(f"  {i}. {r.id}: {r.text[:40]}... (score={r.score:.4f}, source={r.source})")


def demo_with_sharding():
    """分片召回示例"""
    print("\n" + "=" * 60)
    print("Demo 2: 分片召回")
    print("=" * 60)
    
    # 生成大量测试数据
    import random
    categories = ["技术", "商业", "娱乐", "体育", "科学"]
    
    documents = []
    for i in range(100):
        cat = random.choice(categories)
        documents.append({
            "id": f"doc_{i}",
            "text": f"这是第{i}篇{cat}类别的文档，内容包含相关的关键词信息。",
            "category": cat
        })
    
    # 创建分片配置
    shard_config = ShardConfig(num_shards=4, routing_strategy="hash")
    
    # 创建分片召回器
    shard_retriever = ShardRetriever(config=shard_config)
    shard_retriever.index(documents)
    
    print(f"分片统计: {shard_retriever.get_shard_stats()}")
    
    # 查询
    query = "技术类别的内容"
    results = shard_retriever.retrieve(query, top_k=5)
    
    print(f"\n查询: '{query}'")
    print(f"召回结果:")
    for r in results:
        print(f"  - {r.id}: {r.metadata.get('category', 'N/A')} (score={r.score:.4f})")


def demo_rrf_fusion():
    """RRF融合示例"""
    print("\n" + "=" * 60)
    print("Demo 3: RRF (Reciprocal Rank Fusion) 融合")
    print("=" * 60)
    
    # 使用之前的示例数据
    documents = [
        {"id": "a", "text": "苹果是一种水果"},
        {"id": "b", "text": "苹果公司生产iPhone"},
        {"id": "c", "text": "香蕉是黄色的水果"},
        {"id": "d", "text": "乔布斯创立了苹果公司"},
    ]
    
    # 创建召回器
    vector_retriever = VectorRetriever(
        embedder=type('Dummy', (), {
            'encode': lambda self, texts: __import__('numpy').random.randn(len(texts), 64)
        })()
    )
    vector_retriever.index(documents)
    
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(documents)
    
    # 测试加权融合
    config_weighted = RetrievalConfig(
        enable_vector=True,
        enable_bm25=True,
        use_rrf=False,
        vector_weight=0.5,
        bm25_weight=0.5
    )
    
    weighted_retriever = UnifiedRetriever(
        config=config_weighted,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )
    
    # 测试RRF融合
    config_rrf = RetrievalConfig(
        enable_vector=True,
        enable_bm25=True,
        use_rrf=True,
        rrf_k=60
    )
    
    rrf_retriever = UnifiedRetriever(
        config=config_rrf,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )
    
    query = "苹果"
    
    print(f"查询: '{query}'")
    
    weighted_results = weighted_retriever.retrieve(query, final_top_k=4)
    print("\n加权融合结果:")
    for r in weighted_results:
        print(f"  {r.id}: score={r.score:.4f}")
    
    rrf_results = rrf_retriever.retrieve(query, final_top_k=4)
    print("\nRRF融合结果:")
    for r in rrf_results:
        print(f"  {r.id}: score={r.score:.4f}")


def demo_with_filter():
    """带过滤的召回示例"""
    print("\n" + "=" * 60)
    print("Demo 4: 带元数据过滤的召回")
    print("=" * 60)
    
    documents = [
        {"id": "1", "text": "Python教程", "category": "programming", "level": "beginner"},
        {"id": "2", "text": "Python高级编程", "category": "programming", "level": "advanced"},
        {"id": "3", "text": "JavaScript入门", "category": "programming", "level": "beginner"},
        {"id": "4", "text": "机器学习导论", "category": "ai", "level": "beginner"},
        {"id": "5", "text": "深度学习进阶", "category": "ai", "level": "advanced"},
    ]
    
    # 创建召回器
    bm25 = BM25Retriever()
    bm25.index(documents)
    
    config = RetrievalConfig(enable_vector=False, enable_bm25=True)
    unified = UnifiedRetriever(config=config, bm25_retriever=bm25)
    
    query = "教程"
    
    # 不过滤
    results_all = unified.retrieve(query, final_top_k=5)
    print(f"查询: '{query}' (不过滤)")
    for r in results_all:
        print(f"  {r.id}: {r.metadata}")
    
    # 过滤: category=programming
    results_prog = unified.retrieve(query, final_top_k=5, filter_dict={"category": "programming"})
    print(f"\n查询: '{query}' (category=programming)")
    for r in results_prog:
        print(f"  {r.id}: {r.metadata}")
    
    # 过滤: level=beginner
    results_beginner = unified.retrieve(query, final_top_k=5, filter_dict={"level": "beginner"})
    print(f"\n查询: '{query}' (level=beginner)")
    for r in results_beginner:
        print(f"  {r.id}: {r.metadata}")


def demo_save_load():
    """保存和加载示例"""
    print("\n" + "=" * 60)
    print("Demo 5: 保存和加载索引")
    print("=" * 60)
    
    documents = [
        {"id": "1", "text": "测试文档1"},
        {"id": "2", "text": "测试文档2"},
        {"id": "3", "text": "测试文档3"},
    ]
    
    import tempfile
    import os
    
    tmpdir = tempfile.mkdtemp()
    vector_path = os.path.join(tmpdir, "vector")
    bm25_path = os.path.join(tmpdir, "bm25.json")
    
    # 创建并保存
    embedder = type('Dummy', (), {
        'encode': lambda self, texts: __import__('numpy').random.randn(len(texts), 64)
    })()
    
    vector = VectorRetriever(embedder=embedder)
    vector.index(documents)
    vector.save(vector_path)
    
    bm25 = BM25Retriever()
    bm25.index(documents)
    bm25.save(bm25_path)
    
    print(f"索引已保存到: {tmpdir}")
    print(f"  - vector: {vector_path}")
    print(f"  - bm25: {bm25_path}")
    
    # 加载
    vector2 = VectorRetriever(embedder=embedder)
    vector2.load(vector_path)
    
    bm252 = BM25Retriever()
    bm252.load(bm25_path)
    
    results = bm252.retrieve("测试")
    print(f"\n加载后召回结果: {[r.id for r in results]}")
    
    # 清理
    import shutil
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    demo_basic_usage()
    demo_with_sharding()
    demo_rrf_fusion()
    demo_with_filter()
    demo_save_load()
