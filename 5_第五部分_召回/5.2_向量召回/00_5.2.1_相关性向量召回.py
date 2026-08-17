# 00_5.2.1_相关性向量召回

"""
Lecture: 5_第五部分_召回/5.2_向量召回
Content: 00_5.2.1_相关性向量召回
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Dict, List


def embed(tokens: List[str], word_vec: Dict[str, np.ndarray]) -> np.ndarray:
    """Mean-pool word embeddings into a single document vector."""
    vecs = [word_vec[w] for w in tokens if w in word_vec]
    if not vecs:
        return np.zeros_like(list(word_vec.values())[0])
    return np.mean(vecs, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def top_k(query_vec: np.ndarray, doc_vecs: List[np.ndarray], k: int = 3) -> List[int]:
    """Return top-k doc indices (most cosine-similar) to the query vector."""
    sims = [cosine(query_vec, d) for d in doc_vecs]
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]


def main() -> None:
    print("相关性向量召回 (Vector Relevance Recall) Demo")
    rng = np.random.default_rng(0)
    vocab = ["python", "爬虫", "教程", "数据", "分析", "机器学习", "算法", "书"]
    word_vec = {w: rng.normal(0, 1, 8) for w in vocab}

    docs = ["python 爬虫 教程", "python 数据 分析", "机器学习 算法 书"]
    D = [embed(d.split(), word_vec) for d in docs]
    q = "python 爬虫"
    query_vec = embed(q.split(), word_vec)

    print(f"查询: {q}")
    for i in top_k(query_vec, D, 3):
        print(f"  召回文档{i}: '{docs[i]}'")


if __name__ == "__main__":
    main()