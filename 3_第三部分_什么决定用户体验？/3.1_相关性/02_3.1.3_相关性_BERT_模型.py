# 02_3.1.3_相关性_BERT_模型

"""
Lecture: 3_第三部分_什么决定用户体验？/3.1_相关性
Content: 02_3.1.3_相关性_BERT_模型
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Tuple


def cross_bert_score(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """Cross-BERT style: full interaction (element-wise product -> sum).

    Simulates the richer query-doc interaction of a cross encoder as a
    weighted inner product that can capture alignment between dimensions.
    """
    return float(np.dot(query_vec, doc_vec))


def twin_tower_score(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """Twin-tower style: cosine similarity of independently encoded vectors."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_vec / (np.linalg.norm(doc_vec) + 1e-12)
    return float(np.dot(q, d))


def compare(query_vec: np.ndarray, docs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cross, twin) relevance scores for each doc."""
    cross = np.array([cross_bert_score(query_vec, d) for d in docs])
    twin = np.array([twin_tower_score(query_vec, d) for d in docs])
    return cross, twin


def main() -> None:
    print("相关性 BERT 模型 Demo")
    rng = np.random.default_rng(3)
    query = rng.normal(0, 1, 8)
    docs = rng.normal(0, 1, (4, 8))
    # 让第0个文档与查询更相关(增加对齐)
    docs[0] = query + rng.normal(0, 0.2, 8)

    cross, twin = compare(query, docs)
    print("交叉 BERT (Cross) 评分: ", np.round(cross, 3))
    print("双塔 BERT (Twin) 评分: ", np.round(twin, 3))
    print("\n两种架构对比:")
    print("  交叉: 精度高、计算量大 -> 用于精排")
    print("  双塔: 速度快、精度略低 -> 用于召回")


if __name__ == "__main__":
    main()