# 01_4.2.2_基于注意力机制的方法

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.2_词权重
Content: 01_4.2.2_基于注意力机制的方法
"""

import numpy as np
from typing import List, Tuple


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(query: np.ndarray,
                                 keys: np.ndarray,
                                 values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Attention context over a sequence.

    Args:
        query: (d,) query vector.
        keys: (T, d) key matrix.
        values: (T, d) value matrix.

    Returns:
        (context, weights), where context = weights @ values is (d,)
        and weights are (T,) non-negative, summing to 1.
    """
    scores = keys @ query                       # (T,)
    scale = 1.0 / np.sqrt(keys.shape[1])
    weights = softmax(scores * scale)          # (T,)
    context = weights @ values                 # (d,)
    return context, weights


def main() -> None:
    print("基于注意力机制的词权重 (Attention Word Weight) Demo")
    np.random.seed(0)
    words = ["搜索", "引擎", "召回", "排序"]
    embeddings = np.random.randn(4, 8)          # (T, d), 每个词一个向量
    query = embeddings.mean(axis=0)            # 伪query = 文档平均向量
    context, weights = scaled_dot_product_attention(query, embeddings, embeddings)
    print("  各词的注意力权重:")
    for word, w in zip(words, weights):
        print(f"    {word:<6} 权重={w:.4f}")
    print(f"  权重和 = {weights.sum():.4f}")
    print(f"  聚合上下文向量 (前3维) = {np.round(context[:3], 4)}")


if __name__ == "__main__":
    main()