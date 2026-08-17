# 03_2.1.4_排序任务

"""
Lecture: 2_第二部分_机器学习基础/2.1_机器学习任务
Content: 03_2.1.4_排序任务
"""

import numpy as np
from typing import List, Tuple


def pairwise_logistic(pred: np.ndarray, labels: np.ndarray) -> float:
    """Pairwise logistic loss over all (i,j) with y_i > y_j.

    L = sum_{(i,j): y_i > y_j} log(1 + exp(-(p_i - p_j))).
    """
    loss = 0.0
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if labels[i] > labels[j]:
                loss += np.log(1 + np.exp(-(pred[i] - pred[j])))
    return float(loss / (n * n))


def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """DCG = sum_i (2^score_i - 1) / log2(i+1)."""
    s = scores[:k]
    gains = 2 ** s - 1
    pos = np.arange(1, len(s) + 1)
    return float(np.sum(gains / np.log2(pos + 1)))


def ndcg(pred: np.ndarray, labels: np.ndarray) -> float:
    """NDCG = DCG(pred order) / DCG(ideal order)."""
    order = np.argsort(-pred)
    dcg = dcg_at_k(labels[order], len(order))
    ideal = dcg_at_k(np.sort(labels)[::-1], len(labels))
    return dcg / ideal if ideal > 0 else 0.0


def main() -> None:
    print("排序任务 (Ranking Task) Demo")
    # 每个文档一个特征, 标签为相关性
    X = np.array([[1.0], [0.2], [0.8], [0.4]])
    labels = np.array([3.0, 1.0, 2.0, 1.0])
    pred = X[:, 0]  # 直接用特征作评分

    print(f"排序评分: {pred}")
    print(f"排序结果(按评分): {np.argsort(-pred)}")
    print(f"Pairwise Logistic Loss = {pairwise_logistic(pred, labels):.4f}")
    print(f"NDCG = {ndcg(pred, labels):.4f}")


if __name__ == "__main__":
    main()