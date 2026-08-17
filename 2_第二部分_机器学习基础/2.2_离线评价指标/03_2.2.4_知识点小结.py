# 03_2.2.4_知识点小结

"""
Lecture: 2_第二部分_机器学习基础/2.2_离线评价指标
Content: 03_2.2.4_知识点小结
"""

import numpy as np
from typing import List


def accuracy(y: List[int], pred: List[int]) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def ranknet_loss(pred: np.ndarray, labels: np.ndarray) -> float:
    loss = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] > labels[j]:
                loss += np.log(1 + np.exp(-(pred[i] - pred[j])))
    return float(loss)


def ndcg(scores: List[float]) -> float:
    s = np.asarray(scores, dtype=float)
    pos = np.arange(1, len(s) + 1)
    dcgv = np.sum(s / np.log2(pos + 1))
    ideal = np.sum(np.sort(s)[::-1] / np.log2(np.arange(1, len(s) + 1) + 1))
    return dcgv / ideal if ideal > 0 else 0.0


def main() -> None:
    print("=== 离线评价指标小结 (Offline Metrics Summary) Demo ===")
    print("Pointwise(分类):")
    print(f"  准确率 = {accuracy([1, 0, 1, 1, 0], [1, 0, 1, 0, 0]):.3f}")

    print("Pairwise(排序):")
    print(f"  RankNet 损失 = {ranknet_loss(np.array([0.9, 0.1, 0.8]), np.array([3, 0, 2])):.4f}")

    print("Listwise(排序):")
    print(f"  NDCG = {ndcg([3.0, 1.0, 2.0, 0.0]):.4f}")

    print("\n小结: 离线评价用于上线前快速筛选模型, 结合 Pointwise/Pairwise/Listwise。")


if __name__ == "__main__":
    main()