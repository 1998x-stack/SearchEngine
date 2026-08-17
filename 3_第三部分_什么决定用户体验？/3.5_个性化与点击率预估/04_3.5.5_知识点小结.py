# 04_3.5.5_知识点小结

"""
Lecture: 3_第三部分_什么决定用户体验？/3.5_个性化与点击率预估
Content: 04_3.5.5_知识点小结
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def auc(y: List[int], score: List[float]) -> float:
    order = np.argsort(-np.asarray(score))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    pos = ranks[np.asarray(y) == 1]
    neg = len(y) - len(pos)
    if len(pos) == 0 or neg == 0:
        return 0.5
    return float((pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * neg))


def main() -> None:
    print("=== 个性化与点击率预估小结 (CTR Summary) Demo ===")
    y = [1, 0, 1, 1, 0, 1, 0]
    score = [0.9, 0.2, 0.8, 0.7, 0.1, 0.75, 0.3]
    print(f"点击率模型 AUC = {auc(y, score):.3f}")
    print(f"Sigmoid 示例: sigmoid(0.6) = {sigmoid(np.array([0.6]))[0]:.4f}")

    print("\n个性化与点击率链路:")
    print("  特征工程(查询/用户/文档/场景/统计) -> 粗排(双塔) -> 精排(DNN) -> 综合排序")


if __name__ == "__main__":
    main()