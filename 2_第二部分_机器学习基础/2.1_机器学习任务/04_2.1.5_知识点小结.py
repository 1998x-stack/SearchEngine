# 04_2.1.5_知识点小结

"""
Lecture: 2_第二部分_机器学习基础/2.1_机器学习任务
Content: 04_2.1.5_知识点小结
"""

import numpy as np


def sigmoid(z: float) -> float:
    """Sigmoid for binary classification."""
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z) -> np.ndarray:
    """Softmax for multi-class classification."""
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def mse(y, pred) -> float:
    """Mean squared error for regression."""
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def main() -> None:
    print("=== 机器学习任务小结 (ML Task Summary) Demo ===")
    print(f"二分类 Sigmoid(0.5) = {sigmoid(0.5):.4f}")
    print(f"多分类 softmax([1,2,3]) = {np.round(softmax([1, 2, 3]), 4)}")
    print(f"回归 MSE([1,2,3],[1.1,1.9,3.2]) = {mse([1, 2, 3], [1.1, 1.9, 3.2]):.4f}")

    print("\n四类机器学习任务:")
    print("  二分类(Logistic), 多分类(Softmax), 回归(线性回归), 排序(RankNet/NDCG)")


if __name__ == "__main__":
    main()