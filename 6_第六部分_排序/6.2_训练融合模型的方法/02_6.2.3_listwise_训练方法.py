# 02_6.2.3_listwise_训练方法

"""
Lecture: 6_第六部分_排序/6.2_训练融合模型的方法
Content: 02_6.2.3_listwise_训练方法
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def listwise_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    """ListNet-style: cross entropy between score-softmax and label-softmax."""
    p = softmax(labels)
    q = softmax(scores)
    eps = 1e-12
    return float(-np.sum(p * np.log(np.clip(q, eps, 1))))


class ListwiseRanker:
    """Rank documents as a list using listwise (softmax) cross-entropy."""

    def __init__(self, n_features: int, lr: float = 0.1) -> None:
        self.w = np.zeros(n_features)
        self.lr = lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def fit_one(self, X: np.ndarray, y: np.ndarray, steps: int = 200) -> List[float]:
        losses = []
        for _ in range(steps):
            s = self.predict(X)
            losses.append(listwise_loss(s, y))
            p = softmax(y)
            q = softmax(s)
            grad = (q - p) / len(y)
            self.w -= self.lr * (X.T @ grad)
        return losses


def main() -> None:
    print("Listwise 训练方法 Demo")
    X = np.array([[1.0, 0.2], [0.5, 0.8], [0.9, 0.4]])
    y = np.array([3.0, 1.0, 2.0])           # 相关性标签
    ranker = ListwiseRanker(n_features=2)
    losses = ranker.fit_one(X, y, steps=300)
    print(f"训练 loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"打分顺序: {np.round(ranker.predict(X), 3)}  (与标签相关性对齐)")


if __name__ == "__main__":
    main()