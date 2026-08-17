# 00_6.2.1_pointwise_训练方法

"""
Lecture: 6_第六部分_排序/6.2_训练融合模型的方法
Content: 00_6.2.1_pointwise_训练方法
"""

import numpy as np
from typing import List


class PointwiseRanker:
    """Pointwise training: predict each doc's relevance independently (MSE)."""

    def __init__(self, n_features: int, lr: float = 0.05) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> List[float]:
        losses, n = [], len(X)
        for _ in range(epochs):
            pred = self.predict(X)
            losses.append(float(np.mean((pred - y) ** 2)))
            grad = (2 / n) * (pred - y)
            self.w -= self.lr * (X.T @ grad)
            self.b -= self.lr * grad.sum()
        return losses


def main() -> None:
    print("Pointwise 训练方法 Demo")
    X = np.array([[1.0, 0.2], [0.5, 0.8], [0.9, 0.4], [0.2, 0.1]])
    y = np.array([3.0, 1.0, 2.0, 0.0])          # 相关性标签
    ranker = PointwiseRanker(n_features=2)
    losses = ranker.fit(X, y, epochs=300)
    pred = ranker.predict(X)
    print(f"训练 loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"预测得分: {np.round(pred, 3)}  (用于排序)")


if __name__ == "__main__":
    main()