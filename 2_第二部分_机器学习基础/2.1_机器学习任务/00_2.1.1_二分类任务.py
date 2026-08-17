# 00_2.1.1_二分类任务

"""
Lecture: 2_第二部分_机器学习基础/2.1_机器学习任务
Content: 00_2.1.1_二分类任务
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean binary cross-entropy loss."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


class LinearClassifier:
    """Minimal logistic-regression classifier trained by gradient descent."""

    def __init__(self, n_features: int, learning_rate: float = 0.1) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class."""
        return sigmoid(X @ self.w + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 40) -> List[float]:
        """Train with gradient descent; return loss history."""
        losses = []
        for _ in range(epochs):
            pred = self.predict_proba(X)
            losses.append(binary_cross_entropy(y, pred))
            grad_w = X.T @ (pred - y) / len(y)
            grad_b = float(np.mean(pred - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return losses


def main() -> None:
    print("二分类任务 (Binary Classification) Demo")
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0],
                  [0.0, 1.0], [2.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    y = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    clf = LinearClassifier(n_features=2)
    losses = clf.fit(X, y, epochs=40)
    print(f"初始loss={losses[0]:.4f}, 最终loss={losses[-1]:.4f}")
    for point in [(0, 0), (1.5, 1.5), (0.5, 2.5)]:
        row = np.array([point], dtype=float)   # 2D: (1, n_features)
        p = clf.predict_proba(row)[0]
        print(f"  特征 {point} -> 预测概率 {p:.3f}")


if __name__ == "__main__":
    main()