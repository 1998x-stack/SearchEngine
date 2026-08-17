# 01_2.1.2_多分类任务

"""
Lecture: 2_第二部分_机器学习基础/2.1_机器学习任务
Content: 01_2.1.2_多分类任务
"""

import numpy as np
from typing import List


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def cross_entropy(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean categorical cross-entropy between one-hot labels and probs."""
    eps = 1e-12
    return float(-np.mean(np.sum(y_true * np.log(np.clip(probs, eps, 1.0)), axis=-1)))


class SoftmaxClassifier:
    """Multi-class softmax classifier trained by gradient descent."""

    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.1) -> None:
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = learning_rate

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probability distribution per sample."""
        return softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax class index per sample."""
        return np.argmax(self.predict_proba(X), axis=-1)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100) -> List[float]:
        """Train; labels Y one-hot. Returns loss history."""
        losses = []
        n = len(X)
        for _ in range(epochs):
            probs = self.predict_proba(X)
            losses.append(cross_entropy(Y, probs))
            grad = (probs - Y) / n
            self.W -= self.lr * (X.T @ grad)
            self.b -= self.lr * grad.sum(axis=0)
        return losses


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return float(np.mean(pred == y))


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 (treating each class one-vs-rest)."""
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y == c) & (pred == c)))
        fp = int(np.sum((y != c) & (pred == c)))
        fn = int(np.sum((y == c) & (pred != c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    print("多分类任务 (Multi-Class Classification) Demo")
    # 两个特征, 三个类别 (线性可分 toy)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                  [5, 5], [5, 6], [6, 5], [6, 6],
                  [9, 0], [9, 1], [8, 0], [8, 1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    Y = np.eye(3)[y]

    model = SoftmaxClassifier(n_features=2, n_classes=3)
    losses = model.fit(X, Y, epochs=100)
    print(f"初始loss={losses[0]:.4f}, 最终loss={losses[-1]:.4f}")

    pred = model.predict(X)
    print(f"准确率 = {accuracy(pred, y)*100:.1f}%")
    print(f"宏平均F1 = {macro_f1(y, pred, 3):.4f}")
    print(f"新样本 (6,5) 预测类别 = {model.predict(np.array([[6, 5]]))[0]}")


if __name__ == "__main__":
    main()