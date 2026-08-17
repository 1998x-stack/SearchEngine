# 00_4.3.1_多标签分类模型

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.3_类目识别
Content: 00_4.3.1_多标签分类模型
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(y * np.log(np.clip(p, 1e-12, 1)) +
                        (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))


class MultiLabelClassifier:
    """Multi-label classifier: one sigmoid per category (threshold at 0.5)."""

    def __init__(self, n_in: int, n_labels: int, lr: float = 0.1) -> None:
        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.2, (n_in, n_labels))
        self.b = np.zeros(n_labels)
        self.lr = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.W + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 200) -> List[float]:
        losses, n = [], len(X)
        for _ in range(epochs):
            p = self.predict_proba(X)
            losses.append(binary_cross_entropy(Y, p))
            grad = (p - Y) / n
            self.W -= self.lr * (X.T @ grad)
            self.b -= self.lr * grad.sum(axis=0)
        return losses


def main() -> None:
    print("多标签分类模型 (Multi-Label Classification) Demo")
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (300, 4))
    # 3 个标签: 美食/娱乐/科技, 一个样本可属于多个
    Y = (X[:, :3] > 0).astype(int)
    model = MultiLabelClassifier(n_in=4, n_labels=3)
    losses = model.fit(X, Y, epochs=200)
    pred = model.predict(X)
    acc = np.mean(pred == Y)
    print(f"训练 loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"多标签预测准确(逐标签一致)率 = {acc * 100:.1f}%")


if __name__ == "__main__":
    main()