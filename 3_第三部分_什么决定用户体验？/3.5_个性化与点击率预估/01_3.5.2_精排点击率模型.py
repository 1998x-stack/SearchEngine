# 01_3.5.2_精排点击率模型

"""
Lecture: 3_第三部分_什么决定用户体验？/3.5_个性化与点击率预估
Content: 01_3.5.2_精排点击率模型
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class CTRModel:
    """A small DNN (2 hidden layers) predicting click probability.

    Demonstrates the fine-ranking (精排) CTR model structure: dense
    features in, sigmoid probability out, trained with cross entropy.
    """

    def __init__(self, n_in: int, hidden: int = 8, lr: float = 0.1) -> None:
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.2, (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.2, (hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1 + self.b1)          # 隐层
        return sigmoid(h @ self.W2 + self.b2).ravel()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> List[float]:
        losses = []
        n = len(X)
        for _ in range(epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            out = sigmoid(h @ self.W2 + self.b2).ravel()
            loss = float(-np.mean(y * np.log(np.clip(out, 1e-12, 1)) +
                                  (1 - y) * np.log(np.clip(1 - out, 1e-12, 1))))
            losses.append(loss)
            d_out = (out - y) / n
            self.W2 -= self.lr * (h.T @ d_out[:, None])
            self.b2 -= self.lr * d_out.sum()
            d_h = (d_out[:, None] @ self.W2.T) * (1 - h ** 2)
            self.W1 -= self.lr * (X.T @ d_h)
            self.b1 -= self.lr * d_h.sum(axis=0)
        return losses


def main() -> None:
    print("精排点击率模型 (DNN) Demo")
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (400, 5))                  # 特征
    y = (X[:, 0] + X[:, 1] > 0).astype(int)         # 点击标签
    model = CTRModel(n_in=5)
    losses = model.fit(X, y, epochs=300)
    pred = (model.predict_proba(X) > 0.5)
    print(f"CTR loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"点击率模型准确率 = {np.mean(pred == y)*100:.1f}%")
    print("说明: 精排用 DNN 对特征建模, 预测点击概率用于最终排序。")


if __name__ == "__main__":
    main()