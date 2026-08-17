# 01_6.2.2_pairwise_训练方法

"""
Lecture: 6_第六部分_排序/6.2_训练融合模型的方法
Content: 01_6.2.2_pairwise_训练方法
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List, Tuple


class PairwiseRanker:
    """Pairwise training with logistic loss over relevant pairs.

    Uses gradient of log(1 + exp(-(p_i - p_j))) for pairs with y_i > y_j.
    """

    def __init__(self, n_features: int, lr: float = 0.5) -> None:
        self.w = np.zeros(n_features)
        self.lr = lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    @staticmethod
    def pairs(labels: np.ndarray) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(len(labels)) for j in range(len(labels))
                if labels[i] > labels[j]]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300) -> List[float]:
        plist = self.pairs(y)
        losses = []
        for _ in range(epochs):
            p = self.predict(X)
            loss = 0.0
            grad = np.zeros_like(self.w)
            for i, j in plist:
                s = p[i] - p[j]
                loss += np.log(1 + np.exp(-s))
                w = -1.0 / (1 + np.exp(s))
                grad += w * (X[i] - X[j])
            losses.append(float(loss))
            self.w -= self.lr * grad
        return losses


def main() -> None:
    print("Pairwise 训练方法 Demo")
    X = np.array([[1.0, 0.2], [0.5, 0.8], [0.9, 0.4], [0.2, 0.1]])
    y = np.array([3.0, 1.0, 2.0, 0.0])
    ranker = PairwiseRanker(n_features=2)
    losses = ranker.fit(X, y, epochs=300)
    pred = ranker.predict(X)
    print(f"训练 loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"学到的打分顺序: {np.round(pred, 3)}")


if __name__ == "__main__":
    main()