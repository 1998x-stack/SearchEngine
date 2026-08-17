# 02_3.5.3_粗排点击率模型

"""
Lecture: 3_第三部分_什么决定用户体验？/3.5_个性化与点击率预估
Content: 02_3.5.3_粗排点击率模型
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TwoTowerCTR:
    """Two-tower (双塔) model: query/user tower + document tower.

    CTR = sigmoid( query_vec · doc_vec ). Fast, used for 粗排.
    """

    def __init__(self, n_query: int, n_doc: int, dim: int = 8, lr: float = 0.1) -> None:
        rng = np.random.default_rng(0)
        self.Wq = rng.normal(0, 0.2, (n_query, dim))
        self.Wd = rng.normal(0, 0.2, (n_doc, dim))
        self.lr = lr

    def predict_proba(self, Q: np.ndarray, D: np.ndarray, negative_sampling_ratio: float = 1.0) -> np.ndarray:
        """CTR = sigmoid(Q_tower @ D_tower)."""
        qv = Q @ self.Wq
        dv = D @ self.Wd
        return sigmoid(np.sum(qv * dv, axis=1))

    def fit(self, Q: np.ndarray, D: np.ndarray, y: np.ndarray, epochs: int = 150) -> List[float]:
        losses = []
        n = len(y)
        for _ in range(epochs):
            qv = Q @ self.Wq
            dv = D @ self.Wd
            out = sigmoid(np.sum(qv * dv, axis=1))
            loss = float(-np.mean(y * np.log(np.clip(out, 1e-12, 1)) +
                                  (1 - y) * np.log(np.clip(1 - out, 1e-12, 1))))
            losses.append(loss)
            d = out - y                      # (n,)
            self.Wq -= self.lr * (Q.T @ (d[:, None] * dv)) / n
            self.Wd -= self.lr * (D.T @ (d[:, None] * qv)) / n
        return losses


def main() -> None:
    print("粗排点击率模型 (Two-Tower) Demo")
    rng = np.random.default_rng(2)
    Q = rng.normal(0, 1, (300, 4))
    D = rng.normal(0, 1, (300, 4))
    y = (Q[:, 0] * D[:, 0] + Q[:, 1] * D[:, 1] > 0).astype(int)

    model = TwoTowerCTR(n_query=4, n_doc=4)
    losses = model.fit(Q, D, y, epochs=300)
    pred = model.predict_proba(Q, D) > 0.5
    print(f"CTR loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"粗排模型准确率 = {np.mean(pred == y)*100:.1f}%")
    print("说明: 双塔分别编码查询/用户与文档, 内积+Sigmoid 预测 CTR, 用于粗排。")


if __name__ == "__main__":
    main()