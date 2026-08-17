# 03_3.5.4_模型训练

"""
Lecture: 3_第三部分_什么决定用户体验？/3.5_个性化与点击率预估
Content: 03_3.5.4_模型训练
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TwoTowerCTR:
    """Two-tower CTR model trained with clicks + teacher distillation.

    Loss = click cross-entropy + 0.1 * distillation term that pulls the
    student's logits toward the teacher's relevance logits (simplified).
    """

    def __init__(self, dim_in: int, dim: int = 6, lr: float = 0.1) -> None:
        rng = np.random.default_rng(0)
        self.Wq = rng.normal(0, 0.2, (dim_in, dim))
        self.Wd = rng.normal(0, 0.2, (dim_in, dim))
        self.lr = lr

    def logits(self, Q: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Raw inner product of query and doc tower vectors."""
        return np.sum((Q @ self.Wq) * (D @ self.Wd), axis=1)

    def predict_proba(self, q: np.ndarray, d: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(q, d))

    def fit(self, Q: np.ndarray, D: np.ndarray, y: np.ndarray,
            teacher_logits: np.ndarray, epochs: int = 100,
            distill_weight: float = 0.1) -> List[float]:
        """Train with negative sampling (balanced labels) + distillation."""
        losses = []
        n = len(y)
        for _ in range(epochs):
            qv, dv = Q @ self.Wq, D @ self.Wd
            logits = np.sum(qv * dv, axis=1)
            out = sigmoid(logits)
            ce = float(-np.mean(y * np.log(np.clip(out, 1e-12, 1)) +
                                (1 - y) * np.log(np.clip(1 - out, 1e-12, 1))))
            da = float(np.mean((logits - teacher_logits) ** 2))
            loss = ce + distill_weight * da
            losses.append(loss)

            grad_logits = (out - y) / n + distill_weight * (2.0 / n) * (logits - teacher_logits)
            self.Wq -= self.lr * (Q.T @ (grad_logits[:, None] * dv))
            self.Wd -= self.lr * (D.T @ (grad_logits[:, None] * qv))
        return losses


def main() -> None:
    print("粗排点击率模型训练 (含蒸馏) Demo")
    rng = np.random.default_rng(3)
    Q = rng.normal(0, 1, (300, 4))
    D = rng.normal(0, 1, (300, 4))
    y = (Q[:, 0] * D[:, 0] + Q[:, 1] * D[:, 1] > 0).astype(int)
    teacher_logits = 1.5 * rng.normal(0, 1, 300)   # 精排教师模型的输出

    model = TwoTowerCTR(dim_in=4)
    losses = model.fit(Q, D, y, teacher_logits, epochs=300)
    acc = np.mean((model.predict_proba(Q, D) > 0.5) == (y == 1))
    print(f"训练 loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"点击率预测准确率 = {acc * 100:.1f}%")
    print("说明: 使用负采样处理不平衡, 交叉熵 + 教师蒸馏综合训练。")


if __name__ == "__main__":
    main()