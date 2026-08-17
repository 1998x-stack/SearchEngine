# 02_2.3.3_微调

"""
Lecture: 2_第二部分_机器学习基础/2.3_NLP模型的训练
Content: 02_2.3.3_微调
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List


class FineTuner:
    """Demonstrates fine-tuning a pretrained model on labeled data.

    A small logistic head is initialized from a 'pretrained' feature
    extractor and trained on high-quality labeled examples.
    """

    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1) -> None:
        # 预训练参数(此处用随机初始化模拟加载的预训练权重)
        self.W = np.random.default_rng(0).normal(0, 0.3, (n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.lr = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.W + self.b
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        Y = np.eye(self.W.shape[1])[y]
        losses = []
        n = len(X)
        for _ in range(epochs):
            probs = self.predict_proba(X)
            loss = -np.mean(np.sum(Y * np.log(np.clip(probs, 1e-12, 1)), axis=-1))
            losses.append(float(loss))
            grad = (probs - Y) / n
            self.W -= self.lr * (X.T @ grad)
            self.b -= self.lr * grad.sum(axis=0)
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)


def main() -> None:
    print("微调 (Fine-tuning) Demo")
    # 加载预训练特征(特征维度高), 在少量标注数据上微调
    X = np.array([[0.2, 0.9], [0.1, 0.8], [0.9, 0.2], [0.8, 0.1],
                  [0.7, 0.7], [0.3, 0.3]])
    y = np.array([0, 0, 1, 1, 1, 0])
    model = FineTuner(n_features=2, n_classes=2)
    losses = model.fit(X, y, epochs=120)
    print(f"微调 loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"微调后准确率 = {np.mean(model.predict(X) == y)*100:.1f}%")
    print("说明: 微调在预训练基础上用高质量人工标注数据适配下游任务。")


if __name__ == "__main__":
    main()