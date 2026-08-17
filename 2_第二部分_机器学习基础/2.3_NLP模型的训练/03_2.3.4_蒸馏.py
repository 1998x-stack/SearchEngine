# 03_2.3.4_蒸馏

"""
Lecture: 2_第二部分_机器学习基础/2.3_NLP模型的训练
Content: 03_2.3.4_蒸馏
"""

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LogisticScorer:
    """A tiny logistic scorer used for teacher/student models."""

    def __init__(self, n_features: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.5, n_features)
        self.b = 0.0

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw logit, as a stand-in for a relevance score."""
        return X @ self.w + self.b


def distill(student: LogisticScorer, teacher_logits: np.ndarray,
            X: np.ndarray, steps: int = 200, lr: float = 0.1) -> None:
    """Fit student logits toward teacher's soft outputs (pointwise MSE)."""
    for _ in range(steps):
        out = student.score(X)
        loss = float(np.mean((out - teacher_logits) ** 2))
        grad = (2 / len(X)) * (out - teacher_logits)
        student.w -= lr * (X.T @ grad)
        student.b -= lr * grad.sum()
    print(f"  蒸馏最终(soft)损失 = {loss:.4f}")


def main() -> None:
    print("知识蒸馏 (Knowledge Distillation) Demo")
    X = np.random.default_rng(7).normal(0, 1, (100, 4))

    teacher = LogisticScorer(n_features=4, seed=1)   # 大模型
    student = LogisticScorer(n_features=4, seed=2)   # 小模型
    teacher_logits = teacher.score(X)

    print("大模型对小模型打分(soft labels):")
    distill(student, teacher_logits, X)
    print(f"  大模型与蒸馏后小模型输出相关性 = "
          f"{np.corrcoef(teacher_logits, student.score(X))[0, 1]:.4f}")
    print("说明: 蒸馏用大模型输出做监督, 让小模型近似大模型, 节省线上推理资源。")


if __name__ == "__main__":
    main()