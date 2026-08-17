# 03_3.1.4_相关性模型的训练

"""
Lecture: 3_第三部分_什么决定用户体验？/3.1_相关性
Content: 03_3.1.4_相关性模型的训练
"""

import numpy as np
from typing import List


def pointwise_mse(logits: np.ndarray, labels: np.ndarray) -> float:
    """Pointwise loss (MSE) — 提升 AUC."""
    return float(np.mean((logits - labels) ** 2))


def pairwise_logistic(logits: np.ndarray, labels: np.ndarray) -> float:
    """Pairwise logistic loss — 提升正逆序比."""
    loss = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] > labels[j]:
                loss += np.log(1 + np.exp(-(logits[i] - logits[j])))
    return float(loss)


def positive_negative_ratio(logits: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    inverted = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] > labels[j]:
                if logits[i] > logits[j]:
                    correct += 1
                else:
                    inverted += 1
    return correct / inverted if inverted else float("inf")


def main() -> None:
    print("相关性模型训练 Demo")
    # 相关性标签与模型 logits(训练后)
    labels = np.array([3.0, 0.0, 2.0, 1.0, 1.0])
    logits = np.array([3.1, -0.2, 1.9, 0.8, 1.2])

    print(f"相关性标签: {labels}")
    print(f"模型输出:   {logits}")
    print(f"Pointwise MSE  = {pointwise_mse(logits, labels):.4f}")
    print(f"Pairwise Loss  = {pairwise_logistic(logits, labels):.4f}")
    print(f"正逆序比 PNR   = {positive_negative_ratio(logits, labels):.4f}")

    print("\n训练步骤: 预训练 -> 后预训练 -> 微调 -> 蒸馏 (见 2.3/3.1.3)。")


if __name__ == "__main__":
    main()