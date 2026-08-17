# 00_2.2.1_pointwise_评价指标

"""
Lecture: 2_第二部分_机器学习基础/2.2_离线评价指标
Content: 00_2.2.1_pointwise_评价指标
"""

import numpy as np
from typing import List


def mse(y: List[float], pred: List[float]) -> float:
    """Regression: mean squared error."""
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def confusion(y: List[int], pred: List[int]) -> tuple:
    """Classification counts given binary labels and predictions."""
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    return tp, fp, tn, fn


def accuracy(y, pred) -> float:
    tp, fp, tn, fn = confusion(y, pred)
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0


def precision(y, pred) -> float:
    tp, fp, _, _ = confusion(y, pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y, pred) -> float:
    tp, _, _, fn = confusion(y, pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1(y, pred) -> float:
    p, r = precision(y, pred), recall(y, pred)
    return 2 * p * r / (p + r) if p + r else 0.0


def auc(y: List[int], score: List[float]) -> float:
    """Area under ROC using rank-based Mann-Whitney U."""
    order = np.argsort(-np.asarray(score))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[np.asarray(y) == 1]
    neg = len(y) - len(pos_ranks)
    if len(pos_ranks) == 0 or neg == 0:
        return 0.5
    return float((pos_ranks.sum() - len(pos_ranks) * (len(pos_ranks) + 1) / 2) / (len(pos_ranks) * neg))


def main() -> None:
    print("Pointwise 评价指标 Demo")
    y = [1, 0, 1, 1, 0, 0, 1, 0]
    pred = [1, 0, 1, 0, 0, 1, 1, 0]
    score = [0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.85, 0.05]
    print(f"准确率={accuracy(y, pred):.3f} 精确率={precision(y, pred):.3f} "
          f"召回率={recall(y, pred):.3f} F1={f1(y, pred):.3f} AUC={auc(y, score):.3f}")

    print("回归 MSE:")
    print(f"  MSE = {mse([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0]):.4f}")


if __name__ == "__main__":
    main()