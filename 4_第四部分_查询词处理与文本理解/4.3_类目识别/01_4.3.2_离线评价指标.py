# 01_4.3.2_离线评价指标

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.3_类目识别
Content: 01_4.3.2_离线评价指标
"""

import numpy as np
from typing import List


def macro_f1(y: List[List[int]], pred: List[List[int]]) -> float:
    """Macro F1: mean per-label F1 (one-vs-rest)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    f1s = []
    for c in range(y.shape[1]):
        tp = int(np.sum((y[:, c] == 1) & (pred[:, c] == 1)))
        fp = int(np.sum((y[:, c] == 0) & (pred[:, c] == 1)))
        fn = int(np.sum((y[:, c] == 1) & (pred[:, c] == 0)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def micro_f1(y: List[List[int]], pred: List[List[int]]) -> float:
    """Micro F1: aggregate TP/FP/FN across all labels then F1."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> None:
    print("多标签离线评价指标 (Multi-Label Metrics) Demo")
    y = [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
    pred = [[1, 0, 1], [1, 1, 0], [0, 0, 1]]
    print(f"真实标签: {y}")
    print(f"预测标签: {pred}")
    print(f"Macro F1 = {macro_f1(y, pred):.4f}")
    print(f"Micro F1 = {micro_f1(y, pred):.4f}")


if __name__ == "__main__":
    main()