# 01_2.2.2_pairwise_评价指标

"""
Lecture: 2_第二部分_机器学习基础/2.2_离线评价指标
Content: 01_2.2.2_pairwise_评价指标
"""

import numpy as np
from typing import List


def ranknet_loss(pred: np.ndarray, labels: np.ndarray) -> float:
    """RankNet loss = sum_{(i,j): y_i > y_j} log(1 + exp(-(p_i - p_j)))."""
    loss = 0.0
    pairs = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] > labels[j]:
                loss += np.log(1 + np.exp(-(pred[i] - pred[j])))
                pairs += 1
    return float(loss) if pairs else 0.0


def positive_negative_ratio(pred: np.ndarray, labels: np.ndarray) -> float:
    """PNR = number of correctly-ordered pairs / number of inverted pairs."""
    correct = 0
    inverted = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] > labels[j]:
                if pred[i] > pred[j]:
                    correct += 1
                elif pred[i] < pred[j]:
                    inverted += 1
    return correct / inverted if inverted > 0 else float("inf")


def main() -> None:
    print("Pairwise 评价指标 Demo")
    # 5 个文档: 特征作为评分, 标签为相关性
    pred = np.array([0.9, 0.1, 0.8, 0.4, 0.6])
    labels = np.array([3.0, 0.0, 2.0, 1.0, 1.0])

    print(f"排序评分(pred) = {pred}")
    print(f"相关性标签(y)  = {labels}")
    print(f"RankNet 损失 = {ranknet_loss(pred, labels):.4f}")
    print(f"正逆序比 PNR = {positive_negative_ratio(pred, labels):.4f}")


if __name__ == "__main__":
    main()