# 04_3.1.5_知识点小结

"""
Lecture: 3_第三部分_什么决定用户体验？/3.1_相关性
Content: 04_3.1.5_知识点小结
"""

import numpy as np
from typing import List


def auc(y: List[int], score: List[float]) -> float:
    order = np.argsort(-np.asarray(score))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    pos = ranks[np.asarray(y) == 1]
    neg = len(y) - len(pos)
    if neg == 0 or len(pos) == 0:
        return 0.5
    return float((pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * neg))


def tier_for_ratio(ratio: float) -> str:
    if ratio >= 0.5:
        return "高相关"
    if ratio >= 0.2:
        return "中相关"
    if ratio > 0.0:
        return "低相关"
    return "无相关"


def main() -> None:
    print("=== 相关性知识点小结 (Relevance Summary) Demo ===")
    y = [1, 0, 1, 1, 0]
    score = [0.9, 0.1, 0.8, 0.7, 0.2]
    print(f"相关性模型 AUC = {auc(y, score):.3f} (越高越准)")

    print("\n相关性分档:")
    for r in [0.9, 0.3, 0.1, 0.0]:
        print(f"  满足比例 {r:.1f} -> {tier_for_ratio(r)}")

    print("\n小结: 相关性是核心指标; 文本匹配(TF-IDF/BM25) + 语义匹配(BERT)。")


if __name__ == "__main__":
    main()