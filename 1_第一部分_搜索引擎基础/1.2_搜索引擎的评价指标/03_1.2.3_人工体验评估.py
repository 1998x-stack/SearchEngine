# 03_1.2.3_人工体验评估

"""
Lecture: 1_第一部分_搜索引擎基础/1.2_搜索引擎的评价指标
Content: 03_1.2.3_人工体验评估
"""

import numpy as np
from typing import List


def dcg_at_k(scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k.

    DCG@k = sum_i score_i / log2(i+1), 1-based rank.

    Args:
        scores: relevance scores in ranking order.
        k: number of positions to consider.

    Returns:
        DCG@k as a float.
    """
    scores = scores[:k]
    positions = np.arange(1, len(scores) + 1)
    return float(np.sum(np.asarray(scores) / np.log2(positions + 1)))


def main() -> None:
    print("=== 人工体验评估 (Manual Experience Evaluation) Demo ===")

    # 实验组与对照组的相关性评分(人工标注, 越靠前越相关越好)
    control = [3.0, 2.0, 1.0, 1.0]
    treatment = [3.0, 3.0, 2.0, 1.0]

    print("DCG@4 计算:")
    print(f"  对照组 DCG@4 = {dcg_at_k(control, 4):.4f}")
    print(f"  实验组 DCG@4 = {dcg_at_k(treatment, 4):.4f}")

    print("\nSBS(Side by Side)评估思路:")
    print("  在相同用户画像/场景下, 对比实验组与对照组结果页。")
    print("  即使点击率提升, 也可能相关性下降 -> 需要人工评估兜底。")


if __name__ == "__main__":
    main()