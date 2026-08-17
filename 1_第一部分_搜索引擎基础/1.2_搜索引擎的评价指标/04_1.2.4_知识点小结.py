# 04_1.2.4_知识点小结

"""
Lecture: 1_第一部分_搜索引擎基础/1.2_搜索引擎的评价指标
Content: 04_1.2.4_知识点小结
"""

import numpy as np
from typing import List


def ctr(clicks: int, impressions: int) -> float:
    """点击率 = 点击次数 / 曝光次数."""
    return clicks / impressions if impressions else 0.0


def dcg(scores: List[float]) -> float:
    """DCG = sum_i score_i / log2(i+1)."""
    positions = np.arange(1, len(scores) + 1)
    return float(np.sum(np.asarray(scores) / np.log2(positions + 1)))


def ndcg(scores: List[float]) -> float:
    """NDCG = DCG 除以理想(降序)DCG, 取值0~1."""
    ideal = dcg(sorted(scores, reverse=True))
    return dcg(scores) / ideal if ideal > 0 else 0.0


def main() -> None:
    print("=== 搜索引擎评价指标小结 (Evaluation Metrics Summary) Demo ===")

    print("核心指标: 用户规模与留存 (DAU/WAU/MAU, 留存率)")
    print(f"  点击率示例: 200/10000 = {ctr(200, 10_000)*100:.2f}%")

    print("\n人工评估: DCG 与 NDCG")
    ranking = [3.0, 1.0, 2.0, 1.0]
    print(f"  某结果页相关性评分(按排名): {ranking}")
    print(f"  DCG  = {dcg(ranking):.4f}")
    print(f"  NDCG = {ndcg(ranking):.4f}")

    print("\n小结: 核心指标看整体, 中间指标看过程, 人工评估看质量。")


if __name__ == "__main__":
    main()