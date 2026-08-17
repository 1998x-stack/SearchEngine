# 02_2.2.3_listwise_评价指标

"""
Lecture: 2_第二部分_机器学习基础/2.2_离线评价指标
Content: 02_2.2.3_listwise_评价指标
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List


def dcg(scores: List[float], k: int = None) -> float:
    """DCG@k = sum_i (2^y_i - 1) / log2(i + 1)."""
    s = np.asarray(scores)
    if k is not None:
        s = s[:k]
    positions = np.arange(1, len(s) + 1)
    return float(np.sum((2 ** s - 1) / np.log2(positions + 1)))


def idcg(scores: List[float], k: int = None) -> float:
    """IDCG = DCG of the ideal (descending) ordering."""
    return dcg(sorted(scores, reverse=True), k)


def ndcg(scores: List[float], k: int = None) -> float:
    """NDCG = DCG / IDCG in [0, 1]."""
    denom = idcg(scores, k)
    return dcg(scores, k) / denom if denom > 0 else 0.0


def main() -> None:
    print("Listwise 评价指标 Demo")
    ranking = [3.0, 1.0, 2.0, 0.0, 1.0]  # 按当前排序的相关性分数
    print(f"相关性分数(按当前排序): {ranking}")
    print(f"DCG@5   = {dcg(ranking):.4f}")
    print(f"IDCG@5  = {idcg(ranking):.4f}")
    print(f"NDCG@5  = {ndcg(ranking):.4f}")


if __name__ == "__main__":
    main()