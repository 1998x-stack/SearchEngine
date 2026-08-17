# 02_3.3.3_下游链路的承接

"""
Lecture: 3_第三部分_什么决定用户体验？/3.3_时效性
Content: 02_3.3.3_下游链路的承接
"""

import numpy as np
from typing import List


def combine(relevance: np.ndarray, recency: np.ndarray, recency_weight: float) -> np.ndarray:
    """Combine relevance and recency into a final ranking score."""
    return relevance * (1 - recency_weight) + recency * recency_weight


def main() -> None:
    print("下游链路承接: 时效性意图传递到召回/排序 Demo")
    # QP 识别出 强时效 查询
    recency_weight = 0.6

    relevance = np.array([0.9, 0.7, 0.6, 0.4])   # 相关性分
    recency = np.array([0.3, 0.9, 0.5, 0.1])     # 时效性分(新文档高)
    docs = ["老文章A", "最新消息B", "一般文章C", "陈旧内容D"]

    final = combine(relevance, recency, recency_weight)
    order = np.argsort(-final)
    print(f"时效权重 = {recency_weight}")
    print("文档 | 相关性 | 时效性 | 综合分")
    for i in range(len(docs)):
        print(f"  {docs[i]:<10} {relevance[i]:.1f}   {recency[i]:.1f}   {final[i]:.3f}")
    print(f"最终排序: {[docs[i] for i in order]}")

    print("\n策略: QP 识别意图 -> 召回优先最新索引 -> 排序提高时效权重。")


if __name__ == "__main__":
    main()