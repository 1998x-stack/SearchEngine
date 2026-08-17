# 00_4.2.1_词权重的定义与标注方法

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.2_词权重
Content: 00_4.2.1_词权重的定义与标注方法
"""

import math
from typing import Dict, List


def idf_weights(corpus: List[List[str]]) -> Dict[str, float]:
    """Word weight by IDF: 越稀有越重要."""
    df: Dict[str, int] = {}
    for doc in corpus:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1
    n = len(corpus)
    return {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}


def drop_low(terms: List[str], weights: Dict[str, float],
             threshold: float) -> List[str]:
    """丢词召回: 丢弃权重低于阈值的词."""
    return [t for t in terms if weights.get(t, 0) >= threshold]


def main() -> None:
    print("词权重的定义与标注方法 Demo")
    corpus = [
        ["冬季", "卫衣", "推荐", "男士"],
        ["冬季", "外套", "推荐"],
        ["卫衣", "穿搭", "潮流"],
    ]
    weights = idf_weights(corpus)
    query = ["冬季", "卫衣", "推荐"]
    print("词权重(IDF):")
    for t in query:
        print(f"  {t:<4} 权重={weights.get(t, 0):.3f}")

    kept = drop_low(query, weights, 1.1)
    print(f"丢词阈值1.1后保留: {kept}  (用于增加召回复盖)")


if __name__ == "__main__":
    main()