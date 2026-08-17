# 01_1.1.2_什么决定用户满意度？

"""
Lecture: 1_第一部分_搜索引擎基础/1.1_搜索引擎技术概要
Content: 01_1.1.2_什么决定用户满意度？
"""

from typing import Dict


class SatisfactionScore:
    """Weighted satisfaction score from the key quality factors.

    Satisfaction = w1*relevance + w2*quality + w3*recency + w4*personal +
                   w5*geo. Defaults weight relevance most, matching the
                   text (相关性是最重要因素).
    """

    FACTORS = ["relevance", "quality", "recency", "personal", "geo"]

    def __init__(self,
                 w_relevance: float = 0.40,
                 w_quality: float = 0.20,
                 w_recency: float = 0.18,
                 w_personal: float = 0.12,
                 w_geo: float = 0.10) -> None:
        self.weights: Dict[str, float] = {
            "relevance": w_relevance,
            "quality": w_quality,
            "recency": w_recency,
            "personal": w_personal,
            "geo": w_geo,
        }

    def score(self, factors: Dict[str, float]) -> float:
        """Weighted sum of the satisfaction factors (each 0..1)."""
        return sum(self.weights[k] * factors.get(k, 0.0) for k in self.FACTORS)

    def rank(self, docs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Rank documents by satisfaction score."""
        return {name: self.score(f) for name, f in docs.items()}


def main() -> None:
    print("=== 什么决定用户满意度 (Determinants of Satisfaction) Demo ===")

    # 三个候选文档在不同维度上的得分 (0~1)
    engine = SatisfactionScore()
    docs = {
        "Python教程(语义相关, 质量高, 内容新)": {
            "relevance": 0.95, "quality": 0.90, "recency": 0.85,
            "personal": 0.80, "geo": 0.50,
        },
        "Python爬虫科普(部分相关)": {
            "relevance": 0.35, "quality": 0.50, "recency": 0.40,
            "personal": 0.30, "geo": 0.50,
        },
        "Python蛇类百科(不相关)": {
            "relevance": 0.05, "quality": 0.60, "recency": 0.30,
            "personal": 0.10, "geo": 0.50,
        },
    }

    print("各文档的综合满意度分数(越高越好):")
    ranked = engine.rank(docs)
    for name in sorted(ranked, key=ranked.get, reverse=True):
        print(f"  {name:<24} 满意度={ranked[name]:.3f}")

    print("\n结论: 相关性(权重最大)是影响满意度的最重要因素。")


if __name__ == "__main__":
    main()