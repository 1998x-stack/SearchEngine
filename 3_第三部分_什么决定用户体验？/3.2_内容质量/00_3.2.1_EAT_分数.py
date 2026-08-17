# 00_3.2.1_EAT_分数

"""
Lecture: 3_第三部分_什么决定用户体验？/3.2_内容质量
Content: 00_3.2.1_EAT_分数
"""

from typing import Dict


class EATScorer:
    """E-A-T score = weighted Expertise/Authoritativeness/Trustworthiness.

    Each pillar is a 0..1 rating; the combined score places content into
    a quality tier.
    """

    def __init__(self,
                 w_expertise: float = 0.4,
                 w_authority: float = 0.3,
                 w_trust: float = 0.3) -> None:
        self.weights = {"expertise": w_expertise,
                        "authority": w_authority,
                        "trust": w_trust}

    def score(self, ratings: Dict[str, float]) -> float:
        return sum(self.weights[k] * ratings.get(k, 0.0)
                   for k in self.weights)

    @staticmethod
    def tier(eat: float) -> str:
        if eat >= 0.75:
            return "高质量"
        if eat >= 0.5:
            return "中质量"
        if eat > 0.2:
            return "低质量"
        return "劣质"


def main() -> None:
    print("EAT 分数 (专业/权威/可信) Demo")
    scorer = EATScorer()
    docs = {
        "权威机构发布(专业/权威/可信均高)": {"expertise": 0.9, "authority": 0.95, "trust": 0.9},
        "普通博主(中等)": {"expertise": 0.5, "authority": 0.4, "trust": 0.55},
        "营销软文(可信低)": {"expertise": 0.4, "authority": 0.3, "trust": 0.15},
    }
    for name, ratings in docs.items():
        s = scorer.score(ratings)
        print(f"  {name:<20} EAT={s:.2f} -> {scorer.tier(s)}")


if __name__ == "__main__":
    main()