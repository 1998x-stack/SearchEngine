# 01_3.2.2_文本质量

"""
Lecture: 3_第三部分_什么决定用户体验？/3.2_内容质量
Content: 01_3.2.2_文本质量
"""

from typing import Dict


class TextQualityScorer:
    """Text-quality score from clarity/completeness/accuracy/value and writing.

    Each dimension 0..1; combined score maps to a quality label.
    """

    DIMS = ["clarity", "completeness", "accuracy", "value", "craft"]

    def __init__(self, weights: Dict[str, float] = None) -> None:
        self.weights = weights or {
            "clarity": 0.25, "completeness": 0.2, "accuracy": 0.25,
            "value": 0.15, "craft": 0.15,
        }

    def score(self, ratings: Dict[str, float]) -> float:
        return sum(self.weights[d] * ratings.get(d, 0.0) for d in self.DIMS)

    @staticmethod
    def label(q: float) -> str:
        if q >= 0.7:
            return "高质量"
        if q >= 0.4:
            return "中质量"
        return "低质量"


def main() -> None:
    print("文本质量评分 Demo")
    scorer = TextQualityScorer()
    docs = {
        "柴犬科普(清晰/全面/准确)": {"clarity": 0.9, "completeness": 0.9,
                              "accuracy": 0.9, "value": 0.85, "craft": 0.8},
        "旅游心情流水账": {"clarity": 0.4, "completeness": 0.2,
                      "accuracy": 0.5, "value": 0.2, "craft": 0.3},
        "拼凑复制+语法错误": {"clarity": 0.2, "completeness": 0.2,
                       "accuracy": 0.2, "value": 0.15, "craft": 0.1},
    }
    for name, ratings in docs.items():
        s = scorer.score(ratings)
        print(f"  {name:<16} 文本质量={s:.2f} -> {scorer.label(s)}")


if __name__ == "__main__":
    main()