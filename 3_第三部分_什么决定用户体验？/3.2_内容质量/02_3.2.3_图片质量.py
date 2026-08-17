# 02_3.2.3_图片质量

"""
Lecture: 3_第三部分_什么决定用户体验？/3.2_内容质量
Content: 02_3.2.3_图片质量
"""

from typing import Dict


class ImageQualityScorer:
    """Image quality from sharpness/noise/exposure/depth-of-field/saturation/clarity.

    Each dimension rated 1..5 (差..优); the mean maps to a label.
    """

    DIMS = ["sharpness", "noise", "exposure", "depth_of_field",
            "saturation", "clarity"]

    def score(self, ratings: Dict[str, float]) -> float:
        """Mean of the 1..5 dimension ratings."""
        return sum(ratings.get(d, 0.0) for d in self.DIMS) / len(self.DIMS)

    @staticmethod
    def label(q: float) -> str:
        if q >= 4.5:
            return "优"
        if q >= 3.5:
            return "中优"
        if q >= 2.5:
            return "中"
        return "差"


def main() -> None:
    print("图片质量评分 Demo")
    scorer = ImageQualityScorer()
    images = {
        "专业摄影图": {"sharpness": 5, "noise": 4, "exposure": 5,
                  "depth_of_field": 4, "saturation": 5, "clarity": 5},
        "正常手机图": {"sharpness": 4, "noise": 3, "exposure": 4,
                   "depth_of_field": 3, "saturation": 4, "clarity": 4},
        "模糊噪声图": {"sharpness": 1, "noise": 1, "exposure": 2,
                   "depth_of_field": 2, "saturation": 2, "clarity": 1},
    }
    for name, ratings in images.items():
        s = scorer.score(ratings)
        print(f"  {name:<12} 图片质量={s:.2f} -> {scorer.label(s)}")


if __name__ == "__main__":
    main()