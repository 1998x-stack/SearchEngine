# 00_6.1.1_融合模型的特征

"""
Lecture: 6_第六部分_排序/6.1_排序的基本原理
Content: 00_6.1.1_融合模型的特征
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List


class FusionFeature:
    """Feature vector for one (query, doc) candidate in the fusion model."""

    def __init__(self) -> None:
        self.fields = ["relevance", "ctr", "quality", "recency", "geo"]

    def build(self, raw: dict) -> List[float]:
        """Build a normalized feature list from raw dict."""
        return [raw.get(k, 0.0) for k in self.fields]


def main() -> None:
    print("融合模型的特征 (Fusion Model Features) Demo")
    ff = FusionFeature()
    one = {"relevance": 0.9, "ctr": 0.1, "quality": 0.8, "recency": 0.5, "geo": 0.7}
    two = {"relevance": 0.3, "ctr": 0.02, "quality": 0.4, "recency": 0.2, "geo": 0.1}
    print(f"特征字段: {ff.fields}")
    print(f"文档1特征: {ff.build(one)}")
    print(f"文档2特征: {ff.build(two)}")


if __name__ == "__main__":
    main()