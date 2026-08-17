# 00_3.5.1_特征

"""
Lecture: 3_第三部分_什么决定用户体验？/3.5_个性化与点击率预估
Content: 00_3.5.1_特征
"""

import numpy as np
from typing import Dict


class FeatureVector:
    """Turn raw features into a model-ready vector.

    Numeric features are min-max scaled; categorical features are
    one-hot encoded via a fixed vocabulary.
    """

    def __init__(self) -> None:
        # 数字特征名 (0~1 语义已被 min-max 约束)
        self.numeric = ["ctr", "dwell"]
        # 离散特征 -> 允许的取值
        self.categories = {
            "is_weekend": ["yes", "no"],
            "query_cat": ["美食", "娱乐", "科技"],
        }

    def encode(self, raw: Dict[str, float]) -> np.ndarray:
        """Encode one raw sample into a feature vector."""
        parts = []
        for key in self.numeric:
            parts.append(np.clip(raw.get(key, 0.0), 0.0, 1.0))    # 数字特征归一化
        for key, values in self.categories.items():
            hot = np.zeros(len(values))
            val = raw.get(key, values[0])
            if val in values:
                hot[values.index(val)] = 1.0                       # 离散特征 one-hot
            parts.append(hot)
        return np.concatenate([np.atleast_1d(p) for p in parts])


def main() -> None:
    print("个性化特征工程 (Feature Engineering) Demo")
    fv = FeatureVector()
    sample = {"ctr": 0.12, "dwell": 0.7, "is_weekend": "yes", "query_cat": "娱乐"}
    vec = fv.encode(sample)
    print(f"原始特征: {sample}")
    print(f"特征向量 = {np.round(vec, 3)}")
    print("说明: 查询/用户/文档/场景/统计特征 归一化+嵌入后 送入点击率模型。")


if __name__ == "__main__":
    main()