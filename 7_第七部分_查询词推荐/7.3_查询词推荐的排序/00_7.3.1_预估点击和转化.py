# 00_7.3.1_预估点击和转化

"""
Lecture: 7_第七部分_查询词推荐/7.3_查询词推荐的排序
Content: 00_7.3.1_预估点击和转化
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class ClickConvertEstimator:
    """Predict click & conversion for a recommended query via linear logit."""

    def __init__(self, n_features: int) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0

    def set_weights(self, w: List[float], b: float) -> None:
        self.w = np.asarray(w)
        self.b = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)


def main() -> None:
    print("预估推词点击和转化 Demo")
    # 特征: 相关性, 热度, 历史点击
    est = ClickConvertEstimator(3)
    est.set_weights([1.0, 0.5, 0.3], -1.0)
    cands = [
        {"name": "口红平价", "feat": [0.9, 0.8, 0.7]},
        {"name": "口红排行榜", "feat": [0.7, 0.9, 0.5]},
        {"name": "混乱无关词", "feat": [0.1, 0.2, 0.1]},
    ]
    print("推荐查询词的点击率预估:")
    for c in cands:
        p = est.predict(np.array([c["feat"]]))[0]
        print(f"  {c['name']:<10} 预估点击率={p:.3f}")


if __name__ == "__main__":
    main()