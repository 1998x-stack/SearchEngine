# 03_3.4.4_排序

"""
Lecture: 3_第三部分_什么决定用户体验？/3.4_地域性
Content: 03_3.4.4_排序
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import List, Tuple


def combined_score(relevance: float, distance_km: float, quality: float,
                   distance_weight: float = 0.4) -> float:
    """Final ranking score combining relevance and (inverse) distance.

    distance_score rises as distance shrinks; distance_weight controls
    how much geography matters for this query.
    """
    distance_score = 1.0 / (1.0 + distance_km)
    return (1 - distance_weight) * relevance + distance_weight * distance_score + 0.1 * quality


def main() -> None:
    print("地域性排序 Demo")
    # 候选文档: (名称, 相关性, 距离km, 内容质量)
    candidates = [
        ("火锅店A", 0.90, 0.5, 0.8),
        ("火锅店B", 0.70, 0.2, 0.7),
        ("火锅店C", 0.85, 3.0, 0.9),
    ]
    print("组合排序(权重=0.4):")
    scored = [(name, combined_score(rel, dist, qual))
              for name, rel, dist, qual in candidates]
    for name, s in sorted(scored, key=lambda x: x[1], reverse=True):
        print(f"  {name:<10} 综合分={s:.3f}")


if __name__ == "__main__":
    main()