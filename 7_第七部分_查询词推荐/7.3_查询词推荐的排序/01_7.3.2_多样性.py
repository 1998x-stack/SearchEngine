# 01_7.3.2_多样性

"""
Lecture: 7_第七部分_查询词推荐/7.3_查询词推荐的排序
Content: 01_7.3.2_多样性
"""

import numpy as np
from typing import List, Tuple


def greedy_diverse(query_vec: np.ndarray, cands: List[Tuple[str, np.ndarray]],
                   weights: List[float], k: int = 3) -> List[str]:
    """Greedy maximum-marginal-relevance style diversification.

    At each step pick the candidate with the highest relevance that stays
    far enough from already-selected ones (reduces redundancy).

    Args:
        query_vec: query embedding.
        cands: (name, embedding) candidates.
        weights: per-candidate relevance scores.
        picked: number to pick.

    Returns:
        Selected candidate names.
    """
    picked_indices: List[int] = []
    picked = []
    # 按相关性初排
    idx = sorted(range(len(cands)), key=lambda i: weights[i], reverse=True)
    for i in idx:
        v = cands[i][1]
        sim_to_picked = max((np.dot(v, cands[j][1]) for j in picked_indices), default=-1.0)
        if sim_to_picked < 0.6 or not picked_indices:   # 与已选差异足够大才选
            picked.append(cands[i][0])
            picked_indices.append(i)
        if len(picked) >= k:
            break
    return picked


def main() -> None:
    print("推词多样性 (Diversity) Demo")
    rng = np.random.default_rng(0)
    q = rng.normal(0, 1, 6)
    cands = [
        ("口红平价", rng.normal(0, 1, 6)),
        ("口红推荐", rng.normal(0, 1, 6)),
        ("显白口红", rng.normal(0, 1, 6)),
        ("睫毛增长液", rng.normal(0, 1, 6)),
        ("护肤教程", rng.normal(0, 1, 6)),
    ]
    weights = [0.9, 0.88, 0.7, 0.5, 0.4]
    picks = greedy_diverse(q, cands, weights, 3)
    print(f"选择(兼顾相关与多样): {picks}")


if __name__ == "__main__":
    main()