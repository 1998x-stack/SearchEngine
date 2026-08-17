# 02_5.2.3_线上推理

"""
Lecture: 5_第五部分_召回/5.2_向量召回
Content: 02_5.2.3_线上推理
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Dict, List, Tuple


class VectorIndex:
    """Simple (non-ANN) online vector index: embedding + linear scan."""

    def __init__(self) -> None:
        self.doc_id: Dict[int, np.ndarray] = {}

    def build(self, items: List[Tuple[int, np.ndarray]]) -> None:
        for did, v in items:
            self.doc_id[did] = v / (np.linalg.norm(v) + 1e-12)

    def query(self, q: np.ndarray, k: int = 3) -> List[int]:
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = [(d, float(q @ v)) for d, v in self.doc_id.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in sims[:k]]


def main() -> None:
    print("线上向量推理 (Online Vector Inference) Demo")
    rng = np.random.default_rng(2)
    idx = VectorIndex()
    doc_vecs = [(i, rng.normal(0, 1, 8)) for i in range(5)]
    idx.build(doc_vecs)
    q = rng.normal(0, 1, 8)
    print("线上对查询向量做近邻检索:")
    print(f"  召回的 top-k 文档ID = {idx.query(q, k=3)}")

    print("\n说明: 线上用近似最近邻(ANN)索引加速, 这里简化为线性扫描。")


if __name__ == "__main__":
    main()