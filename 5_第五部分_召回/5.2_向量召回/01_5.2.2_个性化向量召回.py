# 01_5.2.2_个性化向量召回

"""
Lecture: 5_第五部分_召回/5.2_向量召回
Content: 01_5.2.2_个性化向量召回
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Dict, List


def personal_embed(user_vec: np.ndarray, query_vec: np.ndarray,
                   alpha: float = 0.3) -> np.ndarray:
    """Personalized query vector = query + alpha * user preference."""
    return query_vec + alpha * user_vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    print("个性化向量召回 Demo")
    rng = np.random.default_rng(1)
    # 用户偏好(喜欢编程) + 查询向量
    user = rng.normal(0, 1, 8)
    query = rng.normal(0, 1, 8)
    doc_vecs = {
        "编程教程": rng.normal(0, 1, 8),
        "美食探店": rng.normal(0, 1, 8),
    }
    # 让编程教程与用户偏好更相似
    doc_vecs["编程教程"] = user * 0.9 + rng.normal(0, 0.2, 8)

    base = {k: cosine(personal_embed(user, query, 0.0), v) for k, v in doc_vecs.items()}
    pers = {k: cosine(personal_embed(user, query, 0.4), v) for k, v in doc_vecs.items()}
    print("个性化权重:")
    for k in doc_vecs:
        print(f"  {k:<6} 无个性化={base[k]:.3f}  个性化={pers[k]:.3f}")


if __name__ == "__main__":
    main()