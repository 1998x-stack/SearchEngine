# 01_6.1.2_融合规则_vs_融合模型

"""
Lecture: 6_第六部分_排序/6.1_排序的基本原理
Content: 01_6.1.2_融合规则_vs_融合模型
"""

from typing import Dict


def rule_fusion(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """规则融合: 固定的加权求和."""
    return sum(weights.get(k, 0.0) * v for k, v in features.items())


def learned_fusion(features: Dict[str, float], w: Dict[str, float],
                   bias: float = 0.0) -> float:
    """融合模型(线性): 由数据学到的权重."""
    return sum(w.get(k, 0.0) * v for k, v in features.items()) + bias


def main() -> None:
    print("融合规则 vs 融合模型 Demo")
    feats = {"relevance": 0.8, "ctr": 0.1, "quality": 0.7}
    weights = {"relevance": 0.5, "ctr": 0.3, "quality": 0.2}
    w = {"relevance": 0.6, "ctr": 0.25, "quality": 0.15}

    print(f"规则融合得分 = {rule_fusion(feats, weights):.3f}")
    print(f"模型融合得分 = {learned_fusion(feats, w, bias=0.05):.3f}")
    print("规则: 由人工设定权重; 模型: 权重由数据学习, 可随样本调整。")


if __name__ == "__main__":
    main()