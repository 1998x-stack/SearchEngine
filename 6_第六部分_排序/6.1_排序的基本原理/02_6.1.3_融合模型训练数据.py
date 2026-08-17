# 02_6.1.3_融合模型训练数据

"""
Lecture: 6_第六部分_排序/6.1_排序的基本原理
Content: 02_6.1.3_融合模型训练数据
"""

from typing import Dict, List


def batch_features(candidates: List[Dict]) -> List[Dict]:
    """把候选文档的原始特征整理成训练样本."""
    return [dict(item) for item in candidates]


def weight_by_judgment(items: List[Dict], labels: List[float]) -> List[float]:
    """根据人工标注相关性给训练样本权重."""
    return [1.0 if l >= 1 else 0.5 for l in labels]


def main() -> None:
    print("融合模型训练数据 Demo")
    cands = [
        {"relevance": 0.9, "ctr": 0.1, "quality": 0.8},
        {"relevance": 0.3, "ctr": 0.02, "quality": 0.4},
        {"relevance": 0.6, "ctr": 0.05, "quality": 0.6},
    ]
    labels = [1.0, 0.0, 1.0]
    X = batch_features(cands)
    w = weight_by_judgment(X, labels)
    print(f"训练样本数 = {len(X)}")
    print(f"样本权重(高相关更强) = {w}")
    print("说明: 训练数据含相关性/CTR/质量等特征 + 人工标注标签。")


if __name__ == "__main__":
    main()