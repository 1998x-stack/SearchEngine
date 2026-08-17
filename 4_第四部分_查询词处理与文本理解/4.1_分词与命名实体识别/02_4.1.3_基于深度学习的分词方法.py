# 02_4.1.3_基于深度学习的分词方法

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别
Content: 02_4.1.3_基于深度学习的分词方法
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Dict, List


def featurize(text: str, pos: int, vocab: Dict[str, int]) -> np.ndarray:
    """Character features: one-hot of prev / current / next char.

    A character-level sequence model uses these to predict a BIO tag,
    which marks how characters group into segments.
    """
    size = len(vocab)
    vec = np.zeros(size * 3)
    for k, offset in enumerate([-1, 0, 1]):
        idx = pos + offset
        ch = text[idx] if 0 <= idx < len(text) else "<PAD>"
        vec[vocab.get(ch, vocab["<PAD>"]) + k * size] = 1.0
    return vec


def main() -> None:
    print("基于深度学习的分词 (Neural Character Segmentation) Demo")
    vocab = {"深": 0, "度": 1, "学": 2, "习": 3, "智": 4, "能": 5, "<PAD>": 6}
    text = "深度学习人工智能"
    feats = np.array([featurize(text, i, vocab) for i in range(len(text))])
    print(f"文本: {text}  特征形状: {feats.shape}")
    print(f"特征示例(第1字): {feats[0].astype(int)}")
    print("说明: 字符级序列标注(BIO)模型根据上下文预测分词边界。")


if __name__ == "__main__":
    main()