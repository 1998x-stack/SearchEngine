# 00_2.3.1_预训练任务

"""
Lecture: 2_第二部分_机器学习基础/2.3_NLP模型的训练
Content: 00_2.3.1_预训练任务
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Dict, List


class TinyMaskedLM:
    """A tiny masked language model (MLM) built on dictionary embeddings.

    Demonstrates the core pretraining idea: mask tokens and predict them
    from the surrounding context. This is a simplified, lossy demo of BERT's
    MLM, not a real transformer.
    """

    def __init__(self, vocab: List[str], dim: int = 8, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.idx = {w: i for i, w in enumerate(vocab)}
        self.emb = rng.normal(0, 1, (len(vocab), dim))

    def embed(self, token: str) -> np.ndarray:
        tok = token if token in self.idx else "<UNK>"
        i = self.idx.get(tok, 0)
        return self.emb[i]

    def predict_masked(self, sentence: List[str], mask_pos: int) -> np.ndarray:
        """Predict the masked token's probabilities from surrounding context."""
        ctx = np.zeros(self.emb.shape[1])
        n = 0
        for p, tok in enumerate(sentence):
            if p == mask_pos:
                continue
            ctx += self.embed(tok)
            n += 1
        ctx /= max(n, 1)
        scores = self.emb @ ctx          # 语义相似度
        scores = scores - scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()


def main() -> None:
    print("预训练任务: Masked Language Model (MLM) Demo")
    vocab = ["机器学习", "是", "人工智能", "的", "一个", "分支", "<UNK>"]
    lm = TinyMaskedLM(vocab)
    sentence = ["机器学习", "[MASK]", "人工智能", "的", "一个", "分支"]
    mask_pos = 1
    probs = lm.predict_masked(sentence, mask_pos)
    print(f"句子: {sentence}")
    top = np.argsort(-probs)[:3]
    print("预测被遮挡词 [MASK] 的概率排序:")
    for i in top:
        print(f"  {vocab[i]:<6} prob={probs[i]:.3f}")
    print("\n说明: 预训练用海量无标签数据做MLM/NSP等自监督任务，学习语言表示。")


if __name__ == "__main__":
    main()