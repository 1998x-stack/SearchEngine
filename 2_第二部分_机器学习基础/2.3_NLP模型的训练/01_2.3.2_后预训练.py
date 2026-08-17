# 01_2.3.2_后预训练

"""
Lecture: 2_第二部分_机器学习基础/2.3_NLP模型的训练
Content: 01_2.3.2_后预训练
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class DomainAdapter:
    """Demonstrates post-pretraining: continued training on domain data.

    Post-pretraining (后预训练) continues masked-language-style training on
    unlabeled data from the target domain, nudging embeddings toward that
    domain before fine-tuning.
    """

    def __init__(self, vocab, dim: int = 6, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.idx = {w: i for i, w in enumerate(vocab)}
        self.emb = rng.normal(0, 1, (len(vocab), dim))

    def adapt(self, corpus, steps: int = 3) -> None:
        """Pull embeddings toward within-domain tokens (simplified)."""
        for step in range(steps):
            moves = np.zeros_like(self.emb)
            count = np.zeros(len(self.vocab))
            for doc in corpus:
                ctx = np.mean([self.emb[self.idx[w]] for w in doc], axis=0)
                for w in doc:
                    moves[self.idx[w]] += 0.1 * (ctx - self.emb[self.idx[w]])
                    count[self.idx[w]] += 1
            for i in range(len(self.vocab)):
                if count[i] > 0:
                    self.emb[i] += moves[i]
            print(f"  后预训练 epoch {step+1}: 词向量向领域语料方向微调")


def main() -> None:
    print("后预训练 (Post-Pretraining) Demo")
    vocab = ["搜索", "推荐", "商品", "订单", "模型"]
    adapter = DomainAdapter(vocab)
    domain_corpus = [
        ["搜索", "商品", "推荐"],
        ["商品", "搜索", "订单"],
        ["推荐", "模型", "搜索"],
    ]
    print("在领域无标签语料上继续预训练:")
    adapter.adapt(domain_corpus)
    print("\n说明: 后预训练用海量领域内无标签数据继续学习, 使模型更适配下游任务。")


if __name__ == "__main__":
    main()