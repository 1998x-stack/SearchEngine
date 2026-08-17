# 01_3.1.2_文本匹配分数

"""
Lecture: 3_第三部分_什么决定用户体验？/3.1_相关性
Content: 01_3.1.2_文本匹配分数
"""

import numpy as np
from typing import Dict, List, Tuple


class TextMatching:
    """Classic lexical relevance scores: TF-IDF, BM25 and term proximity.

    All three metrics share the same toy corpus and query interface, which
    makes it easy to compare how each one ranks the documents.
    """

    def __init__(self, corpus: List[str]) -> None:
        """Store the corpus and precompute shared statistics.

        Args:
            corpus: list of documents, each tokenized by whitespace.
        """
        self.corpus = corpus
        self.num_docs = len(corpus)

        # word -> list of per-document term frequencies (one entry per doc)
        self.tf: List[Dict[str, int]] = []
        # word -> number of documents containing it
        self.df: Dict[str, int] = {}
        # word -> inverse document frequency
        self.idf: Dict[str, float] = {}
        self.avg_doc_length = 0.0
        self._preprocess()

    def _preprocess(self) -> None:
        """Build TF, DF, IDF and the average document length."""
        doc_lengths: List[int] = []
        for doc in self.corpus:
            words = doc.split()
            doc_lengths.append(len(words))
            counts: Dict[str, int] = {}
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            self.tf.append(counts)
            for word in counts:
                self.df[word] = self.df.get(word, 0) + 1

        self.avg_doc_length = sum(doc_lengths) / self.num_docs
        for word, freq in self.df.items():
            # smooth, additive-one IDF so every vocabulary word stays finite
            self.idf[word] = np.log((self.num_docs - freq + 0.5) /
                                    (freq + 0.5) + 1)

    def tfidf(self, query: List[str], doc_index: int) -> float:
        """TF-IDF score = sum over query terms of tf * idf.

        Args:
            query: query terms.
            doc_index: index into the corpus.

        Returns:
            the TF-IDF relevance score.
        """
        tf_doc = self.tf[doc_index]
        total = 0.0
        for word in query:
            total += tf_doc.get(word, 0) * self.idf.get(word, 0.0)
        return total

    def bm25(self, query: List[str], doc_index: int,
             k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 score with term-frequency saturation and length norm.

        Args:
            query: query terms.
            doc_index: index into the corpus.
            k1: term-frequency saturation parameter.
            b: length-normalization parameter.

        Returns:
            the BM25 relevance score.
        """
        tf_doc = self.tf[doc_index]
        doc_len = sum(tf_doc.values())
        norm = 1 - b + b * (doc_len / self.avg_doc_length)
        total = 0.0
        for word in query:
            tf = tf_doc.get(word, 0)
            idf = self.idf.get(word, 0.0)
            total += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        return total

    def term_proximity(self, query: List[str], doc_index: int) -> float:
        """Term-proximity score: reward query terms appearing close together.

        Sums 1 / distance^2 over every ordered pair of distinct query terms
        that both occur in the document.

        Args:
            query: query terms.
            doc_index: index into the corpus.

        Returns:
            the term-proximity score.
        """
        word_positions: Dict[str, List[int]] = {}
        for pos, word in enumerate(self.corpus[doc_index].split()):
            if word in query:
                word_positions.setdefault(word, []).append(pos)

        total = 0.0
        for i, w1 in enumerate(query):
            if w1 not in word_positions:
                continue
            for w2 in query[i + 1:]:
                if w2 not in word_positions:
                    continue
                for p1 in word_positions[w1]:
                    for p2 in word_positions[w2]:
                        dist = abs(p1 - p2)
                        if dist > 0:
                            total += 1.0 / (dist ** 2)
        return total

    def score_all(self, query: List[str]) -> Tuple[Dict[str, float],
                                                    Dict[str, float],
                                                    Dict[str, float]]:
        """Score every document with all three metrics.

        Args:
            query: query terms.

        Returns:
            three dicts mapping doc id -> score, one per metric.
        """
        tfidf_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}
        prox_scores: Dict[str, float] = {}
        for i, doc in enumerate(self.corpus):
            label = f"文档{i}"
            tfidf_scores[label] = self.tfidf(query, i)
            bm25_scores[label] = self.bm25(query, i)
            prox_scores[label] = self.term_proximity(query, i)
        return tfidf_scores, bm25_scores, prox_scores


def main() -> None:
    print("=== 文本匹配分数 (TF-IDF / BM25 / 词距) Demo ===")
    # 使用空格分隔已分词的文本；这样才能被 doc.split() 正确切分为词
    corpus: List[str] = [
        "机器学习 是 人工智能 的 一个 分支",
        "深度学习 是 机器学习 的 一个 重要 领域",
        "自然 语言 处理 是 一个 重要 应用",
    ]
    query: List[str] = ["机器学习", "人工智能"]

    matcher = TextMatching(corpus)
    tfidf_scores, bm25_scores, prox_scores = matcher.score_all(query)

    print(f"查询词: {query}")
    for i, doc in enumerate(corpus):
        label = f"文档{i}"
        print(f"\n{label}: {doc}")
        print(f"  TF-IDF         = {tfidf_scores[label]:8.4f}")
        print(f"  BM25           = {bm25_scores[label]:8.4f}")
        print(f"  词距分数        = {prox_scores[label]:8.4f}")

    best_tfidf = max(tfidf_scores, key=tfidf_scores.get)
    best_bm25 = max(bm25_scores, key=bm25_scores.get)
    best_prox = max(prox_scores, key=prox_scores.get)
    print(f"\n各方法最相关的文档:")
    print(f"  TF-IDF -> {best_tfidf}, BM25 -> {best_bm25}, 词距 -> {best_prox}")


if __name__ == "__main__":
    main()