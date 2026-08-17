# 03_1.1.4_知识点小结

"""
Lecture: 1_第一部分_搜索引擎基础/1.1_搜索引擎技术概要
Content: 03_1.1.4_知识点小结
"""

import math
from typing import Dict, List


class ChapterSummary:
    """Numeric recap of the key stages in the search pipeline.

    Demonstrates the two most fundamental quantities from this
    chapter: word segmentation and word weighting (IDF).
    """

    def __init__(self, corpus: List[List[str]]) -> None:
        self.corpus = corpus
        self.df = self._document_frequencies()

    def _document_frequencies(self) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for doc in self.corpus:
            for w in set(doc):
                df[w] = df.get(w, 0) + 1
        return df

    def idf(self, term: str) -> float:
        """Inverse document frequency for a term."""
        n = len(self.corpus)
        return math.log((n + 1) / (self.df.get(term, 0) + 1)) + 1

    def weight(self, query: List[str]) -> List[float]:
        """Weight each query token by IDF (越罕见越重要)."""
        return [self.idf(w) for w in query]


def main() -> None:
    print("=== 搜索引擎知识点小结 (Chapter Summary) Demo ===")
    corpus = [
        ["搜索引擎", "查询词", "召回"],
        ["搜索引擎", "排序", "模型"],
        ["深度学习", "排序", "模型"],
    ]
    summary = ChapterSummary(corpus)

    print("分词的作用: 把查询词切分成词, 供文本召回使用。")
    query = ["搜索引擎", "排序"]
    print(f"  查询词: {query}")
    print(f"  各词权重(IDF): {[round(w, 3) for w in summary.weight(query)]}")

    print("\n搜索引擎链路核心环节:")
    print("  查询词处理(QP) -> 召回(Retrieval) -> 排序(Ranking)")


if __name__ == "__main__":
    main()