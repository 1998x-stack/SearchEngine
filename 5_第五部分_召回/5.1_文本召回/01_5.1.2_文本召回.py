# 01_5.1.2_文本召回

"""
Lecture: 5_第五部分_召回/5.1_文本召回
Content: 01_5.1.2_文本召回
"""

import math
from typing import Dict, List, Set, Tuple


class TextRetrieval:
    """Text recall using an inverted index + BM25-style scoring."""

    def __init__(self) -> None:
        self.postings: Dict[str, Set[int]] = {}
        self.doc_tf: List[Dict[str, int]] = []
        self.doc_len: List[int] = []

    def add(self, doc_id: int, words: List[str]) -> None:
        self.doc_len.append(len(words))
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
            self.postings.setdefault(w, set()).add(doc_id)
        self.doc_tf.append(tf)

    def recall(self, terms: List[str]) -> List[int]:
        """Intersection of posting lists (AND)."""
        if not terms:
            return []
        res = self.postings.get(terms[0], set())
        for t in terms[1:]:
            res &= self.postings.get(t, set())
        return sorted(res)

    def bm25(self, query_terms: List[str], doc_id: int, k1=1.5, b=0.75) -> float:
        tf = self.doc_tf[doc_id]
        dl = self.doc_len[doc_id]
        avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        score = 0.0
        for t in query_terms:
            if t not in tf:
                continue
            idf = math.log(len(self.doc_len) / (len(self.postings.get(t, set())) + 0.5) + 1)
            score += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        return score

    def search(self, query: str) -> List[Tuple[int, float]]:
        terms = query.split()
        cands = self.recall(terms)
        scored = [(d, self.bm25(terms, d)) for d in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def main() -> None:
    print("文本召回 (Text Recall / BM25) Demo")
    ret = TextRetrieval()
    ret.add(0, "python 爬虫 教程")
    ret.add(1, "python 搜索引擎 索引")
    ret.add(2, "深度学习 模型 召回")
    print("候选(AND):", ret.recall(["python", "索引"]))
    print("BM25 排序(查询 'python 索引'):")
    for d, s in ret.search("python 索引"):
        print(f"  文档{d} 得分={s:.3f}")


if __name__ == "__main__":
    main()