# 02_1.1.3_搜索引擎链路

"""
Lecture: 1_第一部分_搜索引擎基础/1.1_搜索引擎技术概要
Content: 02_1.1.3_搜索引擎链路
"""

from typing import Dict, List, Set


class SearchPipeline:
    """A tiny end-to-end search pipeline: QP -> retrieval -> ranking.

    Demonstrates the three main stages described in the section using
    a small in-memory corpus and simple scoring.
    """

    def __init__(self) -> None:
        # docs: doc_id -> words (already segmented)
        self.docs: Dict[int, List[str]] = {}
        self.index: Dict[str, Set[int]] = {}

    def add_document(self, doc_id: int, words: List[str]) -> None:
        self.docs[doc_id] = words
        for w in set(words):
            self.index.setdefault(w, set()).add(doc_id)

    def query_processing(self, query: str) -> List[str]:
        """QP stage: segment the query into tokens (word weights equal here)."""
        words = [w for w in query.split() if w]  # 分词
        print(f"  [QP] 分词结果: {words}")
        return words

    def retrieval(self, terms: List[str]) -> Set[int]:
        """Retrieval stage: boolean AND via inverted index."""
        if not terms:
            return set()
        result = self.index.get(terms[0], set())
        for t in terms[1:]:
            result &= self.index.get(t, set())
        print(f"  [召回] 候选文档: {sorted(result)}")
        return result

    @staticmethod
    def _tf(words: List[str]) -> Dict[str, int]:
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        return tf

    def ranking(self, terms: List[str], candidates: Set[int]) -> List[Dict[str, object]]:
        """Two-stage style scoring: term coverage then lexical similarity."""
        ranked = []
        for doc_id in candidates:
            doc_words = self._tf(self.docs[doc_id])
            # 粗排: 命中词覆盖数
            covered = sum(1 for t in terms if t in doc_words)
            # 精排: 词频加权
            dense = sum(doc_words.get(t, 0) for t in terms)
            ranked.append({"doc_id": doc_id, "coverage": covered, "score": dense})
        ranked.sort(key=lambda r: (r["coverage"], r["score"]), reverse=True)
        return ranked

    def search(self, query: str) -> List[Dict[str, object]]:
        """Run the full pipeline for a query."""
        print(f"查询: {query}")
        terms = self.query_processing(query)
        candidates = self.retrieval(terms)
        return self.ranking(terms, candidates)


def main() -> None:
    print("=== 搜索引擎链路 (Search Pipeline) Demo ===")
    engine = SearchPipeline()
    engine.add_document(1, ["python", "教程", "入门"])
    engine.add_document(2, ["python", "爬虫", "教程"])
    engine.add_document(3, ["深度学习", "教程"])

    results = engine.search("python 教程")
    print("  [排序] 最终排序:")
    for r in results:
        print(f"    文档 {r['doc_id']} 覆盖词数={r['coverage']} 得分={r['score']}")


if __name__ == "__main__":
    main()