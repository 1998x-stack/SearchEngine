# 00_5.1.1_倒排索引

"""
Lecture: 5_第五部分_召回/5.1_文本召回
Content: 00_5.1.1_倒排索引
"""

from typing import Dict, List, Set


class InvertedIndex:
    """A term -> document-id mapping for boolean retrieval."""

    def __init__(self) -> None:
        self.postings: Dict[str, Set[int]] = {}

    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        """Index one document's tokens (deduplicated per doc).

        Args:
            doc_id: unique document identifier.
            tokens: tokenized document content.
        """
        for term in set(tokens):
            self.postings.setdefault(term, set()).add(doc_id)

    def query(self, terms: List[str]) -> Set[int]:
        """Boolean AND: documents containing every input term."""
        if not terms:
            return set()
        result = self.postings.get(terms[0], set())
        for term in terms[1:]:
            result &= self.postings.get(term, set())
        return result

    def df(self, term: str) -> int:
        """Document frequency of a term."""
        return len(self.postings.get(term, set()))

    def __len__(self) -> int:
        return len(self.postings)


def main() -> None:
    print("倒排索引 (Inverted Index) Demo")
    idx = InvertedIndex()
    corpus = {
        1: ["搜索引擎", "倒排", "索引"],
        2: ["搜索引擎", "召回", "模型"],
        3: ["排序", "模型", "召回"],
    }
    for doc_id, tokens in corpus.items():
        idx.add_document(doc_id, tokens)
    print(f"词汇表词项数: {len(idx)}")
    for term in ["搜索引擎", "模型", "召回"]:
        print(f"  '{term}' 的文档频率: {idx.df(term)}")
    print(f"查询 ['搜索引擎', '模型'] 命中文档: {sorted(idx.query(['搜索引擎', '模型']))}")


if __name__ == "__main__":
    main()