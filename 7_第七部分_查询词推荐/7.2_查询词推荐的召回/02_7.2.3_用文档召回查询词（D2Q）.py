# 02_7.2.3_用文档召回查询词（D2Q）

"""
Lecture: 7_第七部分_查询词推荐/7.2_查询词推荐的召回
Content: 02_7.2.3_用文档召回查询词（D2Q）
"""

from typing import Dict, List, Tuple


def d2q_recall(logs: List[Tuple[str, str]], doc: str, top: int = 3) -> List[str]:
    """Document-to-query recall: queries that led to clicks on `doc`.

    Args:
        logs: list of (query, clicked_doc).
        doc: the target document.
        top: number of queries to return.

    Returns:
        Queries most associated with the document.
    """
    counts: Dict[str, int] = {}
    for q, d in logs:
        if d == doc:
            counts[q] = counts.get(q, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:top]


def main() -> None:
    print("D2Q 召回 (文档->查询词) Demo")
    logs = [("口红", "d1"), ("平价口红", "d1"), ("口红推荐", "d1"),
            ("口红", "d2"), ("显白口红", "d1")]
    print(f"阅读文档 d1 时推荐查询词: {d2q_recall(logs, 'd1')}")


if __name__ == "__main__":
    main()