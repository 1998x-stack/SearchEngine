# 01_7.2.2_用查询词召回查询词（Q2Q）

"""
Lecture: 7_第七部分_查询词推荐/7.2_查询词推荐的召回
Content: 01_7.2.2_用查询词召回查询词（Q2Q）
"""

from typing import Dict, List, Tuple


def q2q_recall(logs: List[Tuple[str, str]], query: str, top: int = 3) -> List[str]:
    """Query-to-query recall from co-click logs.

    Queries that clicked the same documents as `query` are treated as
    related, ranked by co-occurrence count.

    Args:
        logs: list of (query, clicked_doc).
        query: the source query.
        top: how many related queries to return.

    Returns:
        Related queries, most co-occurring first.
    """
    docs_of_q = {d for q, d in logs if q == query}
    counts: Dict[str, int] = {}
    for q, d in logs:
        if q != query and d in docs_of_q:
            counts[q] = counts.get(q, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:top]


def main() -> None:
    print("Q2Q 召回 (查询词->查询词) Demo")
    logs = [("口红", "d1"), ("口红", "d2"), ("平价口红", "d1"),
            ("口红推荐", "d2"), ("口红", "d3"), ("显白口红", "d1")]
    print(f"查询 '口红' 的相关查询词: {q2q_recall(logs, '口红')}")


if __name__ == "__main__":
    main()