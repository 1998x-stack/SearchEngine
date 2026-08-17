# 02_5.3.3_反向召回

"""
Lecture: 5_第五部分_召回/5.3_离线召回
Content: 02_5.3.3_反向召回
"""

from typing import Dict, List, Tuple


def reverse_recall(sessions: List[Tuple[str, str]],
                   target_doc: str) -> List[str]:
    """从行为日志反向构建 文档->查询 关联, 召回 query 相关文档.

    Args:
        sessions: log of (query, clicked_doc).
        target_doc: the doc we want to find related queries for.

    Returns:
        Queries with most co-occurring clicks on target_doc, top first.
    """
    query_ctr: Dict[str, int] = {}
    for q, d in sessions:
        if d == target_doc:
            query_ctr[q] = query_ctr.get(q, 0) + 1
    return sorted(query_ctr, key=query_ctr.get, reverse=True)


def main() -> None:
    print("反向召回 (Reverse/Inverse Recall) Demo")
    sessions = [("口红", "docX"), ("口红", "docX"), ("平价口红", "docX"),
                ("口红", "docY"), ("试色", "docX")]
    related = reverse_recall(sessions, "docX")
    print(f"文档 docX 相关的反查询词: {related}")
    print("作用: 用文档召回相关查询, 提升该文档的召回覆盖。")


if __name__ == "__main__":
    main()