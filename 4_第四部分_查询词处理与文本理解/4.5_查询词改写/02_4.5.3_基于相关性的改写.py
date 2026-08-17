# 02_4.5.3_基于相关性的改写

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.5_查询词改写
Content: 02_4.5.3_基于相关性的改写
"""

from typing import Dict, List


QUERY_HISTORY: Dict[str, List[str]] = {
    "布洛芬副作用": ["布洛芬不良反应", "布洛芬禁忌", "布洛芬说明书"],
    "口红推荐": ["平价口红", "口红排行榜", "哑光口红"],
}


def relevance_based_rewrite(query: str) -> List[str]:
    """Return historical similar queries (from logs) as rewrite candidates."""
    return QUERY_HISTORY.get(query, [query])


def main() -> None:
    print("基于相关性的查询词改写 Demo")
    for q in ["布洛芬副作用", "口红推荐"]:
        cands = relevance_based_rewrite(q)
        print(f" 原查询: {q}")
        print(f"  改写候选(历史相似查询): {cands}")


if __name__ == "__main__":
    main()