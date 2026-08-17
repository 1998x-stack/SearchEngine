# 03_4.5.4_基于意图的改写

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.5_查询词改写
Content: 03_4.5.4_基于意图的改写
"""

from typing import Dict, List


def intent_rewrite(query: str) -> List[str]:
    """Rewrite a query to make its intent explicit for better retrieval."""
    rules: Dict[str, List[str]] = {
        "买": ["购买", "商城"],
        "附近": ["附近", "周边", "同城"],
    }
    derived = [query]
    for key, subs in rules.items():
        if key in query:
            for s in subs:
                if s != key:
                    derived.append(query + " " + s)
    return list(dict.fromkeys(derived))


def main() -> None:
    print("基于意图的查询词改写 Demo")
    for q in ["我想买运动鞋", "附近火锅"]:
        print(f" 原查询: {q}")
        for c in intent_rewrite(q):
            print(f"   改写: {c}")


if __name__ == "__main__":
    main()