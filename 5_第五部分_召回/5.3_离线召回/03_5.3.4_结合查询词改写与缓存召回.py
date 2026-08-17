# 03_5.3.4_结合查询词改写与缓存召回

"""
Lecture: 5_第五部分_召回/5.3_离线召回
Content: 03_5.3.4_结合查询词改写与缓存召回
"""

from typing import Dict, List, Set


def recall_with_rewrite(query: str, synonyms: Dict[str, List[str]],
                        index: Dict[str, Set[str]], cache: Dict[str, List[str]]) -> List[str]:
    """Return union: cached results + rewritten-query recall.

    Args:
        query: input query.
        synonyms: term -> candidate rewrite terms.
        index: term -> doc set (text index).
        cache: query -> cached results (KV index).

    Returns:
        Deduplicated recalled doc list.
    """
    cached = cache.get(query, [])
    rewritten = []
    for w, subs in synonyms.items():
        if w in query:
            for s in subs:
                rewritten.extend(index.get(s, set()))
    seen = set(cached)
    merged = list(cached)
    for d in rewritten:
        if d not in seen:
            seen.add(d)
            merged.append(d)
    return merged


def main() -> None:
    print("结合查询词改写与缓存召回 Demo")
    index = {
        "副作用": {"d1"}, "不良反应": {"d2"},
        "布洛芬": {"d1", "d2"}, "注意事项": {"d3"},
    }
    cache = {"布洛芬": ["d1"]}
    synonyms = {"不良反应": ["副作用", "注意事项"]}
    hits = recall_with_rewrite("布洛芬 不良反应", synonyms, index, cache)
    print(f"查询: 布洛芬 不良反应")
    print(f"召回结果(缓存+改写扩充): {hits}")


if __name__ == "__main__":
    main()