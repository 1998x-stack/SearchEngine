# 00_7.2.1_SUG_召回

"""
Lecture: 7_第七部分_查询词推荐/7.2_查询词推荐的召回
Content: 00_7.2.1_SUG_召回
"""

from collections import Counter


def sug_recall(prefix: str, query_pool: dict, top: int = 4) -> list:
    """按前缀召回高热度查询词."""
    hits = [(q, f) for q, f in query_pool.items()
            if q.startswith(prefix) or prefix in q]
    hits.sort(key=lambda x: x[1], reverse=True)
    return [q for q, _ in hits[:top]]


def main() -> None:
    print("SUG 召回 (前缀+热度) Demo")
    pool = Counter({
        "口红推荐": 100, "口红平价": 88, "口红排行榜": 60,
        "咖啡推荐": 75, "口红显白": 40,
    })
    for p in ["口红", "咖啡"]:
        print(f"  输入 '{p}' -> {sug_recall(p, pool)}")


if __name__ == "__main__":
    main()