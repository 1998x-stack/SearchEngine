# 01_7.1.2_查询建议

"""
Lecture: 7_第七部分_查询词推荐/7.1_查询词推荐的场景
Content: 01_7.1.2_查询建议
"""


def suggest(prefix: str, pool: list, top: int = 4) -> list:
    """查询建议(SUG): 按输入前缀给出补全/相关查询词."""
    return [q for q in pool if q.startswith(prefix) or prefix in q][:top]


def main() -> None:
    print("查询建议 (SUG 前缀补全) Demo")
    pool = ["口红推荐", "口红平价", "口红排行榜", "咖啡推荐", "咖啡好喝"]
    for p in ["口红", "咖啡"]:
        print(f"输入 '{p}':")
        for s in suggest(p, pool):
            print(f"  - {s}")


if __name__ == "__main__":
    main()