# 02_7.1.3_搜索结果页推词

"""
Lecture: 7_第七部分_查询词推荐/7.1_查询词推荐的场景
Content: 02_7.1.3_搜索结果页推词
"""


def show_related(related_map: dict) -> None:
    """打印搜索页上每个查询词的相关推词."""
    for q, rels in related_map.items():
        print(f"  '{q}' 的相关推词: {rels}")


def main() -> None:
    print("搜索结果页推词 Demo")
    related = {
        "口红推荐": ["口红平价", "口红排行榜", "显白口红"],
        "冬季穿搭": ["大衣搭配", "围巾穿搭"],
    }
    show_related(related)
    print("说明: 在结果页展示相关查询词, 方便用户深入探索。")


if __name__ == "__main__":
    main()