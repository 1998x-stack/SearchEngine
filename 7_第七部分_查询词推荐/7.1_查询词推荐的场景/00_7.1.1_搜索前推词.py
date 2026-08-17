# 00_7.1.1_搜索前推词

"""
Lecture: 7_第七部分_查询词推荐/7.1_查询词推荐的场景
Content: 00_7.1.1_搜索前推词
"""


def pre_search_suggestions(prefix: str, pool: list, top: int = 3) -> list:
    """搜索前推词: 依据用户画像/热门给出一组查询词."""
    return pool[:top]


def main() -> None:
    print("搜索前推词 (搜索前场景) Demo")
    hot = ["口红测评", "冬季穿搭", "火锅推荐", "新车资讯", "考研资料"]
    user_interest = ["美妆", "穿搭"]
    picks = pre_search_suggestions("", hot, 3)
    print("用户在搜索框输入前即看到一组推荐查询词:")
    for q in picks:
        print(f"  - {q}")


if __name__ == "__main__":
    main()