# 03_7.2.4_各推词场景的召回方法

"""
Lecture: 7_第七部分_查询词推荐/7.2_查询词推荐的召回
Content: 03_7.2.4_各推词场景的召回方法
"""


def scenario_method(query: str, scenario: str) -> str:
    """为指定场景选择召回方法."""
    if scenario == "搜索前推词":
        return "热门/个性化(用户画像)召回"
    if scenario == "查询建议(SUG)":
        return "前缀 + 热度召回"
    if scenario == "搜索结果页推词":
        return "Q2Q(共同点击)召回"
    if scenario == "文档内推词":
        return "D2Q(文档->查询)召回"
    return "综合召回"


def main() -> None:
    print("各推词场景的召回方法 Demo")
    for s in ["搜索前推词", "查询建议(SUG)", "搜索结果页推词", "文档内推词"]:
        print(f"  {s:<18} -> {scenario_method('', s)}")


if __name__ == "__main__":
    main()