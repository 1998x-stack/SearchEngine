# 04_7.1.5_评价指标总结

"""
Lecture: 7_第七部分_查询词推荐/7.1_查询词推荐的场景
Content: 04_7.1.5_评价指标总结
"""


def ctr(clicks: int, exposures: int) -> float:
    return clicks / exposures if exposures else 0.0


def main() -> None:
    print("查询词推荐评价指标总结 Demo")
    print(f"推词点击率: {ctr(30, 200)*100:.1f}%  (推词被点击/展示)")
    print(f"覆盖率: 推荐的查询词覆盖了多少用户查询")
    print(f"拔动率/点击转化: 用户是否点击推荐词并继续搜索")
    print(f"多样性: 推荐词之间的主题差异")
    print("\n场景: 搜索前/查询建议/结果页/文档内 各有侧重指标。")


if __name__ == "__main__":
    main()