# 00_3.1.1_相关性的定义与分档

"""
Lecture: 3_第三部分_什么决定用户体验？/3.1_相关性
Content: 00_3.1.1_相关性的定义与分档
"""


def tier_for_ratio(ratio: float) -> str:
    """Map a '需求满足比例' to a relevance tier.

    Ratio is the fraction of document content satisfying the query.

    Returns:
        One of 高相关 / 中相关 / 低相关 / 无相关.
    """
    if ratio >= 0.5:
        return "高相关"
    if ratio >= 0.2:
        return "中相关"
    if ratio > 0.0:
        return "低相关"
    return "无相关"


def main() -> None:
    print("相关性定义与分档 (Relevance Definition & Tiers) Demo")
    print("相关性基本原则: 需求匹配, 独立于内容质量/时效性/地域性。")
    print("\n根据文档满足查询需求的比例分档:")
    for ratio in [0.9, 0.55, 0.3, 0.2, 0.1, 0.0]:
        print(f"  满足比例 {ratio:.2f} -> { tier_for_ratio(ratio)}")

    print("\n对应实例:")
    print("  高相关: 查询'泰坦尼克号', 文档50%以上谈这部电影")
    print("  中相关: 查询'小米手机测评', 文档部分篇幅介绍小米")
    print("  低相关: 查询'初二物理考点', 文档为'初三中考物理考点'")
    print("  无相关: 查询'面霜', 文档为'口红测评'")


if __name__ == "__main__":
    main()