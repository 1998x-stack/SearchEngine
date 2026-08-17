# 02_4.2.3_知识点小结

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.2_词权重
Content: 02_4.2.3_知识点小结
"""


def main() -> None:
    print("=== 词权重小结 (Word Weight Summary) Demo ===")
    terms = ["冬季", "卫衣", "推荐"]
    manual_weights = [0.5, 0.9, 0.2]       # 人工标注(0~1)
    print("人工标注词权重 (卫衣最重要):")
    for t, w in zip(terms, manual_weights):
        print(f"  {t:<4} {w:.1f}")

    print("\n主要方法:")
    print("  定义: 词在查询中的重要性")
    print("  标注: 人工打标(0~1)")
    print("  自动: IDF / 注意力机制(见 01_4.2.2)")
    print("  用途: 丢词召回、改写、排序特征")


if __name__ == "__main__":
    main()