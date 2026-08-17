# 03_6.2.4_知识点小结

"""
Lecture: 6_第六部分_排序/6.2_训练融合模型的方法
Content: 03_6.2.4_知识点小结
"""


def main() -> None:
    print("=== 排序训练方法小结 (Ranking Training Summary) Demo ===")
    print("三种训练融合模型(排序)的方法:")
    print("  pointwise: 独立预测每个文档的相关性得分")
    print("  pairwise : 优化文档对的相对顺序(正逆序)")
    print("  listwise : 优化整个列表的排序(如NDCG/softmax)")
    print("应用: 搜索引擎排序、推荐列表排序")


if __name__ == "__main__":
    main()