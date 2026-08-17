# 02_5.1.3_知识点小结

"""
Lecture: 5_第五部分_召回/5.1_文本召回
Content: 02_5.1.3_知识点小结
"""


def main() -> None:
    print("=== 文本召回小结 (Text Recall Summary) Demo ===")
    print("文本召回: 基于倒排索引检索匹配查询词的文档")
    print("  索引: 词项 -> 文档ID倒排表")
    print("  打分: BM25(词频/逆文档频率/长度归一)")
    print("  特点: 快、词面匹配; 对语义匹配不足")
    print("\n见 00_5.1.1(倒排索引) 与 01_5.1.2(BM25文本召回)")


if __name__ == "__main__":
    main()