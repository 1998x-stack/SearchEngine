# 05_4.1.6_知识点小结

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别
Content: 05_4.1.6_知识点小结
"""


def forward_max_match(text: str, vocab, max_len: int = 5) -> list:
    """Dictionary-based forward maximum matching."""
    tokens = []
    i, n = 0, len(text)
    while i < n:
        matched = text[i]
        for size in range(min(max_len, n - i), 1, -1):
            if text[i:i + size] in vocab:
                matched = text[i:i + size]
                break
        tokens.append(matched)
        i += len(matched)
    return tokens


def main() -> None:
    print("=== 分词与NER知识点小结 (Segmentation & NER Summary) Demo ===")
    vocab = {"北京", "欢迎", "深度学习", "人造智能", "人工智能", "学习"}
    text = "北京欢迎深度学习与人工智能"
    print(f"基于词典分词: {'/'.join(forward_max_match(text, vocab))}")

    print("\n主要内容:")
    print("  词典分词(前向最大匹配)、词典构造、深度学习分词(BIO)")
    print("  命名实体识别(PER/ORG/LOC)、指标(Precision/Recall/F1)")


if __name__ == "__main__":
    main()