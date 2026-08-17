# 00_4.1.1_基于词典的分词方法

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别
Content: 00_4.1.1_基于词典的分词方法
"""

from typing import List, Set


def forward_max_match(text: str, vocab: Set[str], max_len: int = 5) -> List[str]:
    """Forward-maximum-matching segmenter using a dictionary.

    Args:
        text: 待切分文本(无空格).
        vocab: 词典(词集合).
        max_len: 最长词长度.

    Returns:
        List of segmented tokens.
    """
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for size in range(min(max_len, n - i), 0, -1):
            cand = text[i:i + size]
            if cand in vocab:
                matched = cand
                break
        if matched is None:
            matched = text[i]          # 未收录的单字
        tokens.append(matched)
        i += len(matched)
    return tokens


def main() -> None:
    print("基于词典的分词 (Dictionary Segmentation) Demo")
    vocab = {"北京", "欢迎", "你", "我们", "搜索引擎", "搜索", "引擎",
             "深度", "学习", "深度学习", "人工智能", "分支"}
    text = "北京欢迎你我们学习深度学习人工智能"
    print(f"文本: {text}")
    print(f"分词结果: {'/'.join(forward_max_match(text, vocab))}")


if __name__ == "__main__":
    main()