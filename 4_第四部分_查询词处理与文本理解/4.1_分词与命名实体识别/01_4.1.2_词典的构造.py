# 01_4.1.2_词典的构造

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别
Content: 01_4.1.2_词典的构造
"""

from collections import Counter
from typing import List


def build_dict(sentences: List[List[str]], min_freq: int = 2) -> List[str]:
    """Build a segmentation dictionary from tokenized sentences.

    Keeps tokens whose corpus frequency >= min_freq, sorted by frequency.
    """
    freq = Counter()
    for sent in sentences:
        for w in sent:
            freq[w] += 1
    return [w for w, c in freq.most_common() if c >= min_freq]


def main() -> None:
    print("词典的构造 (Dictionary Construction) Demo")
    corpus = [
        ["北京", "欢迎", "你"],
        ["北京", "今天", "欢迎", "游客"],
        ["欢迎", "游客", "今天"],
    ]
    vocab = build_dict(corpus)
    print("从切分语料统计词频构造词典:")
    print(f"  词典词数 = {len(vocab)}")
    print(f"  词典 = {vocab}")


if __name__ == "__main__":
    main()