# 01_4.5.2_基于分词的改写

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.5_查询词改写
Content: 01_4.5.2_基于分词的改写
"""

from typing import Dict, List


SYNONYMS: Dict[str, List[str]] = {
    "不良反应": ["副作用", "副反应"],
    "布洛芬": ["神药布洛芬"],
    "减重": ["减肥", "减脂"],
    "电脑": ["计算机"],
}


def rewrite_by_segmentation(query: str) -> List[str]:
    """Segment the query and generate synonyms substitutions."""
    originals = []
    for w, subs in SYNONYMS.items():
        if w in query:
            for s in subs:
                originals.extend([query.replace(w, s), query])  # 保留原词
    return list(dict.fromkeys(originals))


def main() -> None:
    print("基于分词的查询词改写 Demo")
    q = "布洛芬 不良反应"
    candidates = rewrite_by_segmentation(q)
    print(f"原查询: {q}")
    print("改写候选:")
    for c in candidates:
        print(f"  {c}")


if __name__ == "__main__":
    main()