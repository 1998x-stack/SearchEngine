# 00_3.4.1_POI_识别

"""
Lecture: 3_第三部分_什么决定用户体验？/3.4_地域性
Content: 00_3.4.1_POI_识别
"""

from collections import Counter
from typing import Dict, List


def fuse_poi(ratings: Dict[str, List[str]]) -> str:
    """Fuse multi-source POI votes and pick the most confident label.

    Args:
        ratings: source -> list of POI candidates for a note.

    Returns:
        The POI with the highest vote count.
    """
    votes: Counter = Counter()
    for source, cands in ratings.items():
        for c in cands:
            votes[c] += 1
    return votes.most_common(1)[0][0]


def main() -> None:
    print("POI 识别 (多源融合) Demo")
    # 一篇小红书笔记的多种来源 POI 标签
    note = {
        "user_manual": ["外滩", "外滩", "外滩"],
        "gps": ["外滩"],
        "nlp": ["外滩", "陆家嘴"],
        "cv": ["东方明珠"],
    }
    print("各数据源 POI 标签:")
    for src, cands in note.items():
        print(f"  {src:<12}: {cands}")
    print(f"融合后最置信 POI = {fuse_poi(note)}")


if __name__ == "__main__":
    main()