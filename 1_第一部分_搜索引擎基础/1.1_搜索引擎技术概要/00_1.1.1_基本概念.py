# 00_1.1.1_基本概念

"""
Lecture: 1_第一部分_搜索引擎基础/1.1_搜索引擎技术概要
Content: 00_1.1.1_基本概念
"""

from typing import Dict, List


class SearchConcepts:
    """Basic search-engine glossary + simple CTR demo.

    Holds the key terms introduced in this section and provides a
    tiny impression/click accounting helper.
    """

    GLOSSARY: Dict[str, str] = {
        "Query": "用户在搜索框中输入的文本，表达其搜索意图。",
        "SUG": "搜索框下方的推荐查询词，帮助用户快速选择。",
        "Impression": "搜索结果被展示给用户的次数（曝光）。",
        "CTR": "点击率，点击次数 / 曝光次数。",
        "Engagement": "点击后的进一步交互，如点赞、收藏、评论。",
        "UGC": "用户生成内容（小红书、B站等非结构化内容）。",
    }

    def __init__(self) -> None:
        self.impressions = 0
        self.clicks = 0

    def record(self, impressions: int, clicks: int) -> None:
        """Accumulate impression/click counts for CTR."""
        self.impressions += impressions
        self.clicks += clicks

    def ctr(self) -> float:
        """Click-through rate = clicks / impressions."""
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions

    def show_glossary(self) -> None:
        """Print the key basic concepts."""
        print("核心概念表 (Key Concepts):")
        for term, meaning in self.GLOSSARY.items():
            print(f"  {term:<10}: {meaning}")


def main() -> None:
    print("=== 搜索引擎基本概念 (Basic Concepts) Demo ===")
    demo = SearchConcepts()
    demo.show_glossary()

    print("\n点击率(CTR)示例:")
    demo.record(impressions=100, clicks=20)
    demo.record(impressions=150, clicks=45)
    print(f"  累计曝光={demo.impressions}, 累计点击={demo.clicks}")
    print(f"  CTR = 点击/曝光 = {demo.clicks}/{demo.impressions} = {demo.ctr()*100:.1f}%")

    print("\n搜索引擎链路简图:")
    print("  用户输入查询词 -> 查询词处理(QP) -> 召回(Retrieval) -> 排序(Ranking) -> 展示")


if __name__ == "__main__":
    main()