# 02_1.2.2_中间过程指标

"""
Lecture: 1_第一部分_搜索引擎基础/1.2_搜索引擎的评价指标
Content: 02_1.2.2_中间过程指标
"""


class IntermediateMetrics:
    """Intermediate UX metrics that respond faster than retention.

    Computes CTR, 有点比, 首屏有点比, 换词率 and related counts.
    """

    def __init__(self,
                 impressions: int, clicks: int,
                 searches_with_click: int, total_searches: int,
                 first_screen_clicks: int,
                 query_changes: int) -> None:
        self.impressions = impressions
        self.clicks = clicks
        self.searches_with_click = searches_with_click
        self.total_searches = total_searches
        self.first_screen_clicks = first_screen_clicks
        self.query_changes = query_changes

    def ctr(self) -> float:
        """Click-through rate = clicks / impressions."""
        return self.clicks / self.impressions if self.impressions else 0.0

    def hit_rate(self) -> float:
        """有点比 = 有点击的搜索次数 / 总搜索次数."""
        return self.searches_with_click / self.total_searches if self.total_searches else 0.0

    def first_screen_hit_rate(self) -> float:
        """首屏有点比 = 首屏点击 / 总点击."""
        return self.first_screen_clicks / self.clicks if self.clicks else 0.0

    def query_change_rate(self) -> float:
        """换词率 = 主动换词次数 / 总搜索次数."""
        return self.query_changes / self.total_searches if self.total_searches else 0.0


def main() -> None:
    print("=== 中间过程指标 (Intermediate Metrics) Demo ===")
    m = IntermediateMetrics(
        impressions=5_000_000_000, clicks=500_000_000,
        searches_with_click=180_000_000, total_searches=300_000_000,
        first_screen_clicks=400_000_000, query_changes=60_000_000,
    )
    print(f"点击率 (CTR) = {m.ctr()*100:.2f}%")
    print(f"有点比       = {m.hit_rate()*100:.2f}%")
    print(f"首屏有点比   = {m.first_screen_hit_rate()*100:.2f}%")
    print(f"换词率       = {m.query_change_rate()*100:.2f}%")


if __name__ == "__main__":
    main()