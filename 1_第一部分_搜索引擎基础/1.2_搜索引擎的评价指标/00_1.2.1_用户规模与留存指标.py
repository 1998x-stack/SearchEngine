# 00_1.2.1_用户规模与留存指标

"""
Lecture: 1_第一部分_搜索引擎基础/1.2_搜索引擎的评价指标
Content: 00_1.2.1_用户规模与留存指标
"""


class UserMetrics:
    """User-scale and retention metrics for a search product.

    Computes active-user levels, search/feed penetration, and the two
    retention definitions from the section (次 n 留 and 第 n 留).
    """

    def __init__(self, dau: int, wau: int, mau: int,
                 sdau: int, fdau: int) -> None:
        self.dau = dau  # 日活跃用户数
        self.wau = wau  # 周活跃用户数
        self.mau = mau  # 月活跃用户数
        self.sdau = sdau  # 搜索日活
        self.fdau = fdau  # 推荐日活

    def search_penetration(self) -> float:
        """Search penetration = SDAU / DAU."""
        return self.sdau / self.dau if self.dau else 0.0

    def feed_penetration(self) -> float:
        """Feed penetration = FDAU / DAU."""
        return self.fdau / self.dau if self.dau else 0.0

    @staticmethod
    def retention_within_n(returning: int, start: int) -> float:
        """次 n 留 = 今天使用用户中, n 天内再次使用的比例."""
        return returning / start if start else 0.0

    @staticmethod
    def retention_on_day_n(returning_on_day: int, start: int) -> float:
        """第 n 留 = 今天使用用户中, 第 n 天再次使用的比例."""
        return returning_on_day / start if start else 0.0


def main() -> None:
    print("=== 用户规模与留存指标 (User & Retention Metrics) Demo ===")
    m = UserMetrics(dau=10_000_000, wau=25_000_000, mau=40_000_000,
                    sdau=6_000_000, fdau=18_000_000)

    print(f"DAU={m.dau:,}  WAU={m.wau:,}  MAU={m.mau:,}")
    print(f"搜索渗透率 SDAU/DAU = {m.search_penetration()*100:.1f}%")
    print(f"推荐渗透率 FDAU/DAU = {m.feed_penetration()*100:.1f}%")

    start = 100_000  # 今天使用用户
    print("\n留存率示例 (以今天 10 万用户为基数):")
    print(f"  次 7 日留存 = {UserMetrics.retention_within_n(45_000, start)*100:.1f}%")
    print(f"  第 7 日留存 = {UserMetrics.retention_on_day_n(20_000, start)*100:.1f}%")

    print("\n说明: 留存指标存在滞后性, 需要较长周期观测。")


if __name__ == "__main__":
    main()