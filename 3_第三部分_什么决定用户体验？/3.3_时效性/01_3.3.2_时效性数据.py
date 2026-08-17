# 01_3.3.2_时效性数据

"""
Lecture: 3_第三部分_什么决定用户体验？/3.3_时效性
Content: 01_3.3.2_时效性数据
"""

from typing import List, Tuple


def detect_burst(volume: List[float], threshold: float = 3.0,
                 window: int = 3) -> List[int]:
    """Detect sudden-spike (突发) points in a search-volume series.

    A day is 'bursty' when its volume is >= threshold times the rolling
    average of the `window` preceding days.

    Returns:
        Indices flagged as bursty.
    """
    burst = []
    for i in range(window, len(volume)):
        base = sum(volume[i - window:i]) / window
        if base > 0 and volume[i] >= threshold * base:
            burst.append(i)
    return burst


def main() -> None:
    print("时效性数据: 突发时效性检测 Demo")
    # 某查询词的每日搜索量(千次), 注意第6天突增
    volume = [10, 11, 9, 10, 12, 45, 90, 70, 20, 15, 13, 12]
    days = [f"d{i+1}" for i in range(len(volume))]

    spikes = detect_burst(volume)
    print("搜索量序列:", list(zip(days, volume)))
    print(f"识别为突发时效性的时间点: {[days[i] for i in spikes]}")

    print("\n处理: 为时效性强查询建立新文档索引(如24h/7天), 并调整排序时效权重。")


if __name__ == "__main__":
    main()