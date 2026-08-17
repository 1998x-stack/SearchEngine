# 04_3.4.5_实验结果

"""
Lecture: 3_第三部分_什么决定用户体验？/3.4_地域性
Content: 04_3.4.5_实验结果
"""

from typing import Dict


def delta_pt(baseline: Dict[str, float], treatment: Dict[str, float]) -> Dict[str, float]:
    """Effective-CTR uplift (percentage-point) by intent category."""
    return {k: treatment[k] - baseline[k] for k in baseline}


def main() -> None:
    print("地域性实验结果 (A/B) Demo")
    baseline = {
        "显式附近意图": 8.0, "隐式附近意图": 6.0,
        "隐式同城意图": 5.0, "显式同城意图": 4.0,
    }
    treatment = {
        "显式附近意图": 16.5, "隐式附近意图": 7.0,
        "隐式同城意图": 6.4, "显式同城意图": 3.8,
    }
    print("有效点击率变化(pt):")
    for k, delta in delta_pt(baseline, treatment).items():
        print(f"  {k:<8}: {delta:+.1f}pt")

    print("结论: 显式同城意图下降 -> 未全量推行; 其余意图显著提升。")
    print("另: 外显距离/POI 标签使有效点击率 -0.3pt 但深度消费占比 +0.2pt。")


if __name__ == "__main__":
    main()