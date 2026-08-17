# 00_5.3.1_挖掘曝光日志

"""
Lecture: 5_第五部分_召回/5.3_离线召回
Content: 00_5.3.1_挖掘曝光日志
"""

from typing import Dict, List, Tuple


def mine_click_features(log_lines: List[Tuple[str, str, int]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Aggregate per-(query,doc) exposure/click/interaction stats from logs.

    Args:
        log_lines: list of (query, doc, clicked_flag).

    Returns:
        (q, d) -> {exposure, click_rate, interaction_rate}.
    """
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for q, d, clicked in log_lines:
        s = stats.setdefault((q, d), {"exposures": 0, "clicks": 0, "interactions": 0})
        s["exposures"] += 1
        s["clicks"] += clicked
    for k, s in stats.items():
        s["ctr"] = s["clicks"] / s["exposures"]
    return stats


def main() -> None:
    print("挖掘曝光日志 (Mine Exposure Logs) Demo")
    logs = [
        ("python 教程", "docA", 1), ("python 教程", "docA", 0),
        ("python 教程", "docB", 1), ("爬虫", "文档工程档", 0),
        ("爬虫", "docB", 1),
    ]
    feats = mine_click_features(logs)
    print("(查询,文档) 聚合统计:")
    for (q, d), s in feats.items():
        print(f"  ({q}, {d}): 曝光={s['exposures']} 点击={s['clicks']} CTR={s['ctr']:.2f}")


if __name__ == "__main__":
    main()