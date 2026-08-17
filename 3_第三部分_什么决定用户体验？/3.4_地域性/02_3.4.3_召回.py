# 02_3.4.3_召回

"""
Lecture: 3_第三部分_什么决定用户体验？/3.4_地域性
Content: 02_3.4.3_召回
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from typing import Dict, List, Tuple


def geo_recall(pois: Dict[str, float], user_loc: Tuple[float, float],
               radius_km: float) -> List[str]:
    """Filter POIs within a distance radius (metric approx. on lon/lat).

    Args:
        pois: POI name -> (lat, lon).
        user_loc: user (lat, lon).
        radius_km: max distance in km.

    Returns:
        POIs within radius, nearest first.
    """
    lat0, lon0 = user_loc
    results = []
    for name, (lat, lon) in pois.items():
        # 简化球面距离(度->km 近似)
        dist = np.hypot((lat - lat0) * 111, (lon - lon0) * 111 * np.cos(np.radians(lat0)))
        if dist <= radius_km:
            results.append((name, float(dist)))
    results.sort(key=lambda x: x[1])
    return results


def main() -> None:
    print("地域性召回: 地理位置过滤 Demo")
    pois = {
        "火锅店A": (31.2304, 121.4737),
        "咖啡店B": (31.2500, 121.5000),
        "商场C": (31.3000, 121.5500),
        "书店D": (31.2200, 121.4500),
    }
    user = (31.2304, 121.4737)   # 用户位置
    for r in [2.0, 5.0]:
        hits = geo_recall(pois, user, r)
        print(f"半径 {r} km 内召回: {[(n, round(d, 2)) for n, d in hits]}")


if __name__ == "__main__":
    main()