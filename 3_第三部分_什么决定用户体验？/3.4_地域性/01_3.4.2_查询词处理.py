# 01_3.4.2_查询词处理

"""
Lecture: 3_第三部分_什么决定用户体验？/3.4_地域性
Content: 01_3.4.2_查询词处理
"""

from typing import List, Tuple


GEO_HINTS = ["附近", "周边", "同城", "最近", "哪里"]


def segment(query: str) -> List[str]:
    """QP 分词(用空格分隔的简单词典分词)."""
    return [w for w in query.split(" ") if w]


def geo_intent(tokens: List[str]) -> Tuple[str, float]:
    """判断地理意图类型与强度."""
    if any("附近" in t or "周边" in t or "哪里" in t for t in tokens):
        return "显式附近意图", 1.0
    if any("同城" in t for t in tokens):
        return "显式同城意图", 1.0
    for t in tokens:
        if any(h in t for h in ["店", "馆", "火锅", "餐厅"]):
            return "隐式附近意图", 0.5
    return "无地理意图", 0.0


def main() -> None:
    print("查询词处理 (QP) 地理意图识别 Demo")
    for q in ["附近 火锅店", "同城 咖啡", "北京 周边游", "美甲", "杭州 租房 攻略"]:
        tokens = segment(q)
        intent, strength = geo_intent(tokens)
        print(f"  查询 '{q}' -> 分词={tokens} 意图={intent} 强度={strength}")


if __name__ == "__main__":
    main()