# 00_4.4.1_影响下游链路调用的意图

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.4_意图识别
Content: 00_4.4.1_影响下游链路调用的意图
"""

from typing import List


INTENT_RULES = {
    "时效性": ["最新", "今日", "天气", "政策"],
    "地域性": ["附近", "同城", "周边"],
    "求购": ["买", "求购", "推荐", "排行"],
    "用户名": ["@", "账号", "主页"],
}


def recognize(query: str) -> List[str]:
    """Return the intents a query triggers (drives downstream dispatch)."""
    return [intent for intent, rules in INTENT_RULES.items()
            if any(r in query for r in rules)]


def dispatch(intents: List[str]) -> List[str]:
    """Map intents to downstream pipelines (示例)."""
    mapping = {
        "时效性": ["时效索引召回", "时效加权排序"],
        "地域性": ["POI/地域召回链路"],
        "求购": ["购物意图召回", "转化预估"],
        "用户名": ["用户主页召回"],
    }
    jobs = []
    for i in intents:
        jobs.extend(mapping.get(i, []))
    return jobs


def main() -> None:
    print("意图识别 -> 下游链路调用 Demo")
    for q in ["附近奶茶 求购", "今天 天气", "@某博主"]:
        intents = recognize(q)
        print(f" 查询 '{q}'\n   意图: {intents}\n   下游: {dispatch(intents)}")


if __name__ == "__main__":
    main()