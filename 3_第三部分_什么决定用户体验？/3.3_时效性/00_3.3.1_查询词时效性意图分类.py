# 00_3.3.1_查询词时效性意图分类

"""
Lecture: 3_第三部分_什么决定用户体验？/3.3_时效性
Content: 00_3.3.1_查询词时效性意图分类
"""

# 词表: 含明确时间节点的强时效词 / 周期事件词
STRONG_WORDS = ["最新", "最热", "今天", "今日", "政策", "天气", "打折", "行情"]
PERIODIC_WORDS = ["圣诞", "元旦", "春节", "双十一", "618", "高考", "奥运", "世界杯"]


def recency_level(query: str) -> str:
    """Return 强/中/弱/无 from the query's literal recency signals."""
    if any(w in query for w in STRONG_WORDS):
        return "强"
    if "攻略" in query or "测评" in query:
        return "弱"
    return "中"


def intent_class(query: str) -> str:
    """Classify into 突发/一般/周期."""
    if any(w in query for w in PERIODIC_WORDS):
        return "周期时效性"
    if recency_level(query) != "无":
        return "一般时效性"
    return "突发时效性(需数据挖掘识别)"


def main() -> None:
    print("查询词时效性意图分类 Demo")
    queries = ["最新手机评测", "双十一购物攻略", "科比去世", "杭州租房攻略", "今日天气"]
    for q in queries:
        print(f"  {q:<12} 强度={recency_level(q)}  类别={intent_class(q)}")


if __name__ == "__main__":
    main()