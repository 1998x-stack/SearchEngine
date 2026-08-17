# 03_3.3.4_知识点小结

"""
Lecture: 3_第三部分_什么决定用户体验？/3.3_时效性
Content: 03_3.3.4_知识点小结
"""

from typing import List


def recency_level(query: str) -> str:
    if any(w in query for w in ["最新", "今日", "政策", "天气", "打折"]):
        return "强"
    return "中"


def classify(query: str) -> str:
    if any(w in query for w in ["双十一", "春节", "高考", "奥运"]):
        return "周期时效性"
    if "最新" in query:
        return "一般时效性"
    return "突发/一般 (依数据识别)"


def burst_points(volume: List[float], threshold: float = 3.0) -> List[int]:
    burst = []
    for i in range(3, len(volume)):
        base = sum(volume[i - 3:i]) / 3
        if base > 0 and volume[i] >= threshold * base:
            burst.append(i)
    return burst


def main() -> None:
    print("=== 时效性知识点小结 (Recency Summary) Demo ===")
    for q in ["最新手机评测", "双十一购物", "突发新闻", "今日天气"]:
        print(f"  {q:<8} 强度={recency_level(q)}  类别={classify(q)}")

    vol = [10, 11, 9, 10, 12, 50, 88, 60, 22, 15]
    print(f"\n搜索量序列: {vol}")
    print(f"突发时效性检测时间点: {[f'd{i+1}' for i in burst_points(vol)]}")

    print("\n小结: 识别时效意图后, 建立新文档索引并在排序中加权时效性。")


if __name__ == "__main__":
    main()