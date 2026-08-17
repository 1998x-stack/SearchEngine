# 00_4.5.1_改写的目标

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.5_查询词改写
Content: 00_4.5.1_改写的目标
"""

from typing import List


def recall(returned: int, relevant: int) -> float:
    """Recall = 召回的relevant / 总relevant."""
    return returned / relevant if relevant else 0.0


def show_goal() -> None:
    print("查询词语改写目标: 提高召回覆盖(解决语义鸿沟与召回量不足)")
    print("  原查询: 布洛芬不良反应")
    print("  改写:   布洛芬副作用 / 布洛芬注意事项")
    print("  -> 召回更多相关文档, 提升 recall")


def main() -> None:
    print("改写目标 (Rewrite Goal) Demo")
    show_goal()
    base_recall = recall(returned=80, relevant=200)
    rewrite_recall = recall(returned=150, relevant=200)
    print(f"\n改写前 recall = {base_recall:.2f}")
    print(f"改写后 recall = {rewrite_recall:.2f}")


if __name__ == "__main__":
    main()