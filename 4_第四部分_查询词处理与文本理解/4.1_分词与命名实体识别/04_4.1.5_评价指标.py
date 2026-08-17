# 04_4.1.5_评价指标

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别
Content: 04_4.1.5_评价指标
"""

from typing import Dict, List, Tuple


class SegmentationMetrics:
    """Precision/Recall/F1 for word-segmentation evaluation.

    A token is correct when it appears in both the gold and the
    predicted segmentation.
    """

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, ground_truth: List[str], prediction: List[str]) -> None:
        """Accumulate one sample's token-level counts.

        Args:
            ground_truth: gold tokens.
            prediction: predicted tokens.
        """
        g, p = set(ground_truth), set(prediction)
        self.tp += len(g & p)
        self.fp += len(p - g)
        self.fn += len(g - p)

    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def f1(self) -> float:
        """F1 = 2*P*R / (P + R)."""
        p, r = self.precision(), self.recall()
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0

    def summary(self) -> Dict[str, float]:
        """All metrics as a dict, ready to print."""
        return {"precision": self.precision(), "recall": self.recall(), "f1": self.f1()}


def main() -> None:
    print("=== 分词评价指标 (Segmentation Metrics) Demo ===")
    evaluator = SegmentationMetrics()
    cases: List[Tuple[List[str], List[str]]] = [
        (["我", "爱", "北京"], ["我", "爱", "北京"]),      # 完全正确
        (["我", "爱", "北京"], ["我爱", "北京"]),          # 部分正确
        (["北京", "欢迎", "你"], ["北京", "迎接", "你"]),  # 有误
    ]
    for i, (gold, pred) in enumerate(cases, start=1):
        evaluator.add(gold, pred)
        print(f"样本{i}: 正确={gold} 预测={pred}")
    print("\n累计指标:")
    for name, value in evaluator.summary().items():
        print(f"  {name} = {value:.4f}")


if __name__ == "__main__":
    main()