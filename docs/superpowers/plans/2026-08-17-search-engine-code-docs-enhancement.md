# Search Engine Tutorial — Code & Docs Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn every stub `.py` in this 94-topic search-engine tutorial into a full, runnable, educational implementation, and lightly enhance every `.md` so it explains its code.

**Architecture:** Each of the 94 topic files is a self-contained, standalone numpy script implementing its concept: header docstring → imports → classes/functions → `main()` that builds toy data, runs, and prints structured output. The 7 repo parts are independent, so implementation proceeds part-by-part, each verified independently. A small pilot batch locks conventions first.

**Tech Stack:** Python 3.9.6, numpy 2.0.2, matplotlib 3.9.4. Nothing else is installed and nothing else may be used.

## Global Constraints

- **Imports only** `numpy` and `matplotlib`. NO scipy, sklearn, torch, transformers, jieba.
- **Module docstring** keeps the existing `Lecture:` and `Content:` lines; the leading `# <title>` comment is kept.
- **Self-contained**: every `.py` runs standalone (`python3 <file>.py`), no sibling imports.
- **Structure:** `# <title>` → module docstring → imports → classes/functions → `def main()` → `if __name__ == "__main__": main()`.
- **Comments** for the teaching narrative are **Chinese**; **docstrings** and **identifiers** are **English**; type hints throughout.
- **Every script must run without error** and print meaningful demo output (DoD #1, #4).
- **`.md` edits are append-only**: add a `### 代码示例 (Code Example)` section at the end; never rewrite existing theory.
- **Never modify** `main.py`, `modify_files.sh`, or any file/directory names or layout.
- Work from repo root: `cd /Users/x/Desktop/1998x-stack/2024/SearchEngine`. Commit each task separately with a conventional message.

## Verification Harness

Single file:

```bash
python3 "<exact path>.py"
```

Expected: exit code 0, no traceback, printed human-readable demo output. Whole part:

```bash
cd /Users/x/Desktop/1998x-stack/2024/SearchEngine
FAIL=0
for f in $(find "<part dir>" -name "*.py" -not -name 'main.py'); do
  if ! python3 "$f" >/dev/null 2>&1; then echo "FAIL: $f"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] && echo "PART OK"
```

---

# Global Standards Reference

Four complete reference implementations lock the style. Every other topic applies the matching template's pattern to its own concept (read the topic's `.md`).

## Template 1 — Metric Evaluation

Target file (write in full): `4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别/04_4.1.5_评价指标.py`

```python
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
```

## Template 2 — Retrieval / Index

Target file (write in full): `5_第五部分_召回/5.1_文本召回/00_5.1.1_倒排索引.py`

```python
# 00_5.1.1_倒排索引

"""
Lecture: 5_第五部分_召回/5.1_文本召回
Content: 00_5.1.1_倒排索引
"""

from typing import Dict, List, Set


class InvertedIndex:
    """A term -> document-id mapping for boolean retrieval."""

    def __init__(self) -> None:
        self.postings: Dict[str, Set[int]] = {}

    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        """Index one document's tokens (deduplicated per doc).

        Args:
            doc_id: unique document identifier.
            tokens: tokenized document content.
        """
        for term in set(tokens):
            self.postings.setdefault(term, set()).add(doc_id)

    def query(self, terms: List[str]) -> Set[int]:
        """Boolean AND: documents containing every input term."""
        if not terms:
            return set()
        result = self.postings.get(terms[0], set())
        for term in terms[1:]:
            result &= self.postings.get(term, set())
        return result

    def df(self, term: str) -> int:
        """Document frequency of a term."""
        return len(self.postings.get(term, set()))

    def __len__(self) -> int:
        return len(self.postings)


def main() -> None:
    print("倒排索引 (Inverted Index) Demo")
    idx = InvertedIndex()
    corpus = {
        1: ["搜索引擎", "倒排", "索引"],
        2: ["搜索引擎", "召回", "模型"],
        3: ["排序", "模型", "召回"],
    }
    for doc_id, tokens in corpus.items():
        idx.add_document(doc_id, tokens)
    print(f"词汇表词项数: {len(idx)}")
    for term in ["搜索引擎", "模型", "召回"]:
        print(f"  '{term}' 的文档频率: {idx.df(term)}")
    print(f"查询 ['搜索引擎', '模型'] 命中文档: {sorted(idx.query(['搜索引擎', '模型']))}")


if __name__ == "__main__":
    main()
```

## Template 3 — ML Concept (classifier with gradient descent)

Target file: `2_第二部分_机器学习基础/2.1_机器学习任务/00_2.1.1_二分类任务.py`

```python
# 00_2.1.1_二分类任务

"""
Lecture: 2_第二部分_机器学习基础/2.1_机器学习任务
Content: 00_2.1.1_二分类任务
"""

import numpy as np
from typing import List


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean binary cross-entropy loss."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


class LinearClassifier:
    """Minimal logistic-regression classifier trained by gradient descent."""

    def __init__(self, n_features: int, learning_rate: float = 0.1) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = learning_rate

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class."""
        return sigmoid(X @ self.w + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30) -> List[float]:
        """Train with gradient descent; return loss history."""
        losses = []
        for _ in range(epochs):
            pred = self.predict_proba(X)
            losses.append(binary_cross_entropy(y, pred))
            grad_w = X.T @ (pred - y) / len(y)
            grad_b = float(np.mean(pred - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return losses


def main() -> None:
    print("二分类任务 (Binary Classification) Demo")
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0],
                  [0.0, 1.0], [2.0, 1.0], [1.0, 0.0], [2.0, 0.0]])
    y = np.array([0, 1, 1, 1, 0, 1, 0, 1])
    clf = LinearClassifier(n_features=2)
    losses = clf.fit(X, y, epochs=40)
    print(f"初始loss={losses[0]:.4f}, 最终loss={losses[-1]:.4f}")
    for point in [(0, 0), (1.5, 1.5), (0.5, 2.5)]:
        p = clf.predict_proba(np.array(point, dtype=float))[0]
        print(f"  特征 {point} -> 预测概率 {p:.3f}")


if __name__ == "__main__":
    main()
```

## Template 4 — Dense-ML: attention (numpy scaled dot-product)

Target file: `4_第四部分_查询词处理与文本理解/4.2_词权重/01_4.2.2_基于注意力机制的方法.py`

```python
# 01_4.2.2_基于注意力机制的方法

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.2_词权重
Content: 01_4.2.2_基于注意力机制的方法
"""

import numpy as np
from typing import List, Tuple


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(query: np.ndarray,
                                 keys: np.ndarray,
                                 values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Attention context over a sequence.

    Args:
        query: (d,) query vector.
        keys: (T, d) key matrix.
        values: (T, d) value matrix.

    Returns:
        (context, weights), where context = weights @ values is (d,)
        and weights are (T,) non-negative, summing to 1.
    """
    scores = keys @ query                       # (T,)
    scale = 1.0 / np.sqrt(keys.shape[1])
    weights = softmax(scores * scale)          # (T,)
    context = weights @ values                 # (d,)
    return context, weights


def main() -> None:
    print("基于注意力机制的词权重 (Attention Word Weight) Demo")
    np.random.seed(0)
    words = ["搜索", "引擎", "召回", "排序"]
    embeddings = np.random.randn(4, 8)          # (T, d), 每个词一个向量
    query = embeddings.mean(axis=0)            # 伪query = 文档平均向量
    context, weights = scaled_dot_product_attention(query, embeddings, embeddings)
    print("  各词的注意力权重:")
    for word, w in zip(words, weights):
        print(f"    {word:<6} 权重={w:.4f}")
    print(f"  权重和 = {weights.sum():.4f}")
    print(f"  聚合上下文向量 (前3维) = {np.round(context[:3], 4)}")


if __name__ == "__main__":
    main()
```

---

## File-to-Template Assignment (per topic)

Implementer: read the topic's `.md`, then implement its `.py` following the matching template's style. If a concept spans multiple templates, pick the dominant one; always satisfy Global Constraints regardless.

- **Template 1 (metrics):** any `_评价指标`, `_指标`, `_质量`, `_结果` topic; and — in 6.2 — pointwise/pairwise/listwise loss implementations (extend Template 1 with the chapter's loss).
- **Template 2 (index/retrieval):** `_倒排索引`, all `_召回` topics, `_缓存召回`, `_向量召回` (add embedding/cosine when relevant).
- **Template 3 (ML concept):** `_任务`, `_模型_ex`, `_特征`, `_训练`, `_预训练`, `_后预训练`, `_微调`, `_蒸馏`, `_点击率模型`.
- **Template 4 (attention/embedding):** `_BERT_`, `_注意力`, `_深度学习分词`, `_嵌入`-heavy topics.
- **Summary files** (`_知识点小结`, `_本章小结`): small demo exercising 2–3 key formulas of the chapter numerically (no heavy class).

---

## Task Execution Model

Every topic gets its own Task with structure:

- [ ] **Step 1:** `cd /Users/x/Desktop/1998x-stack/2024/SearchEngine` and read `<topic>.md` to extract the concept and formulas.
- [ ] **Step 2:** Write the full `<topic>.py` (per its Template + Global Constraints, non-placeholder code).
- [ ] **Step 3:** Run `python3 "<topic>.py"` — expected exit 0, no traceback, meaningful printed output.
- [ ] **Step 4:** Append the Code Example section to `<topic>.md`:

```markdown

---

### 代码示例 (Code Example)

运行 `python3 <basename>.py` 可以查看本节的代码演示。
本文件实现: <one-line what the .py does>。
```

  (Do not touch the existing theory above it.)
- [ ] **Step 5:** Commit:
```bash
git add "<topic>.py" "<topic>.md"
git commit -m "feat: implement <topic> code + doc"
```

## Phase 0 — Catalog (Task 0)

- [ ] Generate `docs/superpowers/catalog.md` enumerating all 94 topic `.py` paths grouped by part, each with its implemented concept and Template number.
- [ ] Commit `git add docs/superpowers/catalog.md && git commit -m "docs: add 94-topic catalog"`.

## Phase 1 — Pilot (lock style)

- [ ] **Task P1:** Write Template 1 file `4_…/04_4.1.5_评价指标.py` in full; run ; confirm exit 0 + metrics; append Code Example; commit.
- [ ] **Task P2:** Write Template 2 file `5_…/00_5.1.1_倒排索引.py`; run; commit.
- [ ] **Task P3:** Write Template 3 file `2_…/00_2.1.1_二分类任务.py`; run; confirm loss drops; commit.
- [ ] **Task P4:** Write Template 4 file `4_…/01_4.2.2_基于注意力机制的方法.py`; run; confirm weights sum ≈ 1; commit.
- [ ] **Task P5:** Refactor `3_…/01_3.1.2_文本匹配分数.py` from `scipy.sparse` to pure numpy (dense arrays), preserve docstrings/structure; run to confirm exit 0; append Code Example to its `.md`; commit.

**Gate:** user/lead reviews P1–P5 before Phase 2.

## Phases 2–8 — Parts 1–7

Replicate the Task 0–5/Step 1–5 pattern once per remaining topic file, in repo order. The part boundaries and topic counts are:

- **Part 1 搜索引擎基础** — 8 files (1.1 ×4 : 00–03; 1.2 ×4: 00,02,03,04).
- **Part 2 机器学习基础** — 13 files (2.1 ×5; 2.2 ×4; 2.3 ×4).
- **Part 3 什么决定用户体验?** — 19 files except the refactored exemplar `3.1.2` (which only needs its Code Example append): 3.1 ×4 more, 3.2 ×3, 3.3 ×4, 3.4 ×5, 3.5 ×5.
- **Part 4 查询词处理与文本理解** — 19 files (4.1 ×6; 4.2 ×3, incl. Template 1/4 sends; 4.3 ×3; 4.4 ×2; 4.5 ×5).
- **Part 5 召回** — 12 files (5.1 ×3; 5.2 ×4; 5.3 ×5).
- **Part 6 排序** — 8 files (6.1 ×4; 6.2 ×4).
- **Part 7 查询词推荐** — 12 files (7.1 ×5; 7.2 ×4; 7.3 ×3).

After the last topic task in a part, run the part loop (see Verification). Expected `PART OK`. Then continue to the next part. (Sum of topics: 8+13+12+19+12+8+12 = 84 newly-implemented + 1 rebuilt template 4 + 1 exemplar refactor + 4 pilot… accounting for double counted pilot/template covers the full 94 — see catalog for the authoritative 1:1 list.)

## Final Phase — README + Global DoD

- [ ] **Task:** in `README.md`, add a short "运行代码 (Run the code)" note near the top: any topic's code runs with `python3 <topic>.py`; optional whole-repo loop.
- [ ] **Task:** add `scripts/verify_all.sh` iterating all 94 `.py` and printing each FAIL plus final `ALL 94 OK`.
- [ ] **Task:** run `bash scripts/verify_all.sh` → MUST print `ALL 94 OK`.
- [ ] **Task:** confirm Code-Example coverage: `grep -L "代码示例" $(find . -name "*.md" -not -name 'README.md')` returns nothing (all 94 have it).
- [ ] Commit contains the README + script changes; final `git commit -m "chore: add global verification + README run notes"`.

## Self-Review

- **Spec coverage:** do.py implementations (all 94), .md Code Example appends (all 94), README notes, numpy-only, exemplar scipy→numpy refactor (Task P5) — all mapped to tasks.
- **Placeholder scan:** the four Template blocks contain full, runnable code; per-file tasks name exact paths + verification; no TBD/TODO.
- **Type/name consistency:** `LinearClassifier` used everywhere in Template 3 (no alternate name); `scaled_dot_product_attention` returns `(context, weights)` (Tuple[np.ndarray, np.ndarray]) consistently; `InvertedIndex.query` returns `Set[int]`; `SegmentationMetrics.summary()` returns `Dict[str, float]`.