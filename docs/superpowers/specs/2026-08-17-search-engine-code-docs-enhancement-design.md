# Search Engine Tutorial — Code & Docs Enhancement Design

**Date:** 2026-08-17
**Status:** Validated design (approved by user)

## 1. Overview

This is a Chinese search-engine tutorial organized into 7 parts, subdivided into
sections, with ~95 topic files. Every topic has a `.md` (theory) and a `.py` file.
Today, **94 of the 95 `.py` files are empty stubs** containing only a header
docstring. The single exception (`3.1.2_文本匹配分数.py`, ~600 lines) is a fully
implemented, exemplar-quality `TfIdfVectorizer` and defines the target style.

**Goal:** turn every stub `.py` into a real, runnable, exemplar-quality, educational
implementation, and lightly enhance every `.md` so it explains its code. Do this for
all 7 parts, in one spec + one plan, implemented in order.

## 2. Scope

### In scope
- All 94 stub `.py` files → full, runnable, exemplar-quality implementations.
- All 94 `.md` files → append a short "代码示例 (Code Example)" section.
- `README.md` → brief addition noting code is runnable + how to run it.
- `main.py` and `modify_files.sh` → left untouched.

### Out of scope (explicitly not doing)
- No real model training; no heavy ML framework installs.
- No rewrites of existing theory in `.md` (append-only).
- No changes to file/directory structure or names.
- No network/API calls, no real web crawling/indexing of real web data.
- No multi-language release (docs stay Chinese, code comments Chinese, docstrings English).

## 3. Method

It is not feasible to write this as one giant monolith. Approach:

1. **Single spec + single plan** spanning all 7 parts (user chose this sequencing).
2. **Style lock-in via a small pilot batch first** — implement a handful of topics
   spanning different domains (evaluation metrics, retrieval, ML concepts), run and
   review them, lock conventions, then scale.
3. **Part-by-part verification** — after each part, run every `.py` in that part and
   confirm clean execution with expected output.
4. **Parallelism candidate** — the 7 parts are independent; the plan may dispatch
   subagents to implement different parts/sections in parallel, then one
   human-in-the-loop review pass.
5. **Dense-ML realism guardrail** — topics that are inherently dense-model (BERT
   attention, distillation, pretraining, finetuning) get a compact numpy theory demo
   exercising the core math (scaled dot-product attention forward pass, embedding
   lookup + cosine recall, etc.) rather than a fake full training run.

## 4. Code Standards (.py world) — Section 1

- **Imports:** only `numpy` and `matplotlib` (both installed). No scipy, sklearn,
  torch, transformers, jieba. scipy usages (e.g. the exemplar's `scipy.sparse`) are
  replaced with numpy equivalents (dense numpy arrays for tutorial-sized data).
- **Per-file structure:**
  1. Header comment + module docstring (keep existing Lecture/Content lines)
  2. Import block
  3. One or more focused classes/functions implementing the topic's core concept
  4. English docstrings with Args/Returns; Chinese `#` comments for the teaching narrative
  5. Type hints throughout
  6. `def main():` + `if __name__ == "__main__":` building a small toy dataset,
     running the code, and printing structured, reader-followable output
- **Self-contained:** every file runs standalone (`python3 <file>.py`) with no imports
  from sibling files.
- **Runnable & verified:** every script must execute without errors before considered
  done.

## 5. Per-File Cataloging & Doc Adaptation — Section 2

- **Topic catalog:** before implementation, enumerate all 94 topics (from `main.py`'s
  `structure`) and classify each into a domain bucket, e.g.:
  - Evaluation metrics (Precision/Recall/DCG/NDCG/IoU …)
  - Retrieval (inverted index, BM25, vector recall)
  - ML concepts (sigmoid/softmax/losses/GD)
  - Text processing (tokenization, NER, TF-IDF)
  - Ranking (pointwise/pairwise/listwise)
  - Recommendation (SUG, Q2Q/D2Q)
  - ML training (pretrain/finetune/distill → numpy theory demos)
  Classification lets shared scaffolding repeat consistently across files.
- **`.md` light touch:** for each topic, append a **"代码示例 (Code Example)"** section
  summarizing what the `.py` demonstrates and how to run it (`python3 <file>.py`).
  Existing theory stays intact (append-only).
- **No structural changes** to layout, names, or the `main.py`/`modify_files.sh`
  scaffold.

## 6. Definition of Done (global)

1. Every one of the 94 `.py` files runs `python3 <file>.py` without error and prints
   meaningful demo output.
2. Every `.md` has its Code Example section.
3. `README.md` notes runnable code + run instructions.
4. No heavy dependencies introduced; each part verified.
5. Updated README reflects the enhancement.

## 7. Environment

- Python 3.9.6
- Installed: numpy 2.0.2, matplotlib 3.9.4
- Not installed (and intentionally not used): scipy, scikit-learn, torch,
  transformers, jieba