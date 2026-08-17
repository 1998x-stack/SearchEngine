# 01_5.3.2_离线搜索链路

"""
Lecture: 5_第五部分_召回/5.3_离线召回
Content: 01_5.3.2_离线搜索链路
"""

from typing import Dict, List, Tuple


def build_kv_index(high_rel_pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """离线构建 KV 索引: 查询词 -> 高相关文档列表."""
    kv: Dict[str, List[str]] = {}
    for q, d in high_rel_pairs:
        kv.setdefault(q, []).append(d)
    return kv


def main() -> None:
    print("离线搜索链路 (Offline Search Link) Demo")
    pairs = [("布洛芬", "docA"), ("布洛芬", "docB"), ("口红", "docC")]
    kv = build_kv_index(pairs)
    print("离线挖掘高相关二元组, 构建 KV 索引:")
    for q, docs in kv.items():
        print(f"  {q} -> {docs}")
    print("线上直接读取 KV 索引补充召回结果。")


if __name__ == "__main__":
    main()