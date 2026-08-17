# 03_7.1.4_文档内推词

"""
Lecture: 7_第七部分_查询词推荐/7.1_查询词推荐的场景
Content: 03_7.1.4_文档内推词
"""


def doc_inner_queries(doc_queries: dict) -> None:
    """文档内推词: 在阅读文档时给出相关查询词."""
    for doc, queries in doc_queries.items():
        print(f"  文档 '{doc}' -> {queries}")


def main() -> None:
    print("文档内推词 (文档阅读场景) Demo")
    docs = {
        "《口红选购指南》": ["口红平价推荐", "显白口红", "哑光口红"],
        "《冬季穿搭技巧》": ["大衣搭配", "围巾推荐"],
    }
    doc_inner_queries(docs)
    print("说明: 用户在文档内阅读时推荐相关查询, 促进后续搜索。")


if __name__ == "__main__":
    main()