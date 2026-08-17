# 04_5.3.5_知识点小结

"""
Lecture: 5_第五部分_召回/5.3_离线召回
Content: 04_5.3.5_知识点小结
"""


def main() -> None:
    print("=== 离线召回小结 (Offline Recall Summary) Demo ===")
    print("离线召回: 离线挖掘并构建索引, 线上直接读取补充召回")
    print("  挖掘曝光日志 -> 高相关二元组(q,d)")
    print("  反向召回(文档->查询) / 查询词改写 / 缓存(KD)召回")
    print("特点: 高质量、快, 补充文本/向量召回的不足")


if __name__ == "__main__":
    main()