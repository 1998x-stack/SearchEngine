structure = {
    "1_第一部分_搜索引擎基础": [
        {
            "1.1_搜索引擎技术概要": [
                "1.1.1_基本概念",
                "1.1.2_什么决定用户满意度？",
                "1.1.3_搜索引擎链路",
                "1.1.4_知识点小结"
            ]
        },
        {
            "1.2_搜索引擎的评价指标": [
                "1.2.1_用户规模与留存指标",
                {
                    "1.2.1.1_用户规模": [],
                    "1.2.1.2_用户留存": []
                },
                "1.2.2_中间过程指标",
                "1.2.3_人工体验评估",
                "1.2.4_知识点小结"
            ]
        }
    ],
    "2_第二部分_机器学习基础": [
        {
            "2.1_机器学习任务": [
                "2.1.1_二分类任务",
                "2.1.2_多分类任务",
                "2.1.3_回归任务",
                "2.1.4_排序任务",
                "2.1.5_知识点小结"
            ]
        },
        {
            "2.2_离线评价指标": [
                "2.2.1_pointwise_评价指标",
                "2.2.2_pairwise_评价指标",
                "2.2.3_listwise_评价指标",
                "2.2.4_知识点小结"
            ]
        },
        {
            "2.3_NLP模型的训练": [
                "2.3.1_预训练任务",
                "2.3.2_后预训练",
                "2.3.3_微调",
                "2.3.4_蒸馏"
            ]
        }
    ],
    "3_第三部分_什么决定用户体验？": [
        {
            "3.1_相关性": [
                "3.1.1_相关性的定义与分档",
                "3.1.2_文本匹配分数",
                "3.1.3_相关性_BERT_模型",
                "3.1.4_相关性模型的训练",
                "3.1.5_知识点小结"
            ]
        },
        {
            "3.2_内容质量": [
                "3.2.1_EAT_分数",
                "3.2.2_文本质量",
                "3.2.3_图片质量"
            ]
        },
        {
            "3.3_时效性": [
                "3.3.1_查询词时效性意图分类",
                "3.3.2_时效性数据",
                "3.3.3_下游链路的承接",
                "3.3.4_知识点小结"
            ]
        },
        {
            "3.4_地域性": [
                "3.4.1_POI_识别",
                "3.4.2_查询词处理",
                "3.4.3_召回",
                "3.4.4_排序",
                "3.4.5_实验结果"
            ]
        },
        {
            "3.5_个性化与点击率预估": [
                "3.5.1_特征",
                "3.5.2_精排点击率模型",
                "3.5.3_粗排点击率模型",
                "3.5.4_模型训练",
                "3.5.5_知识点小结"
            ]
        }
    ],
    "4_第四部分_查询词处理与文本理解": [
        {
            "4.1_分词与命名实体识别": [
                "4.1.1_基于词典的分词方法",
                "4.1.2_词典的构造",
                "4.1.3_基于深度学习的分词方法",
                "4.1.4_命名实体识别",
                "4.1.5_评价指标",
                "4.1.6_知识点小结"
            ]
        },
        {
            "4.2_词权重": [
                "4.2.1_词权重的定义与标注方法",
                "4.2.2_基于注意力机制的方法",
                "4.2.3_知识点小结"
            ]
        },
        {
            "4.3_类目识别": [
                "4.3.1_多标签分类模型",
                "4.3.2_离线评价指标",
                "4.3.3_知识点小结"
            ]
        },
        {
            "4.4_意图识别": [
                "4.4.1_影响下游链路调用的意图",
                "4.4.2_知识点小结"
            ]
        },
        {
            "4.5_查询词改写": [
                "4.5.1_改写的目标",
                "4.5.2_基于分词的改写",
                "4.5.3_基于相关性的改写",
                "4.5.4_基于意图的改写",
                "4.5.5_本章小结"
            ]
        }
    ],
    "5_第五部分_召回": [
        {
            "5.1_文本召回": [
                "5.1.1_倒排索引",
                "5.1.2_文本召回",
                "5.1.3_知识点小结"
            ]
        },
        {
            "5.2_向量召回": [
                "5.2.1_相关性向量召回",
                "5.2.2_个性化向量召回",
                "5.2.3_线上推理",
                "5.2.4_知识点小结"
            ]
        },
        {
            "5.3_离线召回": [
                "5.3.1_挖掘曝光日志",
                "5.3.2_离线搜索链路",
                "5.3.3_反向召回",
                "5.3.4_结合查询词改写与缓存召回",
                "5.3.5_知识点小结"
            ]
        }
    ],
    "6_第六部分_排序": [
        {
            "6.1_排序的基本原理": [
                "6.1.1_融合模型的特征",
                "6.1.2_融合规则_vs_融合模型",
                "6.1.3_融合模型训练数据",
                "6.1.4_知识点小结"
            ]
        },
        {
            "6.2_训练融合模型的方法": [
                "6.2.1_pointwise_训练方法",
                "6.2.2_pairwise_训练方法",
                "6.2.3_listwise_训练方法",
                "6.2.4_知识点小结"
            ]
        }
    ],
    "7_第七部分_查询词推荐": [
        {
            "7.1_查询词推荐的场景": [
                "7.1.1_搜索前推词",
                "7.1.2_查询建议",
                "7.1.3_搜索结果页推词",
                "7.1.4_文档内推词",
                "7.1.5_评价指标总结"
            ]
        },
        {
            "7.2_查询词推荐的召回": [
                "7.2.1_SUG_召回",
                "7.2.2_用查询词召回查询词（Q2Q）",
                "7.2.3_用文档召回查询词（D2Q）",
                "7.2.4_各推词场景的召回方法"
            ]
        },
        {
            "7.3_查询词推荐的排序": [
                "7.3.1_预估点击和转化",
                "7.3.2_多样性",
                "7.3.3_知识点小结"
            ]
        }
    ]
}


import os
from typing import Dict, Any

def create_directories_and_files(
        base_path: str, 
        structure: Dict[str, Any], 
        readme_file, 
        parent_path: str = "", 
        level: int = 1
    ):
    heading = "#" * level

    for key, value in structure.items():
        current_path = os.path.join(base_path, key.replace(" ", "_").replace("/", "_").replace("-", "_"))

        # 创建目录
        os.makedirs(current_path, exist_ok=True)

        # 在README中添加章节标题
        if parent_path:
            readme_file.write(f"{heading} {parent_path}/{key}\n\n")
        else:
            readme_file.write(f"{heading} {key}\n\n")

        # 递归调用创建子目录和文件
        if isinstance(value, dict) and value:
            create_directories_and_files(
                current_path, 
                value, 
                readme_file, 
                parent_path + "/" + key if parent_path else key, 
                level + 1
            )
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict) and item:
                    create_directories_and_files(
                        current_path, 
                        item, 
                        readme_file, 
                        parent_path + "/" + key if parent_path else key, 
                        level + 1
                    )
                else:
                    item = f"{idx:02d}_{item}"
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.py)\n")
                    
                    
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.md)\n")
        else:
            # 创建文件并写入初始内容
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")
            
            
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")

        # 添加空行以分隔不同的章节
        readme_file.write("\n")

def main():
    root_dir = './'
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)

    # 创建 README.md 文件
    with open(os.path.join(root_dir, "README.md"), 'w', encoding='utf-8') as readme_file:
        readme_file.write("# SearchEngine\n\n")
        readme_file.write("这是一个关于SearchEngine的目录结构。\n\n")
        create_directories_and_files(root_dir, structure, readme_file)

    print("目录和文件结构已生成，并创建 README.md 文件。")

if __name__ == "__main__":
    main()