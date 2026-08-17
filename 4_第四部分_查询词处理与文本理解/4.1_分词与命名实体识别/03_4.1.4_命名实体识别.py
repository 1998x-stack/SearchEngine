# 03_4.1.4_命名实体识别

"""
Lecture: 4_第四部分_查询词处理与文本理解/4.1_分词与命名实体识别
Content: 03_4.1.4_命名实体识别
"""

from typing import Dict, List, Tuple


class DictionaryNER:
    """Rule/dictionary-based named-entity recognizer (PER/ORG/LOC)."""

    def __init__(self) -> None:
        self.entities: Dict[str, List[str]] = {
            "PER": ["张三", "李四", "王五"],
            "ORG": ["百度", "华为", "阿里"],
            "LOC": ["北京", "上海", "杭州"],
        }

    def recognize(self, text: str) -> List[Tuple[str, str]]:
        """Return a list of (entity, type) found in text."""
        found: List[Tuple[str, str]] = []
        for etype, names in self.entities.items():
            for name in names:
                if name in text:
                    found.append((name, etype))
        return found


def main() -> None:
    print("命名实体识别 (NER) Demo")
    ner = DictionaryNER()
    text = "李四在华为工作, 经常出差到北京"
    print(f"文本: {text}")
    print("识别出的命名实体:")
    for ent, etype in ner.recognize(text):
        print(f"  {ent} -> {etype}")


if __name__ == "__main__":
    main()