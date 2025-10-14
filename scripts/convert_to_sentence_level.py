"""
將 Aspect-Level 資料集轉換為 Sentence-Level 資料集

目標：
- 將同一句子的多個 aspect 合併成單一句子級別的情感標籤
- 使用多數決 (majority voting) 決定句子的整體情感極性
- 處理票數相同的情況（標記為 conflict）
- 從訓練集生成 train/test 分割（80/20），不使用官方測試集

輸入：SemEval XML 格式（aspect-level）
輸出：CSV 格式（sentence-level）
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import csv
import sys
import random

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class SentenceLevelConverter:
    """Aspect-Level 到 Sentence-Level 轉換器"""

    # 情感極性優先順序（用於平局時）
    POLARITY_PRIORITY = {
        'positive': 3,
        'negative': 2,
        'neutral': 1,
        'conflict': 0
    }

    def __init__(self, xml_path: str):
        """
        初始化轉換器

        Args:
            xml_path: 輸入的 XML 檔案路徑
        """
        self.xml_path = Path(xml_path)
        self.sentences = []

    def parse_xml(self):
        """解析 XML 檔案並提取句子和 aspects"""
        if not self.xml_path.exists():
            raise FileNotFoundError(f"找不到檔案: {self.xml_path}")

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        for sentence in root.findall('.//sentence'):
            sent_id = sentence.get('id')
            text_elem = sentence.find('text')
            text = text_elem.text if text_elem is not None else ""

            # 提取所有 aspect 的情感極性
            aspect_polarities = []
            aspect_terms_elem = sentence.find('aspectTerms')

            if aspect_terms_elem is not None:
                for aspect_term in aspect_terms_elem.findall('aspectTerm'):
                    polarity = aspect_term.get('polarity')
                    term = aspect_term.get('term')
                    if polarity:
                        aspect_polarities.append({
                            'term': term,
                            'polarity': polarity
                        })

            # 儲存句子資訊
            self.sentences.append({
                'id': sent_id,
                'text': text,
                'aspects': aspect_polarities
            })

    def decide_sentence_polarity(self, aspect_polarities: List[str]) -> Tuple[str, Dict]:
        """
        使用多數決決定句子的整體情感極性

        Args:
            aspect_polarities: aspect 的情感極性列表

        Returns:
            (句子極性, 統計資訊)
        """
        if not aspect_polarities:
            return 'neutral', {'reason': 'no_aspects'}

        # 統計各極性的票數
        polarity_counts = Counter(aspect_polarities)

        # 獲取最高票數
        max_count = max(polarity_counts.values())

        # 找出所有具有最高票數的極性
        top_polarities = [p for p, count in polarity_counts.items() if count == max_count]

        stats = {
            'counts': dict(polarity_counts),
            'total_aspects': len(aspect_polarities),
            'top_count': max_count
        }

        # 情況 1: 只有一個極性獲得最高票數 -> 明確的多數決
        if len(top_polarities) == 1:
            stats['reason'] = 'majority'
            return top_polarities[0], stats

        # 情況 2: 多個極性票數相同 -> conflict
        # 但如果所有 aspect 都是同一極性，則不是 conflict
        if len(polarity_counts) == 1:
            stats['reason'] = 'unanimous'
            return top_polarities[0], stats

        # 有多個不同極性且票數相同 -> conflict
        stats['reason'] = 'tie'
        stats['tied_polarities'] = top_polarities

        # 使用優先順序來打破平局（可選）
        # 這裡我們選擇標記為 conflict 來保留資訊
        return 'conflict', stats

    def convert_to_sentence_level(self, filter_invalid: bool = True) -> List[Dict]:
        """
        將 aspect-level 資料轉換為 sentence-level

        Args:
            filter_invalid: 是否過濾掉無效樣本（無aspect或conflict標籤）

        Returns:
            sentence-level 資料列表
        """
        sentence_level_data = []
        filtered_count = {'no_aspects': 0, 'conflict': 0}

        for sent in self.sentences:
            # 提取所有 aspect 的極性
            aspect_polarities = [asp['polarity'] for asp in sent['aspects']]

            # 決定句子級別的極性
            sentence_polarity, stats = self.decide_sentence_polarity(aspect_polarities)

            # 過濾：移除無aspect的句子
            if filter_invalid and len(aspect_polarities) == 0:
                filtered_count['no_aspects'] += 1
                continue

            # 過濾：移除conflict標籤的句子
            if filter_invalid and sentence_polarity == 'conflict':
                filtered_count['conflict'] += 1
                continue

            # 建立 sentence-level 記錄
            record = {
                'sentence_id': sent['id'],
                'text': sent['text'],
                'sentence_polarity': sentence_polarity,
                'num_aspects': len(aspect_polarities),
                'aspect_polarities': ','.join(aspect_polarities) if aspect_polarities else '',
                'decision_reason': stats.get('reason', ''),
                'polarity_counts': str(stats.get('counts', {}))
            }

            sentence_level_data.append(record)

        if filter_invalid:
            print(f"  過濾統計:")
            print(f"    - 移除無aspect句子: {filtered_count['no_aspects']}")
            print(f"    - 移除conflict句子: {filtered_count['conflict']}")
            print(f"    - 保留有效句子: {len(sentence_level_data)}")

        return sentence_level_data

    def save_to_csv(self, output_path: str, data: List[Dict]):
        """
        將資料儲存為 CSV 格式

        Args:
            output_path: 輸出 CSV 檔案路徑
            data: sentence-level 資料
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not data:
            print("警告：沒有資料可以儲存")
            return

        fieldnames = [
            'sentence_id',
            'text',
            'sentence_polarity',
            'num_aspects',
            'aspect_polarities',
            'decision_reason',
            'polarity_counts'
        ]

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"[OK] 已儲存到: {output_path}")

    def print_statistics(self, data: List[Dict]):
        """列印轉換統計資訊"""
        print("\n" + "="*80)
        print("Sentence-Level 資料集統計")
        print("="*80)

        total_sentences = len(data)
        print(f"總句子數: {total_sentences}")

        # 統計情感分佈
        polarity_dist = Counter([d['sentence_polarity'] for d in data])
        print(f"\n句子級別情感分佈:")
        for polarity, count in sorted(polarity_dist.items()):
            percentage = (count / total_sentences * 100) if total_sentences > 0 else 0
            print(f"  {polarity:12s}: {count:5d} ({percentage:5.2f}%)")

        # 統計決策原因
        reason_dist = Counter([d['decision_reason'] for d in data])
        print(f"\n決策原因分佈:")
        reason_names = {
            'no_aspects': '無 aspects',
            'unanimous': '一致同意',
            'majority': '多數決',
            'tie': '平局 (conflict)'
        }
        for reason, count in sorted(reason_dist.items()):
            percentage = (count / total_sentences * 100) if total_sentences > 0 else 0
            reason_str = reason_names.get(reason, reason)
            print(f"  {reason_str:20s}: {count:5d} ({percentage:5.2f}%)")

        # 統計 aspect 數量分佈
        aspect_count_dist = Counter([d['num_aspects'] for d in data])
        print(f"\n每句 Aspect 數量分佈:")
        for count, freq in sorted(aspect_count_dist.items()):
            percentage = (freq / total_sentences * 100) if total_sentences > 0 else 0
            print(f"  {count} aspects: {freq:5d} ({percentage:5.2f}%)")

        avg_aspects = sum(d['num_aspects'] for d in data) / total_sentences if total_sentences > 0 else 0
        print(f"\n平均每句 Aspect 數: {avg_aspects:.2f}")
        print("="*80)


def convert_dataset(xml_path: str, output_csv_path: str):
    """
    轉換單個資料集

    Args:
        xml_path: 輸入 XML 檔案路徑
        output_csv_path: 輸出 CSV 檔案路徑
    """
    print(f"\n處理: {xml_path}")
    print("-"*80)

    converter = SentenceLevelConverter(xml_path)
    converter.parse_xml()

    sentence_level_data = converter.convert_to_sentence_level()
    converter.save_to_csv(output_csv_path, sentence_level_data)
    converter.print_statistics(sentence_level_data)


def split_train_test(data: List[Dict], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    將資料分割為訓練集和測試集

    Args:
        data: 資料列表
        test_ratio: 測試集比例
        seed: 隨機種子

    Returns:
        (train_data, test_data)
    """
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * (1 - test_ratio))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data


def convert_and_split_dataset(xml_path: str,
                              train_csv_path: str,
                              test_csv_path: str,
                              test_ratio: float = 0.2,
                              seed: int = 42):
    """
    轉換資料集並分割為 train/test

    Args:
        xml_path: 輸入 XML 檔案路徑
        train_csv_path: 訓練集輸出路徑
        test_csv_path: 測試集輸出路徑
        test_ratio: 測試集比例
        seed: 隨機種子
    """
    print(f"\n處理: {xml_path}")
    print("-"*80)

    # 轉換為 sentence-level
    converter = SentenceLevelConverter(xml_path)
    converter.parse_xml()
    sentence_level_data = converter.convert_to_sentence_level(filter_invalid=True)

    if not sentence_level_data:
        print("警告：沒有有效資料")
        return

    # 分割為 train/test
    train_data, test_data = split_train_test(sentence_level_data, test_ratio, seed)

    print(f"\n分割結果:")
    print(f"  總樣本數: {len(sentence_level_data)}")
    print(f"  訓練集: {len(train_data)} ({len(train_data)/len(sentence_level_data)*100:.1f}%)")
    print(f"  測試集: {len(test_data)} ({len(test_data)/len(sentence_level_data)*100:.1f}%)")

    # 儲存
    converter.save_to_csv(train_csv_path, train_data)
    converter.save_to_csv(test_csv_path, test_data)

    # 列印統計
    print("\n訓練集統計:")
    converter.print_statistics(train_data)

    print("\n測試集統計:")
    converter.print_statistics(test_data)


def main():
    """主函數：轉換所有 SemEval-2014 資料集"""
    print("\n" + "="*80)
    print("Aspect-Level → Sentence-Level 資料集轉換工具 (Train/Test Split)")
    print("="*80)

    # 定義資料集路徑
    base_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = base_dir / 'data' / 'raw' / 'SemEval-2014'
    output_dir = base_dir / 'data' / 'processed' / 'sentence_level'

    datasets = {
        'Restaurant': {
            'xml': raw_data_dir / 'Restaurants_Train_v2.xml',
            'train_csv': output_dir / 'SemEval2014_Restaurant_train_sentence_level.csv',
            'test_csv': output_dir / 'SemEval2014_Restaurant_test_sentence_level.csv'
        },
        'Laptop': {
            'xml': raw_data_dir / 'Laptop_Train_v2.xml',
            'train_csv': output_dir / 'SemEval2014_Laptop_train_sentence_level.csv',
            'test_csv': output_dir / 'SemEval2014_Laptop_test_sentence_level.csv'
        }
    }

    # 轉換每個資料集
    for name, paths in datasets.items():
        if paths['xml'].exists():
            convert_and_split_dataset(
                xml_path=str(paths['xml']),
                train_csv_path=str(paths['train_csv']),
                test_csv_path=str(paths['test_csv']),
                test_ratio=0.2,
                seed=42
            )
        else:
            print(f"\n警告：找不到檔案 {paths['xml']}")

    print("\n" + "="*80)
    print("[OK] 所有資料集轉換完成！")
    print(f"[OK] 輸出目錄: {output_dir}")
    print(f"[OK] 分割比例: 80% train / 20% test")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
