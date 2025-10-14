"""
SemEval 資料讀取器
支援 SemEval-2014 和 SemEval-2016 資料集的統一載入介面
"""

import xml.etree.ElementTree as ET
import os
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict


class SemEvalDataLoader:
    """
    SemEval 資料讀取器
    支援載入 SemEval-2014 和 SemEval-2016 的 Laptop 和 Restaurant 資料集
    """

    # 資料集配置
    DATASET_CONFIG = {
        'SemEval2014_Laptop': {
            'year': 2014,
            'domain': 'laptop',
            'supported': True,  # 支援面向詞級別標註
            'files': {
                'train': 'SemEval-2014/Laptop_Train_v2.xml',
                'test': 'SemEval-2014/Laptops_Test_Data_phaseB.xml'
            }
        },
        'SemEval2014_Restaurant': {
            'year': 2014,
            'domain': 'restaurant',
            'supported': True,  # 支援面向詞級別標註
            'files': {
                'train': 'SemEval-2014/Restaurants_Train_v2.xml',
                'test': 'SemEval-2014/Restaurants_Test_Data_phaseB.xml'
            }
        },
        'SemEval2016_Laptop': {
            'year': 2016,
            'domain': 'laptop',
            'supported': False,  # 不支援（使用類別級別標註）
            'files': {
                'train': 'SemEval-2016/Laptops_Train_sb1.xml',
                'test': 'SemEval-2016/laptops_test_sb1.xml'
            }
        },
        'SemEval2016_Restaurant': {
            'year': 2016,
            'domain': 'restaurant',
            'supported': False,  # 不支援（使用類別級別標註）
            'files': {
                'train': 'SemEval-2016/restaurants_train_sb1.xml',
                'test': 'SemEval-2016/restaurants_test_sb1.xml'
            }
        }
    }

    # 支援的情感極性
    VALID_POLARITIES = {'positive', 'negative', 'neutral', 'conflict'}

    def __init__(self, data_dir: str = 'data/raw'):
        """
        初始化資料讀取器

        Args:
            data_dir: 資料目錄路徑
        """
        self.data_dir = data_dir
        self._validate_data_dir()

    def _validate_data_dir(self) -> None:
        """驗證資料目錄是否存在"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"資料目錄不存在: {self.data_dir}\n"
                f"請確認資料已下載到正確位置"
            )

    def _get_file_path(self, dataset_name: str, split: str) -> str:
        """
        獲取資料檔案路徑

        Args:
            dataset_name: 資料集名稱
            split: 資料分割（train/test）

        Returns:
            完整的檔案路徑
        """
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(
                f"不支援的資料集: {dataset_name}\n"
                f"支援的資料集: {list(self.DATASET_CONFIG.keys())}"
            )

        config = self.DATASET_CONFIG[dataset_name]
        if split not in config['files']:
            raise ValueError(
                f"不支援的資料分割: {split}\n"
                f"支援的分割: {list(config['files'].keys())}"
            )

        file_path = os.path.join(self.data_dir, config['files'][split])

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"找不到資料檔案: {file_path}\n"
                f"請確認檔案是否存在"
            )

        return file_path

    def _parse_xml(self, xml_path: str, include_unlabeled: bool = False) -> List[Dict]:
        """
        解析 XML 檔案

        Args:
            xml_path: XML 檔案路徑
            include_unlabeled: 是否包含無標籤資料（測試集）

        Returns:
            解析後的資料列表
        """
        data = []
        unlabeled_count = 0

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 遍歷所有句子
            for sentence in root.findall('.//sentence'):
                sent_id = sentence.get('id')
                text_elem = sentence.find('text')

                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text

                # 解析面向詞
                aspect_terms = sentence.find('aspectTerms')

                # 如果沒有面向詞，跳過（只保留有標註的句子）
                if aspect_terms is None:
                    continue

                # 提取所有面向詞
                for aspect_term in aspect_terms.findall('aspectTerm'):
                    aspect = aspect_term.get('term')
                    polarity = aspect_term.get('polarity')
                    from_idx = aspect_term.get('from')
                    to_idx = aspect_term.get('to')

                    # 資料驗證
                    if not aspect:
                        continue

                    # 如果沒有極性標籤（測試集）
                    if not polarity:
                        unlabeled_count += 1
                        if not include_unlabeled:
                            continue
                        polarity = 'unknown'  # 標記為未知
                    else:
                        # 極性標準化（轉小寫）
                        polarity = polarity.lower()

                        # 驗證極性是否有效
                        if polarity not in self.VALID_POLARITIES:
                            print(f"警告: 未知的極性標籤 '{polarity}' 在句子 {sent_id}")
                            continue

                    # 解析位置索引
                    try:
                        from_idx = int(from_idx) if from_idx else 0
                        to_idx = int(to_idx) if to_idx else len(aspect)
                    except (ValueError, TypeError):
                        from_idx = 0
                        to_idx = len(aspect)

                    # 驗證面向詞位置
                    if from_idx >= 0 and to_idx <= len(text):
                        extracted_aspect = text[from_idx:to_idx]
                        # 檢查提取的文本是否與標註一致
                        if extracted_aspect != aspect:
                            print(f"警告: 面向詞不一致 - 標註:'{aspect}' vs 提取:'{extracted_aspect}'")

                    # 建立資料記錄
                    data_item = {
                        'text': text,
                        'aspect': aspect,
                        'polarity': polarity,
                        'aspect_position': [from_idx, to_idx],
                        'sentence_id': sent_id  # 額外資訊，用於追蹤
                    }
                    data.append(data_item)

            if unlabeled_count > 0:
                print(f"注意: 發現 {unlabeled_count} 個無標籤樣本（測試集）")

        except ET.ParseError as e:
            raise RuntimeError(f"XML 解析錯誤: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"載入資料時發生錯誤: {str(e)}")

        return data

    def load(self, dataset_name: str, split: str = 'train', include_unlabeled: bool = False) -> List[Dict]:
        """
        載入資料集

        Args:
            dataset_name: 資料集名稱
                - SemEval2014_Laptop
                - SemEval2014_Restaurant
                - SemEval2016_Laptop
                - SemEval2016_Restaurant
            split: 資料分割（train/test）
            include_unlabeled: 是否包含無標籤資料（預設為 False，用於訓練）

        Returns:
            資料列表，每個元素包含：
                - text: 句子文本
                - aspect: 面向詞
                - polarity: 情感極性（測試集可能為 'unknown'）
                - aspect_position: [start_idx, end_idx]
                - sentence_id: 句子ID（額外資訊）
        """
        # 獲取檔案路徑
        file_path = self._get_file_path(dataset_name, split)

        # 解析 XML
        print(f"載入資料: {dataset_name} ({split})")
        print(f"檔案路徑: {file_path}")

        data = self._parse_xml(file_path, include_unlabeled=include_unlabeled)

        print(f"載入完成: {len(data)} 個樣本\n")

        return data

    def get_statistics(self, data: List[Dict]) -> Dict:
        """
        計算資料集統計資訊

        Args:
            data: 資料列表

        Returns:
            統計資訊字典
        """
        if not data:
            return {
                'total_samples': 0,
                'unique_sentences': 0,
                'polarity_distribution': {},
                'aspect_stats': {}
            }

        # 統計情感分佈
        polarity_counter = Counter([item['polarity'] for item in data])

        # 統計唯一句子數
        unique_sentences = len(set(item['text'] for item in data))

        # 統計面向詞
        aspect_counter = Counter([item['aspect'] for item in data])
        unique_aspects = len(aspect_counter)
        avg_aspect_length = sum(len(item['aspect']) for item in data) / len(data)

        # 統計句子長度
        sentence_lengths = [len(item['text'].split()) for item in data]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

        stats = {
            'total_samples': len(data),
            'unique_sentences': unique_sentences,
            'unique_aspects': unique_aspects,
            'avg_aspects_per_sentence': len(data) / unique_sentences,
            'avg_aspect_length': avg_aspect_length,
            'avg_sentence_length': avg_sentence_length,
            'polarity_distribution': dict(polarity_counter),
            'top_aspects': aspect_counter.most_common(10)
        }

        return stats

    def print_statistics(self, data: List[Dict], dataset_name: str = "Dataset") -> None:
        """
        列印資料集統計資訊

        Args:
            data: 資料列表
            dataset_name: 資料集名稱（用於顯示）
        """
        stats = self.get_statistics(data)

        print("=" * 70)
        print(f" {dataset_name} 統計資訊")
        print("=" * 70)
        print(f"總樣本數: {stats['total_samples']:,}")
        print(f"唯一句子數: {stats['unique_sentences']:,}")
        print(f"唯一面向詞數: {stats['unique_aspects']:,}")
        print(f"平均每句面向詞數: {stats['avg_aspects_per_sentence']:.2f}")
        print(f"平均面向詞長度: {stats['avg_aspect_length']:.2f} 字元")
        print(f"平均句子長度: {stats['avg_sentence_length']:.2f} 詞")

        print(f"\n情感極性分佈:")
        total = stats['total_samples']
        for polarity, count in sorted(stats['polarity_distribution'].items()):
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {polarity:12s}: {count:5d} ({percentage:5.2f}%) {bar}")

        print(f"\n最常見的面向詞 (Top 10):")
        for aspect, count in stats['top_aspects']:
            print(f"  {aspect:20s}: {count:4d}")

        print("=" * 70)
        print()

    def print_samples(self, data: List[Dict], n: int = 5) -> None:
        """
        列印資料樣本

        Args:
            data: 資料列表
            n: 要列印的樣本數
        """
        print(f"\n資料樣本 (顯示前 {n} 筆):")
        print("-" * 70)

        for i, item in enumerate(data[:n], 1):
            print(f"\n[樣本 {i}]")
            print(f"句子: {item['text']}")
            print(f"面向詞: '{item['aspect']}'")
            print(f"情感極性: {item['polarity']}")
            print(f"位置: {item['aspect_position']}")

            # 顯示面向詞在句子中的標記
            text = item['text']
            start, end = item['aspect_position']
            marked_text = text[:start] + f"[{text[start:end]}]" + text[end:]
            print(f"標記: {marked_text}")

        print("-" * 70)

    def load_all_datasets(self, only_supported: bool = True) -> Dict[str, Dict[str, List[Dict]]]:
        """
        載入所有支援的資料集

        Args:
            only_supported: 是否只載入支援的資料集（預設 True，跳過 SemEval-2016）

        Returns:
            巢狀字典: {dataset_name: {split: data}}
        """
        all_data = {}

        print("=" * 70)
        print(" 載入 SemEval 資料集")
        print("=" * 70)
        print()

        for dataset_name, config in self.DATASET_CONFIG.items():
            # 跳過不支援的資料集
            if only_supported and not config.get('supported', False):
                print(f"跳過 {dataset_name} (不支援的格式)\n")
                continue

            all_data[dataset_name] = {}

            for split in ['train', 'test']:
                try:
                    # 測試集包含無標籤資料
                    include_unlabeled = (split == 'test')
                    data = self.load(dataset_name, split, include_unlabeled=include_unlabeled)
                    all_data[dataset_name][split] = data
                except Exception as e:
                    print(f"警告: 無法載入 {dataset_name} ({split}): {str(e)}\n")
                    all_data[dataset_name][split] = []

        return all_data

    def get_available_datasets(self) -> List[str]:
        """
        獲取所有資料集列表（包含不支援的）

        Returns:
            資料集名稱列表
        """
        return list(self.DATASET_CONFIG.keys())

    def get_supported_datasets(self) -> List[str]:
        """
        獲取支援的資料集列表（僅 SemEval-2014）

        Returns:
            支援的資料集名稱列表
        """
        return [
            name for name, config in self.DATASET_CONFIG.items()
            if config.get('supported', False)
        ]

    def is_dataset_supported(self, dataset_name: str) -> bool:
        """
        檢查資料集是否支援

        Args:
            dataset_name: 資料集名稱

        Returns:
            是否支援
        """
        if dataset_name not in self.DATASET_CONFIG:
            return False
        return self.DATASET_CONFIG[dataset_name].get('supported', False)


def validate_data_format(data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    驗證資料格式是否正確

    Args:
        data: 資料列表

    Returns:
        (是否有效, 錯誤訊息列表)
    """
    errors = []
    required_keys = {'text', 'aspect', 'polarity', 'aspect_position'}

    for i, item in enumerate(data):
        # 檢查必要欄位
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            errors.append(f"樣本 {i}: 缺少欄位 {missing_keys}")
            continue

        # 檢查資料型態
        if not isinstance(item['text'], str):
            errors.append(f"樣本 {i}: text 必須是字串")

        if not isinstance(item['aspect'], str):
            errors.append(f"樣本 {i}: aspect 必須是字串")

        if not isinstance(item['polarity'], str):
            errors.append(f"樣本 {i}: polarity 必須是字串")

        if not isinstance(item['aspect_position'], list) or len(item['aspect_position']) != 2:
            errors.append(f"樣本 {i}: aspect_position 必須是長度為 2 的列表")

    return len(errors) == 0, errors


if __name__ == "__main__":
    # 測試程式碼
    print("=" * 70)
    print(" SemEval 資料讀取器測試")
    print("=" * 70)
    print()

    # 建立資料讀取器
    loader = SemEvalDataLoader()

    # 顯示可用資料集
    print("可用的資料集:")
    for dataset in loader.get_available_datasets():
        print(f"  - {dataset}")
    print()

    # 測試載入單一資料集
    print("\n" + "=" * 70)
    print(" 測試載入 SemEval2014_Laptop 訓練集")
    print("=" * 70)

    data = loader.load('SemEval2014_Laptop', 'train')

    # 驗證資料格式
    is_valid, errors = validate_data_format(data)
    print(f"\n資料格式驗證: {'通過' if is_valid else '失敗'}")
    if errors:
        print("錯誤訊息:")
        for error in errors[:5]:  # 只顯示前 5 個錯誤
            print(f"  - {error}")

    # 列印統計資訊
    print()
    loader.print_statistics(data, "SemEval2014_Laptop (train)")

    # 列印範例
    loader.print_samples(data, n=5)

    # 測試載入所有資料集
    print("\n" + "=" * 70)
    print(" 測試載入所有資料集")
    print("=" * 70)
    print()

    all_data = loader.load_all_datasets()

    # 顯示所有資料集的摘要
    print("\n" + "=" * 70)
    print(" 所有資料集摘要")
    print("=" * 70)

    total_samples = 0
    for dataset_name, splits in all_data.items():
        for split, data in splits.items():
            if data:
                print(f"{dataset_name:30s} ({split:5s}): {len(data):5d} 樣本")
                total_samples += len(data)

    print(f"\n總計: {total_samples:,} 個樣本")
