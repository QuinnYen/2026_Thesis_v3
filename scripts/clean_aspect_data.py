"""
Task 1.2: 資料清理策略實作
清理 aspect-level 資料集中的情感衝突句子
"""

import xml.etree.ElementTree as ET
import csv
from collections import Counter, defaultdict
from pathlib import Path

# 設定路徑
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "SemEval-2014"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "aspect_level_cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DataCleaner:
    """資料清理器"""

    def __init__(self, cleaning_rule='A'):
        """
        初始化清理器

        Args:
            cleaning_rule: 'A' = 移除衝突句子（預設）, 'B' = 保留多數極性
        """
        self.cleaning_rule = cleaning_rule
        self.stats = {
            'original_sentences': 0,
            'original_aspects': 0,
            'removed_sentences': 0,
            'removed_aspects': 0,
            'kept_sentences': 0,
            'kept_aspects': 0,
            'removed_examples': [],
            'original_class_dist': Counter(),
            'cleaned_class_dist': Counter()
        }

    def parse_xml(self, file_path):
        """解析 SemEval XML 檔案（支援 2014 和 2016 格式）"""
        tree = ET.parse(file_path)
        root = tree.getroot()

        sentences = []

        # 檢測 XML 格式
        # 2014 格式: <sentences><sentence><aspectTerms>
        # 2016 格式: <Reviews><Review><sentences><sentence><Opinions>

        # 嘗試 2014 格式
        sentence_elements = root.findall('sentence')
        if not sentence_elements:
            # 嘗試 2016 格式
            sentence_elements = root.findall('.//sentence')

        for sentence in sentence_elements:
            sent_id = sentence.get('id')
            text_elem = sentence.find('text')
            if text_elem is None or text_elem.text is None:
                continue
            text = text_elem.text

            aspects = []

            # 嘗試解析 2014 格式 (aspectTerms)
            aspect_terms = sentence.find('aspectTerms')
            if aspect_terms is not None:
                for aspect_term in aspect_terms.findall('aspectTerm'):
                    polarity = aspect_term.get('polarity')
                    # 跳過沒有 polarity 的 aspect（測試集）
                    if polarity is None:
                        continue
                    aspects.append({
                        'term': aspect_term.get('term'),
                        'polarity': polarity,
                        'from': int(aspect_term.get('from')),
                        'to': int(aspect_term.get('to'))
                    })

            # 嘗試解析 2016 格式 (Opinions)
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    polarity = opinion.get('polarity')
                    target = opinion.get('target')
                    # 跳過沒有 polarity 的 opinion 或 target 為 NULL
                    if polarity is None or target == 'NULL':
                        continue
                    from_pos = opinion.get('from', '0')
                    to_pos = opinion.get('to', '0')
                    aspects.append({
                        'term': target,
                        'polarity': polarity,
                        'from': int(from_pos),
                        'to': int(to_pos)
                    })

            if aspects:  # 只包含有 aspect terms 的句子
                sentences.append({
                    'id': sent_id,
                    'text': text,
                    'aspects': aspects
                })

        return sentences

    def has_conflict(self, aspects):
        """
        判斷 aspects 是否有衝突

        衝突定義：同時包含 positive 和 negative
        不視為衝突：positive/negative + neutral
        """
        polarities = set(a['polarity'] for a in aspects)
        return 'positive' in polarities and 'negative' in polarities

    def clean_rule_a(self, sentences):
        """
        規則 A：移除明顯衝突的句子

        - 先移除 aspect 極性為 'conflict' 的項目
        - 移除同時有 positive 和 negative aspects 的整個句子
        - 保留情感一致或僅有 neutral 差異的句子
        """
        cleaned_sentences = []

        for sent in sentences:
            self.stats['original_sentences'] += 1
            self.stats['original_aspects'] += len(sent['aspects'])

            # 統計原始類別分佈
            for aspect in sent['aspects']:
                self.stats['original_class_dist'][aspect['polarity']] += 1

            # 步驟 1：先過濾掉 aspect 極性為 'conflict' 的項目
            filtered_aspects = [a for a in sent['aspects'] if a['polarity'] != 'conflict']

            # 如果過濾後沒有 aspects，移除整個句子
            if len(filtered_aspects) == 0:
                self.stats['removed_sentences'] += 1
                self.stats['removed_aspects'] += len(sent['aspects'])
                if len(self.stats['removed_examples']) < 20:
                    self.stats['removed_examples'].append({
                        'id': sent['id'],
                        'text': sent['text'],
                        'aspects': sent['aspects'],
                        'reason': '所有 aspects 極性為 conflict'
                    })
                continue

            # 步驟 2：檢查過濾後的 aspects 是否有衝突（positive 和 negative 同時存在）
            if self.has_conflict(filtered_aspects):
                # 移除整個句子
                self.stats['removed_sentences'] += 1
                self.stats['removed_aspects'] += len(sent['aspects'])

                # 記錄前 20 個移除案例
                if len(self.stats['removed_examples']) < 20:
                    self.stats['removed_examples'].append({
                        'id': sent['id'],
                        'text': sent['text'],
                        'aspects': sent['aspects'],
                        'reason': '同時包含 positive 和 negative aspects'
                    })
            else:
                # 保留句子（使用過濾後的 aspects）
                cleaned_sent = sent.copy()
                cleaned_sent['aspects'] = filtered_aspects
                cleaned_sentences.append(cleaned_sent)
                self.stats['kept_sentences'] += 1
                self.stats['kept_aspects'] += len(filtered_aspects)

                # 統計清理後類別分佈（使用過濾後的 aspects）
                for aspect in filtered_aspects:
                    self.stats['cleaned_class_dist'][aspect['polarity']] += 1

        return cleaned_sentences

    def clean_rule_b(self, sentences):
        """
        規則 B：保留多數極性

        - 保留句子中佔多數的 aspect 極性
        - 移除少數極性的 aspects
        - 如果票數相同，則移除整句
        """
        cleaned_sentences = []

        for sent in sentences:
            self.stats['original_sentences'] += 1
            self.stats['original_aspects'] += len(sent['aspects'])

            # 統計原始類別分佈
            for aspect in sent['aspects']:
                self.stats['original_class_dist'][aspect['polarity']] += 1

            # 只有一個 aspect，直接保留
            if len(sent['aspects']) == 1:
                cleaned_sentences.append(sent)
                self.stats['kept_sentences'] += 1
                self.stats['kept_aspects'] += 1
                self.stats['cleaned_class_dist'][sent['aspects'][0]['polarity']] += 1
                continue

            # 檢查是否有衝突
            if not self.has_conflict(sent['aspects']):
                # 無衝突，直接保留
                cleaned_sentences.append(sent)
                self.stats['kept_sentences'] += 1
                self.stats['kept_aspects'] += len(sent['aspects'])
                for aspect in sent['aspects']:
                    self.stats['cleaned_class_dist'][aspect['polarity']] += 1
                continue

            # 有衝突，計算多數極性（不包含 neutral）
            polarity_counts = Counter()
            for aspect in sent['aspects']:
                if aspect['polarity'] in ['positive', 'negative']:
                    polarity_counts[aspect['polarity']] += 1

            if not polarity_counts:
                # 全部是 neutral，保留
                cleaned_sentences.append(sent)
                self.stats['kept_sentences'] += 1
                self.stats['kept_aspects'] += len(sent['aspects'])
                for aspect in sent['aspects']:
                    self.stats['cleaned_class_dist'][aspect['polarity']] += 1
                continue

            # 找出多數極性
            most_common = polarity_counts.most_common(2)
            if len(most_common) == 2 and most_common[0][1] == most_common[1][1]:
                # 票數相同，移除整句
                self.stats['removed_sentences'] += 1
                self.stats['removed_aspects'] += len(sent['aspects'])

                if len(self.stats['removed_examples']) < 20:
                    self.stats['removed_examples'].append({
                        'id': sent['id'],
                        'text': sent['text'],
                        'aspects': sent['aspects'],
                        'reason': f'positive 和 negative 票數相同（各 {most_common[0][1]} 票）'
                    })
            else:
                # 保留多數極性和 neutral
                majority_polarity = most_common[0][0]
                kept_aspects = [a for a in sent['aspects']
                              if a['polarity'] == majority_polarity or a['polarity'] == 'neutral']
                removed_count = len(sent['aspects']) - len(kept_aspects)

                cleaned_sentences.append({
                    'id': sent['id'],
                    'text': sent['text'],
                    'aspects': kept_aspects
                })

                self.stats['kept_sentences'] += 1
                self.stats['kept_aspects'] += len(kept_aspects)
                self.stats['removed_aspects'] += removed_count

                for aspect in kept_aspects:
                    self.stats['cleaned_class_dist'][aspect['polarity']] += 1

                if len(self.stats['removed_examples']) < 20:
                    removed_aspects = [a for a in sent['aspects'] if a not in kept_aspects]
                    self.stats['removed_examples'].append({
                        'id': sent['id'],
                        'text': sent['text'],
                        'aspects': sent['aspects'],
                        'kept_aspects': kept_aspects,
                        'removed_aspects': removed_aspects,
                        'reason': f'保留多數極性 {majority_polarity}，移除少數極性'
                    })

        return cleaned_sentences

    def clean(self, sentences):
        """根據規則清理資料"""
        if self.cleaning_rule == 'A':
            return self.clean_rule_a(sentences)
        elif self.cleaning_rule == 'B':
            return self.clean_rule_b(sentences)
        else:
            raise ValueError(f"未知的清理規則: {self.cleaning_rule}")

    def save_to_csv(self, sentences, output_path):
        """將清理後的資料儲存為 CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'aspect_id', 'sentence_id', 'sentence_text',
                'aspect_term', 'aspect_polarity', 'from_position', 'to_position'
            ])

            aspect_id = 1
            for sent in sentences:
                for aspect in sent['aspects']:
                    writer.writerow([
                        aspect_id,
                        sent['id'],
                        sent['text'],
                        aspect['term'],
                        aspect['polarity'],
                        aspect['from'],
                        aspect['to']
                    ])
                    aspect_id += 1

    def get_stats(self):
        """取得清理統計"""
        return self.stats

    def reset_stats(self):
        """重設統計資料"""
        self.stats = {
            'original_sentences': 0,
            'original_aspects': 0,
            'removed_sentences': 0,
            'removed_aspects': 0,
            'kept_sentences': 0,
            'kept_aspects': 0,
            'removed_examples': [],
            'original_class_dist': Counter(),
            'cleaned_class_dist': Counter()
        }


def generate_cleaning_log(all_stats, cleaning_rule, output_path):
    """生成清理日誌"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"資料清理日誌 - 清理規則：{cleaning_rule}")
    lines.append("=" * 80)
    lines.append("")

    if cleaning_rule == 'A':
        lines.append("清理策略：移除明顯衝突的句子")
        lines.append("定義：移除同時包含 positive 和 negative aspects 的整個句子")
    elif cleaning_rule == 'B':
        lines.append("清理策略：保留多數極性")
        lines.append("定義：保留佔多數的極性，移除少數極性；票數相同則移除整句")
    lines.append("")
    lines.append("")

    # 彙總所有資料集的統計
    for dataset_name, stats in all_stats.items():
        lines.append(f"{'=' * 80}")
        lines.append(f"{dataset_name}")
        lines.append(f"{'=' * 80}")
        lines.append("")

        # 1. 資料量統計
        lines.append("【1. 資料量統計】")
        lines.append("-" * 80)
        lines.append(f"原始句子數量: {stats['original_sentences']}")
        lines.append(f"保留句子數量: {stats['kept_sentences']}")
        lines.append(f"移除句子數量: {stats['removed_sentences']}")
        if stats['original_sentences'] > 0:
            lines.append(f"句子保留率: {stats['kept_sentences']/stats['original_sentences']*100:.2f}%")
        else:
            lines.append(f"句子保留率: N/A (無數據)")
        if stats['original_sentences'] > 0:
            lines.append(f"句子移除率: {stats['removed_sentences']/stats['original_sentences']*100:.2f}%")
        else:
            lines.append(f"句子移除率: N/A (無數據)")
        lines.append("")

        lines.append(f"原始 aspect 數量: {stats['original_aspects']}")
        lines.append(f"保留 aspect 數量: {stats['kept_aspects']}")
        lines.append(f"移除 aspect 數量: {stats['removed_aspects']}")
        if stats['original_aspects'] > 0:
            lines.append(f"Aspect 保留率: {stats['kept_aspects']/stats['original_aspects']*100:.2f}%")
            lines.append(f"Aspect 移除率: {stats['removed_aspects']/stats['original_aspects']*100:.2f}%")
        else:
            lines.append(f"Aspect 保留率: N/A (無數據)")
            lines.append(f"Aspect 移除率: N/A (無數據)")
        lines.append("")
        lines.append("")

        # 2. 類別分佈對比
        lines.append("【2. 清理前後類別分佈對比】")
        lines.append("-" * 80)
        lines.append("清理前：")
        total_orig = sum(stats['original_class_dist'].values())
        if total_orig > 0:
            for polarity, count in sorted(stats['original_class_dist'].items()):
                polarity_str = str(polarity) if polarity else 'unknown'
                lines.append(f"  {polarity_str:10s}: {count:5d} ({count/total_orig*100:5.2f}%)")
        else:
            lines.append("  （無資料）")

        lines.append("")
        lines.append("清理後：")
        total_clean = sum(stats['cleaned_class_dist'].values())
        if total_clean > 0:
            for polarity, count in sorted(stats['cleaned_class_dist'].items()):
                polarity_str = str(polarity) if polarity else 'unknown'
                orig_count = stats['original_class_dist'].get(polarity, 0)
                change = count - orig_count
                lines.append(f"  {polarity_str:10s}: {count:5d} ({count/total_clean*100:5.2f}%) [變化: {change:+d}]")
        else:
            lines.append("  （無資料）")

        lines.append("")
        lines.append("")

        # 3. 移除案例
        lines.append("【3. 移除案例（前 20 個）】")
        lines.append("-" * 80)
        for i, example in enumerate(stats['removed_examples'], 1):
            lines.append(f"\n案例 {i}:")
            lines.append(f"  句子 ID: {example['id']}")
            lines.append(f"  文本: {example['text']}")
            lines.append(f"  Aspects:")
            for aspect in example['aspects']:
                lines.append(f"    - '{aspect['term']}' → {aspect['polarity']}")
            lines.append(f"  移除原因: {example['reason']}")

            # 如果是規則 B，顯示保留和移除的 aspects
            if 'kept_aspects' in example:
                lines.append(f"  保留的 aspects:")
                for aspect in example['kept_aspects']:
                    lines.append(f"    - '{aspect['term']}' → {aspect['polarity']}")
                lines.append(f"  移除的 aspects:")
                for aspect in example['removed_aspects']:
                    lines.append(f"    - '{aspect['term']}' → {aspect['polarity']}")

        lines.append("")
        lines.append("")

    # 寫入檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    print("=" * 80)
    print("Task 1.2: 資料清理策略實作")
    print("=" * 80)
    print()

    # 定義資料集（僅處理訓練集，測試集無標籤）
    datasets = {
        # SemEval 2014 數據集
        'SemEval2014 Restaurants Train': {
            'input': DATA_DIR / 'Restaurants_Train_v2.xml',
            'output': OUTPUT_DIR / 'SemEval2014_Restaurant_train_cleaned.csv'
        },
        'SemEval2014 Laptops Train': {
            'input': DATA_DIR / 'Laptop_Train_v2.xml',
            'output': OUTPUT_DIR / 'SemEval2014_Laptop_train_cleaned.csv'
        },
        # SemEval 2016 數據集
        'SemEval2016 Restaurants Train': {
            'input': BASE_DIR / 'data' / 'raw' / 'SemEval-2016' / 'restaurants_train_sb1.xml',
            'output': OUTPUT_DIR / 'SemEval2016_Restaurant_train_cleaned.csv'
        },
        'SemEval2016 Laptops Train': {
            'input': BASE_DIR / 'data' / 'raw' / 'SemEval-2016' / 'Laptops_Train_sb1.xml',
            'output': OUTPUT_DIR / 'SemEval2016_Laptop_train_cleaned.csv'
        }
    }

    # 使用規則 A（預設）
    cleaning_rule = 'A'
    print(f"使用清理規則: {cleaning_rule}")
    print("規則說明: 移除同時包含 positive 和 negative aspects 的句子")
    print("注意: 測試集無標籤，僅處理訓練集")
    print()

    # 處理所有資料集
    all_stats = {}
    cleaner = DataCleaner(cleaning_rule=cleaning_rule)

    for dataset_name, paths in datasets.items():
        print(f"\n處理 {dataset_name}...")
        print("-" * 80)

        # 重設統計
        cleaner.reset_stats()

        # 解析資料
        sentences = cleaner.parse_xml(paths['input'])
        print(f"原始句子數: {len(sentences)}")

        # 清理資料
        cleaned_sentences = cleaner.clean(sentences)
        print(f"清理後句子數: {len(cleaned_sentences)}")

        # 儲存 CSV
        cleaner.save_to_csv(cleaned_sentences, paths['output'])
        print(f"[完成] 已儲存至: {paths['output']}")

        # 保存統計資料
        all_stats[dataset_name] = cleaner.get_stats()

    # 生成清理日誌
    log_path = OUTPUT_DIR / f'cleaning_log_rule_{cleaning_rule}.txt'
    generate_cleaning_log(all_stats, cleaning_rule, log_path)
    print(f"\n[完成] 清理日誌已儲存: {log_path}")

    # 輸出彙總統計
    print()
    print("=" * 80)
    print("清理結果彙總")
    print("=" * 80)

    total_orig_sent = sum(s['original_sentences'] for s in all_stats.values())
    total_kept_sent = sum(s['kept_sentences'] for s in all_stats.values())
    total_removed_sent = sum(s['removed_sentences'] for s in all_stats.values())

    print(f"總句子數（原始）: {total_orig_sent}")
    print(f"總句子數（保留）: {total_kept_sent} ({total_kept_sent/total_orig_sent*100:.2f}%)")
    print(f"總句子數（移除）: {total_removed_sent} ({total_removed_sent/total_orig_sent*100:.2f}%)")
    print()

    total_orig_asp = sum(s['original_aspects'] for s in all_stats.values())
    total_kept_asp = sum(s['kept_aspects'] for s in all_stats.values())
    total_removed_asp = sum(s['removed_aspects'] for s in all_stats.values())

    print(f"總 aspect 數（原始）: {total_orig_asp}")
    print(f"總 aspect 數（保留）: {total_kept_asp} ({total_kept_asp/total_orig_asp*100:.2f}%)")
    print(f"總 aspect 數（移除）: {total_removed_asp} ({total_removed_asp/total_orig_asp*100:.2f}%)")

    print()
    print("=" * 80)
    print("[成功] 資料清理完成！")
    print(f"所有檔案已儲存至: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
