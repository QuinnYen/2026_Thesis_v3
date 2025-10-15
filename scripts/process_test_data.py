"""
處理測試集數據

將 SemEval 測試集轉換為 CSV 格式（不進行衝突移除）
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "aspect_level_cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_xml_test(file_path, format_type='2014'):
    """
    解析測試集 XML 文件

    Args:
        file_path: XML 文件路徑
        format_type: '2014' 或 '2016'
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    aspect_id = 1

    if format_type == '2014':
        # 2014 格式: <sentences><sentence><aspectTerms>
        for sentence in root.findall('sentence'):
            sent_id = sentence.get('id')
            text_elem = sentence.find('text')
            if text_elem is None or text_elem.text is None:
                continue
            text = text_elem.text

            aspect_terms = sentence.find('aspectTerms')
            if aspect_terms is not None:
                for aspect_term in aspect_terms.findall('aspectTerm'):
                    term = aspect_term.get('term')
                    polarity = aspect_term.get('polarity', '')  # 測試集可能無標籤
                    from_pos = aspect_term.get('from')
                    to_pos = aspect_term.get('to')

                    data.append({
                        'aspect_id': aspect_id,
                        'sentence_id': sent_id,
                        'sentence_text': text,
                        'aspect_term': term,
                        'aspect_polarity': polarity,
                        'from_position': from_pos,
                        'to_position': to_pos
                    })
                    aspect_id += 1

    elif format_type == '2016':
        # 2016 格式: <Reviews><Review><sentences><sentence><Opinions>
        for sentence in root.findall('.//sentence'):
            sent_id = sentence.get('id')
            text_elem = sentence.find('text')
            if text_elem is None or text_elem.text is None:
                continue
            text = text_elem.text

            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    target = opinion.get('target')
                    if target == 'NULL':  # 跳過 NULL target
                        continue
                    polarity = opinion.get('polarity', '')
                    from_pos = opinion.get('from', '0')
                    to_pos = opinion.get('to', '0')

                    data.append({
                        'aspect_id': aspect_id,
                        'sentence_id': sent_id,
                        'sentence_text': text,
                        'aspect_term': target,
                        'aspect_polarity': polarity,
                        'from_position': from_pos,
                        'to_position': to_pos
                    })
                    aspect_id += 1

    return data


def main():
    print("=" * 80)
    print("處理測試集數據（轉換為 CSV 格式）")
    print("=" * 80)
    print()

    datasets = [
        # SemEval 2014
        {
            'name': 'SemEval2014 Restaurants Test',
            'input': BASE_DIR / 'data' / 'raw' / 'SemEval-2014' / 'Restaurants_Test_Data_phaseB.xml',
            'output': OUTPUT_DIR / 'SemEval2014_Restaurant_test_cleaned.csv',
            'format': '2014'
        },
        {
            'name': 'SemEval2014 Laptops Test',
            'input': BASE_DIR / 'data' / 'raw' / 'SemEval-2014' / 'Laptops_Test_Data_phaseB.xml',
            'output': OUTPUT_DIR / 'SemEval2014_Laptop_test_cleaned.csv',
            'format': '2014'
        },
        # SemEval 2016
        {
            'name': 'SemEval2016 Restaurants Test',
            'input': BASE_DIR / 'data' / 'raw' / 'SemEval-2016' / 'restaurants_test_sb1.xml',
            'output': OUTPUT_DIR / 'SemEval2016_Restaurant_test_cleaned.csv',
            'format': '2016'
        },
        {
            'name': 'SemEval2016 Laptops Test',
            'input': BASE_DIR / 'data' / 'raw' / 'SemEval-2016' / 'laptops_test_sb1.xml',
            'output': OUTPUT_DIR / 'SemEval2016_Laptop_test_cleaned.csv',
            'format': '2016'
        },
    ]

    for dataset in datasets:
        print(f"\n處理 {dataset['name']}...")
        print("-" * 80)

        if not dataset['input'].exists():
            print(f"[跳過] 文件不存在: {dataset['input']}")
            continue

        # 解析 XML
        data = parse_xml_test(dataset['input'], dataset['format'])
        print(f"解析到 {len(data)} 個 aspects")

        # 轉換為 DataFrame 並儲存
        df = pd.DataFrame(data)
        df.to_csv(dataset['output'], index=False, encoding='utf-8')
        print(f"[完成] 已儲存至: {dataset['output']}")

    print("\n" + "=" * 80)
    print("[成功] 所有測試集處理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
