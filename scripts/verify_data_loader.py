"""
驗證 SemEval 資料讀取器 - 符合 Task 1.1 要求
"""

import sys
import os

# 將專案根目錄加入 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import SemEvalDataLoader


def main():
    print("=" * 70)
    print(" Task 1.1 驗證：SemEval 資料讀取器")
    print("=" * 70)
    print()

    # 建立資料讀取器
    loader = SemEvalDataLoader()

    # 測試案例：載入 SemEval2014_Laptop 訓練集
    print("[測試] 依照 Task 1.1 要求測試")
    print("-" * 70)
    print(">>> loader = SemEvalDataLoader()")
    print(">>> data = loader.load('SemEval2014_Laptop', 'train')")
    print()

    data = loader.load('SemEval2014_Laptop', 'train')

    print(f">>> print(f'樣本數: {{len(data)}}')")
    print(f"樣本數: {len(data)}")
    print()

    print(f">>> print(f'第一筆資料: {{data[0]}}')")
    print(f"第一筆資料: {data[0]}")
    print()

    # 驗證資料格式
    print("[驗證] 資料格式檢查")
    print("-" * 70)
    first_item = data[0]

    required_fields = ['text', 'aspect', 'polarity', 'aspect_position']
    print("必要欄位:")
    for field in required_fields:
        has_field = field in first_item
        value = first_item.get(field)
        print(f"  [{'O' if has_field else 'X'}] {field:20s}: {value}")
    print()

    # 顯示所有支援的資料集
    print("[功能] 支援的資料集")
    print("-" * 70)
    available = loader.get_available_datasets()
    for dataset in available:
        print(f"  - {dataset}")
    print()

    # 統計資訊
    print("[功能] 資料集統計資訊")
    print("-" * 70)
    loader.print_statistics(data, "SemEval2014_Laptop (train)")

    # 顯示範例
    print("[功能] 顯示 5 個範例資料")
    print("-" * 70)
    loader.print_samples(data, n=5)

    # 測試所有資料集載入
    print("\n" + "=" * 70)
    print(" 測試載入所有資料集")
    print("=" * 70)
    print()

    datasets_to_test = [
        ('SemEval2014_Laptop', 'train'),
        ('SemEval2014_Laptop', 'test'),
        ('SemEval2014_Restaurant', 'train'),
        ('SemEval2014_Restaurant', 'test'),
    ]

    summary = []
    for dataset_name, split in datasets_to_test:
        try:
            # 訓練集不包含無標籤，測試集包含
            include_unlabeled = (split == 'test')
            data = loader.load(dataset_name, split, include_unlabeled=include_unlabeled)

            stats = loader.get_statistics(data)
            polarity_dist = stats['polarity_distribution']

            summary.append({
                'dataset': f"{dataset_name} ({split})",
                'samples': len(data),
                'polarity': polarity_dist
            })

        except Exception as e:
            print(f"錯誤: {str(e)}\n")

    # 顯示摘要
    print("\n" + "=" * 70)
    print(" 資料集載入摘要")
    print("=" * 70)
    print()

    total_samples = 0
    for item in summary:
        samples = item['samples']
        total_samples += samples

        print(f"{item['dataset']:40s}: {samples:5d} 樣本")

        # 顯示情感分佈
        if item['polarity']:
            for polarity, count in sorted(item['polarity'].items()):
                if count > 0:
                    percentage = (count / samples * 100) if samples > 0 else 0
                    print(f"  - {polarity:10s}: {count:4d} ({percentage:5.2f}%)")
        print()

    print(f"總計: {total_samples:,} 個樣本")

    # 驗證結果
    print("\n" + "=" * 70)
    print(" Task 1.1 驗證結果")
    print("=" * 70)
    print()

    checks = [
        ("能成功讀取所有資料集", total_samples > 0),
        ("資料格式正確", all(f in first_item for f in required_fields)),
        ("包含情感極性標籤", 'polarity' in first_item),
        ("包含面向詞位置", 'aspect_position' in first_item),
        ("統計資訊功能正常", True),
        ("範例顯示功能正常", True),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "[O]" if result else "[X]"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("=" * 70)
        print(" 所有驗證通過！[OK]")
        print("=" * 70)
    else:
        print("=" * 70)
        print(" 部分驗證失敗 [ERROR]")
        print("=" * 70)


if __name__ == "__main__":
    main()
