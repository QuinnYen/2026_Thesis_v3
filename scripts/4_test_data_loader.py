"""
測試資料讀取器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import SemEvalDataLoader


def main():
    print("=" * 70)
    print(" 測試 SemEvalDataLoader")
    print("=" * 70)
    print()

    loader = SemEvalDataLoader()

    # 1. 顯示支援的資料集
    print("[1] 資料集支援狀態")
    print("-" * 70)

    all_datasets = loader.get_available_datasets()
    supported_datasets = loader.get_supported_datasets()

    print(f"所有資料集 ({len(all_datasets)} 個):")
    for dataset in all_datasets:
        is_supported = loader.is_dataset_supported(dataset)
        status = "[支援]" if is_supported else "[不支援]"
        print(f"  {status} {dataset}")

    print(f"\n支援的資料集 ({len(supported_datasets)} 個):")
    for dataset in supported_datasets:
        print(f"  - {dataset}")
    print()

    # 2. 載入所有支援的資料集
    print("\n[2] 載入所有支援的資料集（含測試集）")
    print("=" * 70)

    all_data = loader.load_all_datasets(only_supported=True)

    # 3. 顯示摘要
    print("\n[3] 資料集摘要")
    print("=" * 70)

    total_train = 0
    total_test = 0

    for dataset_name in supported_datasets:
        if dataset_name not in all_data:
            continue

        splits = all_data[dataset_name]
        train_count = len(splits.get('train', []))
        test_count = len(splits.get('test', []))

        total_train += train_count
        total_test += test_count

        print(f"\n{dataset_name}:")
        print(f"  訓練集: {train_count:,} 樣本")
        print(f"  測試集: {test_count:,} 樣本")

        # 顯示情感分佈
        if train_count > 0:
            train_data = splits['train']
            stats = loader.get_statistics(train_data)
            print(f"  情感分佈 (訓練集):")
            for polarity, count in sorted(stats['polarity_distribution'].items()):
                percentage = (count / train_count * 100) if train_count > 0 else 0
                print(f"    - {polarity:10s}: {count:4d} ({percentage:5.2f}%)")

    print(f"\n{'='*70}")
    print(f"總訓練樣本: {total_train:,}")
    print(f"總測試樣本: {total_test:,}")
    print(f"總計: {total_train + total_test:,}")
    print(f"{'='*70}")

    # 4. 驗證測試集有正確載入
    print("\n[4] 驗證測試集載入")
    print("-" * 70)

    for dataset_name in supported_datasets:
        test_data = all_data[dataset_name].get('test', [])
        if test_data:
            print(f"\n{dataset_name} (test):")
            print(f"  樣本數: {len(test_data)}")
            print(f"  第一筆: {test_data[0]}")
            print(f"  極性標籤: {test_data[0]['polarity']} (應為 'unknown')")

    # 5. 確認問題已解決
    print("\n" + "=" * 70)
    print(" 問題檢查")
    print("=" * 70)

    checks = []

    # 檢查 1: 測試集有載入
    laptop_test = all_data['SemEval2014_Laptop'].get('test', [])
    restaurant_test = all_data['SemEval2014_Restaurant'].get('test', [])

    checks.append(("測試集載入 (Laptop)", len(laptop_test) > 0, f"{len(laptop_test)} 樣本"))
    checks.append(("測試集載入 (Restaurant)", len(restaurant_test) > 0, f"{len(restaurant_test)} 樣本"))

    # 檢查 2: SemEval-2016 被跳過
    has_2016 = 'SemEval2016_Laptop' in all_data
    checks.append(("SemEval-2016 已跳過", not has_2016, "正確跳過不支援的格式"))

    # 檢查 3: 總樣本數正確
    expected_total = 7839  # 2358 + 654 + 3693 + 1134
    actual_total = total_train + total_test
    checks.append(("總樣本數正確", actual_total == expected_total, f"{actual_total:,} == {expected_total:,}"))

    print()
    all_passed = True
    for check_name, passed, detail in checks:
        status = "[O]" if passed else "[X]"
        print(f"{status} {check_name}: {detail}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("=" * 70)
        print(" 所有問題已解決！[OK]")
        print("=" * 70)
    else:
        print("=" * 70)
        print(" 仍有問題需要修正 [ERROR]")
        print("=" * 70)


if __name__ == "__main__":
    main()
