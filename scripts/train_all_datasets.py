"""
在多個數據集上訓練模型的包裝腳本

根據 configs/experiment_config.py 中的配置，
在所有指定的數據集上訓練指定的模型。

使用方式:
    python scripts/train_all_datasets.py --model baseline
    python scripts/train_all_datasets.py --model prefusion
    python scripts/train_all_datasets.py --model postfusion
    python scripts/train_all_datasets.py --model all
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "configs"))

from experiment_config import DATASETS, print_config


def run_training(script_name, dataset_name):
    """
    執行單個訓練腳本

    Args:
        script_name: 訓練腳本名稱
        dataset_name: 數據集名稱

    Returns:
        bool: 是否成功
    """
    print("\n" + "=" * 80)
    print(f"訓練模型: {script_name}")
    print(f"數據集: {dataset_name}")
    print("=" * 80)

    script_path = BASE_DIR / "scripts" / script_name
    cmd = [sys.executable, str(script_path), "--dataset", dataset_name]

    print(f"執行命令: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[成功] {script_name} on {dataset_name} - 耗時 {elapsed:.1f}秒")
            return True
        else:
            print(f"\n[失敗] {script_name} on {dataset_name} - 返回碼 {result.returncode}")
            return False

    except Exception as e:
        print(f"\n[錯誤] {script_name} on {dataset_name} - {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='在多個數據集上訓練模型')
    parser.add_argument('--model',
                       choices=['baseline', 'prefusion', 'postfusion', 'all'],
                       default='all',
                       help='要訓練的模型類型')

    args = parser.parse_args()

    print("=" * 80)
    print("多數據集訓練腳本")
    print("=" * 80)

    print_config()

    # 確定要執行的訓練腳本
    scripts = []
    if args.model in ['baseline', 'all']:
        scripts.append('train_baseline_cleaned.py')
    if args.model in ['prefusion', 'all']:
        scripts.append('train_prefusion_cleaned.py')
    if args.model in ['postfusion', 'all']:
        scripts.append('train_postfusion_cleaned.py')

    print(f"\n將執行以下訓練:")
    print(f"  數據集: {DATASETS}")
    print(f"  模型: {scripts}")

    response = input("\n是否開始執行？(y/n): ")
    if response.lower() != 'y':
        print("已取消執行")
        return

    total_start = time.time()
    results = {}

    # 遍歷所有數據集和模型
    for dataset in DATASETS:
        results[dataset] = {}
        for script in scripts:
            success = run_training(script, dataset)
            results[dataset][script] = success

    total_time = time.time() - total_start

    # 輸出總結
    print("\n" + "=" * 80)
    print("訓練結果總結")
    print("=" * 80)

    for dataset, script_results in results.items():
        print(f"\n{dataset}:")
        for script, success in script_results.items():
            status = "[成功]" if success else "[失敗]"
            print(f"  {status} {script}")

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\n總執行時間: {hours}時 {minutes}分 {seconds}秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
