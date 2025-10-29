"""
在多個數據集上訓練模型的包裝腳本

根據 configs/experiment_config.py 中的配置，
在所有指定的數據集上訓練指定的模型。

此腳本是 train_unified.py 的簡化包裝器。

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

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "configs"))

from experiment_config import print_config


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

    # 建立命令
    cmd = [sys.executable, "scripts/train_unified.py", "--use_config_datasets"]

    if args.model == 'all':
        cmd.extend(["--models", "baseline", "prefusion", "postfusion"])
    else:
        cmd.extend(["--model", args.model])

    print(f"\n將執行命令: {' '.join(cmd)}")

    response = input("\n是否開始執行？(y/n): ")
    if response.lower() != 'y':
        print("已取消執行")
        return

    # 執行統一訓練腳本
    print("\n" + "=" * 80)
    print("開始訓練...")
    print("=" * 80 + "\n")

    result = subprocess.run(
        cmd,
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("[成功] 所有訓練完成")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[失敗] 訓練過程中發生錯誤")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
