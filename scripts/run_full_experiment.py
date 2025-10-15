"""
完整實驗流程主控腳本

自動執行以下所有步驟：
1. 資料清理與驗證
2. 訓練所有模型（Baseline + Pre-Fusion + Post-Fusion）
3. 生成比較表格
4. 生成高品質圖表
5. 產生完整實驗報告

使用方式：
    python scripts/run_full_experiment.py

參數：
    --skip-cleaning     跳過資料清理步驟
    --skip-training     跳過模型訓練步驟
    --skip-analysis     跳過分析與視覺化步驟
    --models            指定要訓練的模型（預設：all）
                       可選：baseline, prefusion, postfusion, all
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse
import json

# 設定專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "configs"))

# 導入配置
from experiment_config import DATASETS


def print_section(title):
    """印出分隔線標題"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80 + "\n")


def run_command(description, command, timeout=3600):
    """
    執行系統命令並即時顯示輸出

    Args:
        description: 命令描述
        command: 要執行的命令（字串或列表）
        timeout: 超時時間（秒）

    Returns:
        bool: 命令是否成功執行
    """
    print(f"\n>>> {description}")
    print(f">>> 執行命令: {' '.join(command) if isinstance(command, list) else command}")
    print("-" * 80)

    start_time = time.time()

    try:
        # 使用 subprocess.run 並即時顯示輸出
        result = subprocess.run(
            command if isinstance(command, list) else command.split(),
            cwd=BASE_DIR,
            capture_output=False,  # 直接顯示輸出
            text=True,
            timeout=timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[成功] {description} - 耗時 {elapsed_time:.1f} 秒")
            return True
        else:
            print(f"\n[失敗] {description} - 返回碼 {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n[超時] {description} - 超過 {timeout} 秒")
        return False
    except Exception as e:
        print(f"\n[錯誤] {description} - {str(e)}")
        return False


def step1_data_cleaning():
    """步驟 1：資料清理與驗證"""
    print_section("步驟 1/4：資料清理與驗證")

    # 1.1 清理資料
    success = run_command(
        "清理 Aspect-Level 資料（移除衝突標籤）",
        ["python", "scripts/clean_aspect_data.py"],
        timeout=300
    )
    if not success:
        print("\n[警告] 資料清理失敗，但將繼續執行")

    # 1.2 驗證清理結果
    success = run_command(
        "驗證清理後資料品質",
        ["python", "scripts/verify_cleaning.py"],
        timeout=120
    )
    if not success:
        print("\n[警告] 資料驗證失敗，但將繼續執行")

    return True


def step2_train_models(models='all'):
    """步驟 2：訓練所有模型（在所有配置的數據集上）"""
    print_section("步驟 2/4：模型訓練")

    print(f"將在以下 {len(DATASETS)} 個數據集上訓練模型:")
    for dataset in DATASETS:
        print(f"  - {dataset}")
    print()

    # 使用 train_all_datasets.py 腳本
    success = run_command(
        f"在所有數據集上訓練模型（模型類型: {models}）",
        ["python", "scripts/train_all_datasets.py", "--model", models],
        timeout=14400  # 4 小時
    )

    if not success:
        print("\n[錯誤] 模型訓練失敗")
        return False

    print("\n[成功] 所有模型訓練完成")
    return True


def step3_generate_analysis():
    """步驟 3：生成分析報告與視覺化"""
    print_section("步驟 3/4：結果分析與視覺化")

    # 3.1 生成比較表格
    print("\n--- 生成比較表格 ---")
    success = run_command(
        "生成論文比較表格（Table 1-3）",
        ["python", "scripts/generate_comparison_tables.py"],
        timeout=300
    )
    if not success:
        print("\n[錯誤] 表格生成失敗")
        return False

    # 3.2 生成高品質圖表
    print("\n--- 生成高品質圖表 ---")
    success = run_command(
        "生成論文圖表（Figure 1-5，300 DPI）",
        ["python", "scripts/generate_figures.py"],
        timeout=300
    )
    if not success:
        print("\n[錯誤] 圖表生成失敗")
        return False

    return True


def step4_generate_report():
    """步驟 4：產生完整實驗報告（包含所有數據集）"""
    print_section("步驟 4/4：產生完整實驗報告")

    # 使用獨立的報告生成腳本
    success = run_command(
        "生成完整實驗報告（包含所有數據集）",
        ["python", "scripts/generate_report.py"],
        timeout=120
    )

    if not success:
        print("\n[錯誤] 報告生成失敗")
        return False

    return True

def step4_generate_report_old():
    """步驟 4：產生完整實驗報告（包含所有數據集）- 舊版本"""
    print_section("步驟 4/4：產生完整實驗報告")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("完整實驗報告")
    report_lines.append("=" * 80)
    report_lines.append("")

    base_experiment_dir = BASE_DIR / "outputs" / "experiments" / "aspect_level_v2"

    # 遍歷所有數據集
    for dataset_name in DATASETS:
        experiment_dir = base_experiment_dir / dataset_name

        if not experiment_dir.exists():
            report_lines.append(f"\n⚠️  {dataset_name}: 無實驗結果")
            continue

        report_lines.append(f"\n{'=' * 80}")
        report_lines.append(f"數據集: {dataset_name}")
        report_lines.append(f"{'=' * 80}")

        # 掃描所有實驗結果
        models_found = []

        # Baseline
        baseline_path = experiment_dir / "baseline_cleaned" / "results" / "experiment_result.json"
        if baseline_path.exists():
            with open(baseline_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                models_found.append({
                    'name': 'Baseline',
                    'accuracy': data['final_metrics']['accuracy'],
                    'f1': data['final_metrics']['macro_f1'],
                    'params': data.get('model_info', {}).get('total_params', 'N/A')
                })

        # Pre-Fusion
        for layer_dir in experiment_dir.glob("pre_fusion_*layer_cleaned"):
            result_file = layer_dir / "results" / "experiment_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    layers = layer_dir.name.split('_')[2].replace('layer', '')
                    models_found.append({
                        'name': f'Pre-Fusion {layers}L',
                        'accuracy': data['final_metrics']['accuracy'],
                        'f1': data['final_metrics']['macro_f1'],
                        'params': data.get('model_info', {}).get('total_params', 'N/A')
                    })

        # Post-Fusion
        for layer_dir in experiment_dir.glob("post_fusion_*layer_cleaned"):
            result_file = layer_dir / "results" / "experiment_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    layers = layer_dir.name.split('_')[2].replace('layer', '')
                    models_found.append({
                        'name': f'Post-Fusion {layers}L',
                        'accuracy': data['final_metrics']['accuracy'],
                        'f1': data['final_metrics']['macro_f1'],
                        'params': data.get('model_info', {}).get('total_params', 'N/A')
                    })

        if not models_found:
            report_lines.append("  無模型結果")
            continue

        # 排序並顯示結果
        models_found.sort(key=lambda x: x['f1'], reverse=True)

        report_lines.append("\n模型性能排名（依 Macro-F1 排序）")
        report_lines.append("-" * 80)
        report_lines.append(f"{'排名':<6} {'模型':<20} {'Accuracy':<12} {'Macro-F1':<12} {'參數量':<15}")
        report_lines.append("-" * 80)

        for i, model in enumerate(models_found, 1):
            rank_marker = "⭐" if i == 1 else f"{i}."
            # 格式化參數量（處理 N/A 或數字）
            params_str = f"{model['params']:,}" if isinstance(model['params'], int) else str(model['params'])
            report_lines.append(
                f"{rank_marker:<6} {model['name']:<20} "
                f"{model['accuracy']:<12.4f} {model['f1']:<12.4f} "
                f"{params_str:<15}"
            )

        report_lines.append("-" * 80)

        # 最佳模型
        best = models_found[0]
        best_params_str = f"{best['params']:,}" if isinstance(best['params'], int) else str(best['params'])
        report_lines.append(f"\n🏆 最佳模型：{best['name']}")
        report_lines.append(f"   - Accuracy: {best['accuracy']:.4f}")
        report_lines.append(f"   - Macro-F1: {best['f1']:.4f}")
        report_lines.append(f"   - 參數量: {best_params_str}")

    # 輸出位置
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("輸出檔案位置")
    report_lines.append("=" * 80)
    report_lines.append(f"\n📊 比較表格: {BASE_DIR / 'outputs' / 'paper_materials' / 'tables' / ''}")
    report_lines.append(f"📈 高品質圖表: {BASE_DIR / 'outputs' / 'paper_materials' / 'figures' / ''}")
    report_lines.append(f"💾 模型檢查點: {BASE_DIR / 'outputs' / 'checkpoints' / ''}")
    report_lines.append(f"📁 實驗結果: {base_experiment_dir}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("[成功] 完整實驗流程執行完畢！")
    report_lines.append("=" * 80)

    # 印出報告
    report_text = "\n".join(report_lines)
    print(report_text)

    # 儲存報告
    report_path = BASE_DIR / "outputs" / "paper_materials" / "experiment_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n報告已儲存: {report_path}")

    return True


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='執行完整的 ABSA 實驗流程')
    parser.add_argument('--skip-cleaning', action='store_true', help='跳過資料清理步驟')
    parser.add_argument('--skip-training', action='store_true', help='跳過模型訓練步驟')
    parser.add_argument('--skip-analysis', action='store_true', help='跳過分析與視覺化步驟')
    parser.add_argument('--models', choices=['all', 'baseline', 'prefusion', 'postfusion'],
                       default='all', help='指定要訓練的模型')

    args = parser.parse_args()

    print("=" * 80)
    print("完整 ABSA 實驗流程")
    print("=" * 80)
    print("\n流程說明：")
    print("  1. 資料清理與驗證")
    print("  2. 模型訓練（Baseline + Pre-Fusion + Post-Fusion）")
    print("  3. 結果分析與視覺化")
    print("  4. 產生完整實驗報告")
    print("\n預估總耗時：3-4 小時")

    # 確認執行
    response = input("\n是否開始執行？(y/n): ")
    if response.lower() != 'y':
        print("已取消執行")
        return

    start_time = time.time()

    # 步驟 1：資料清理
    if not args.skip_cleaning:
        success = step1_data_cleaning()
        if not success:
            print("\n[錯誤] 資料清理階段失敗")
            return
    else:
        print("\n[跳過] 資料清理步驟")

    # 步驟 2：模型訓練
    if not args.skip_training:
        success = step2_train_models(args.models)
        if not success:
            print("\n[錯誤] 模型訓練階段失敗")
            return
    else:
        print("\n[跳過] 模型訓練步驟")

    # 步驟 3：分析與視覺化
    if not args.skip_analysis:
        success = step3_generate_analysis()
        if not success:
            print("\n[錯誤] 分析與視覺化階段失敗")
            return
    else:
        print("\n[跳過] 分析與視覺化步驟")

    # 步驟 4：產生報告
    success = step4_generate_report()

    # 總結
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("\n" + "=" * 80)
    print(f"總執行時間: {hours} 小時 {minutes} 分 {seconds} 秒")
    print("=" * 80)


if __name__ == "__main__":
    main()
