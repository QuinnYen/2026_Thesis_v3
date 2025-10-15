"""
生成實驗報告

掃描所有數據集的實驗結果並生成完整報告
"""

import sys
from pathlib import Path
import json

# 設定專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "configs"))

from experiment_config import DATASETS


def generate_report():
    """產生完整實驗報告（包含所有數據集）"""
    print("=" * 80)
    print("產生完整實驗報告")
    print("=" * 80)
    print()

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("完整實驗報告")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"生成時間: {Path(__file__).stat().st_mtime}")
    report_lines.append("")

    base_experiment_dir = BASE_DIR / "outputs" / "experiments" / "aspect_level_v2"

    # 遍歷所有數據集
    datasets_with_results = 0
    total_models = 0

    for dataset_name in DATASETS:
        experiment_dir = base_experiment_dir / dataset_name

        if not experiment_dir.exists():
            print(f"[WARNING] {dataset_name}: Directory not found")
            report_lines.append(f"\n[WARNING] {dataset_name}: No experimental results")
            continue

        report_lines.append(f"\n{'=' * 80}")
        report_lines.append(f"數據集: {dataset_name}")
        report_lines.append(f"{'=' * 80}")

        # 掃描所有實驗結果
        models_found = []

        # Baseline
        baseline_path = experiment_dir / "baseline_cleaned" / "results" / "experiment_result.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    models_found.append({
                        'name': 'Baseline',
                        'accuracy': data['final_metrics']['accuracy'],
                        'f1': data['final_metrics']['macro_f1'],
                        'params': data.get('model_info', {}).get('total_params', 'N/A')
                    })
                    print(f"[OK] {dataset_name} - Baseline")
            except Exception as e:
                print(f"[FAIL] {dataset_name} - Baseline: {e}")

        # Pre-Fusion
        for layer_dir in sorted(experiment_dir.glob("pre_fusion_*layer_cleaned")):
            result_file = layer_dir / "results" / "experiment_result.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        layers = layer_dir.name.split('_')[2].replace('layer', '')
                        models_found.append({
                            'name': f'Pre-Fusion {layers}L',
                            'accuracy': data['final_metrics']['accuracy'],
                            'f1': data['final_metrics']['macro_f1'],
                            'params': data.get('model_info', {}).get('total_params', 'N/A')
                        })
                        print(f"[OK] {dataset_name} - Pre-Fusion {layers}L")
                except Exception as e:
                    print(f"[FAIL] {dataset_name} - Pre-Fusion: {e}")

        # Post-Fusion
        for layer_dir in sorted(experiment_dir.glob("post_fusion_*layer_cleaned")):
            result_file = layer_dir / "results" / "experiment_result.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        layers = layer_dir.name.split('_')[2].replace('layer', '')
                        models_found.append({
                            'name': f'Post-Fusion {layers}L',
                            'accuracy': data['final_metrics']['accuracy'],
                            'f1': data['final_metrics']['macro_f1'],
                            'params': data.get('model_info', {}).get('total_params', 'N/A')
                        })
                        print(f"[OK] {dataset_name} - Post-Fusion {layers}L")
                except Exception as e:
                    print(f"[FAIL] {dataset_name} - Post-Fusion: {e}")

        if not models_found:
            print(f"[WARNING] {dataset_name}: No model results found")
            report_lines.append("  無模型結果")
            continue

        datasets_with_results += 1
        total_models += len(models_found)

        # 排序並顯示結果
        models_found.sort(key=lambda x: x['f1'], reverse=True)

        report_lines.append(f"\n找到 {len(models_found)} 個模型結果")
        report_lines.append("\n模型性能排名（依 Macro-F1 排序）")
        report_lines.append("-" * 80)
        report_lines.append(f"{'排名':<6} {'模型':<20} {'Accuracy':<12} {'Macro-F1':<12} {'參數量':<15}")
        report_lines.append("-" * 80)

        for i, model in enumerate(models_found, 1):
            rank_marker = "[*]" if i == 1 else f"{i}."
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
        report_lines.append(f"\n[BEST] Model: {best['name']}")
        report_lines.append(f"   - Accuracy: {best['accuracy']:.4f}")
        report_lines.append(f"   - Macro-F1: {best['f1']:.4f}")
        report_lines.append(f"   - 參數量: {best_params_str}")

    # 總結
    report_lines.append("\n\n" + "=" * 80)
    report_lines.append("實驗總結")
    report_lines.append("=" * 80)
    report_lines.append(f"配置的數據集: {len(DATASETS)} 個")
    report_lines.append(f"有結果的數據集: {datasets_with_results} 個")
    report_lines.append(f"總模型數量: {total_models} 個")

    # 輸出位置
    report_lines.append("\n" + "=" * 80)
    report_lines.append("輸出檔案位置")
    report_lines.append("=" * 80)
    report_lines.append(f"\n[Tables]     {BASE_DIR / 'outputs' / 'paper_materials' / 'tables' / ''}")
    report_lines.append(f"[Figures]    {BASE_DIR / 'outputs' / 'paper_materials' / 'figures' / ''}")
    report_lines.append(f"[Checkpoints] {BASE_DIR / 'outputs' / 'checkpoints' / ''}")
    report_lines.append(f"[Results]    {base_experiment_dir}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("[成功] 實驗報告生成完畢！")
    report_lines.append("=" * 80)

    # 印出報告
    report_text = "\n".join(report_lines)
    print("\n" + "=" * 80)
    print("報告內容:")
    print("=" * 80)
    print(report_text)

    # 儲存報告
    report_path = BASE_DIR / "outputs" / "paper_materials" / "experiment_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n報告已儲存: {report_path}")

    return True


if __name__ == "__main__":
    generate_report()
