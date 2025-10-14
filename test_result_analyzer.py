"""
測試結果分析器

使用模擬數據測試結果分析器的功能
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import json
import os

# 創建測試數據
def create_mock_results():
    """創建模擬實驗結果"""
    print("創建模擬實驗結果...")

    output_dir = Path('outputs/test_experiments')

    # 模擬數據
    datasets = ['SemEval2014_Restaurant', 'SemEval2014_Laptop']
    models = [
        ('baseline', 'Baseline', 1),
        ('post_fusion_2', 'Post-Fusion', 2),
        ('post_fusion_3', 'Post-Fusion', 3),
        ('post_fusion_4', 'Post-Fusion', 4),
        ('post_fusion_5', 'Post-Fusion', 5),
        ('pre_fusion_2', 'Pre-Fusion', 2),
        ('pre_fusion_3', 'Pre-Fusion', 3),
        ('pre_fusion_4', 'Pre-Fusion', 4),
        ('pre_fusion_5', 'Pre-Fusion', 5),
    ]

    # 基準性能（模擬數據）
    base_accuracy = 0.70
    base_f1 = 0.60

    for dataset in datasets:
        for model_name, model_type, num_layers in models:
            # 模擬性能（層數越多，性能略有提升）
            layer_boost = (num_layers - 1) * 0.01
            dataset_boost = 0.02 if 'Restaurant' in dataset else 0.0

            accuracy = base_accuracy + layer_boost + dataset_boost
            macro_f1 = base_f1 + layer_boost + dataset_boost

            # 模擬訓練時間（層數越多，時間越長）
            training_time = 100 + num_layers * 50

            # 創建結果結構
            result_dir = output_dir / dataset / model_name / 'results'
            result_dir.mkdir(parents=True, exist_ok=True)

            result = {
                'dataset': dataset,
                'model': model_name,
                'model_layers': num_layers,
                'status': 'success',
                'training_time': training_time,
                'best_epoch': 15,
                'train_samples': 2954,
                'val_samples': 739,
                'test_samples': 739,
                'vocab_size': 3234,
                'num_classes': 4,
                'test_metrics': {
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'micro_f1': accuracy,
                    'macro_precision': macro_f1 + 0.01,
                    'macro_recall': macro_f1 - 0.01,
                },
            }

            # 儲存結果
            result_file = result_dir / 'experiment_result.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

    print(f"模擬結果已創建: {output_dir}")
    print(f"數據集: {len(datasets)}")
    print(f"模型: {len(models)}")
    print(f"總實驗數: {len(datasets) * len(models)}")


def test_result_analyzer():
    """測試結果分析器"""
    print("\n" + "="*80)
    print("測試結果分析器")
    print("="*80)

    # 創建模擬數據
    create_mock_results()

    # 導入分析器
    from src.evaluation.result_analyzer import ResultAnalyzer

    # 創建分析器
    analyzer = ResultAnalyzer(
        experiments_dir='outputs/test_experiments',
        output_dir='outputs/test_analysis'
    )

    print("\n" + "-"*80)
    print("運行完整分析...")
    print("-"*80)

    # 運行完整分析
    analyzer.run_full_analysis()

    print("\n" + "="*80)
    print("測試完成！")
    print("="*80)

    print("\n檢查輸出文件：")

    # 檢查生成的文件
    expected_files = [
        'tables/comparison_table.csv',
        'tables/comparison_table.md',
        'tables/comparison_table.tex',
        'figures/accuracy_comparison.png',
        'figures/f1_comparison.png',
        'figures/training_time_comparison.png',
        'figures/heatmap_semeval2014_restaurant.png',
        'figures/heatmap_semeval2014_laptop.png',
        'statistical_tests.md',
        'analysis_report.md',
    ]

    output_dir = Path('outputs/test_analysis')
    missing_files = []

    for file_path in expected_files:
        full_path = output_dir / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n警告：缺少 {len(missing_files)} 個文件")
    else:
        print(f"\n所有文件都已生成！")

    print(f"\n輸出目錄: {output_dir}")


if __name__ == '__main__':
    test_result_analyzer()
