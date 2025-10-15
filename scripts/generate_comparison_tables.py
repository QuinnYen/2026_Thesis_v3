"""
生成完整實驗對比表

此腳本收集所有實驗結果並生成論文用表格
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def load_experiment_result(result_path: Path) -> Dict:
    """載入實驗結果 JSON"""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_all_results() -> List[Dict]:
    """收集所有實驗結果"""
    base_dir = Path('outputs/experiments/aspect_level_v2/SemEval2014_Restaurant')

    experiments = []

    # 定義要收集的實驗
    experiment_configs = [
        ('baseline_cleaned', 'Baseline', 1, '清理後'),
        ('pre_fusion_2layer_cleaned', 'Pre-Fusion', 2, '清理後'),
        ('post_fusion_2layer_cleaned', 'Post-Fusion', 2, '清理後'),
        ('post_fusion_3layer_cleaned', 'Post-Fusion', 3, '清理後'),
        ('post_fusion_4layer_cleaned', 'Post-Fusion', 4, '清理後'),
        ('post_fusion_5layer_cleaned', 'Post-Fusion', 5, '清理後'),
    ]

    for exp_dir, model_type, num_layers, data_version in experiment_configs:
        result_path = base_dir / exp_dir / 'results' / 'experiment_result.json'

        if result_path.exists():
            result = load_experiment_result(result_path)

            # 提取關鍵指標
            exp_data = {
                'model_type': model_type,
                'num_layers': num_layers,
                'data_version': data_version,
                'accuracy': result['final_metrics']['accuracy'],
                'macro_f1': result['final_metrics']['macro_f1'],
                'precision': result['final_metrics']['precision'],
                'recall': result['final_metrics']['recall'],
                'training_time': result.get('training_time_seconds', 0),
                'total_params': result.get('total_params', 0),
                'best_epoch': result.get('best_epoch', 0),
            }

            experiments.append(exp_data)
            print(f"[OK] 載入: {model_type} ({num_layers} 層)")
        else:
            print(f"[警告] 未找到: {result_path}")

    return experiments


def generate_table1_data_cleaning(experiments: List[Dict]) -> pd.DataFrame:
    """
    表 1：資料清理前後對比

    注意：目前沒有「原始資料」的結果，僅顯示清理後的結果
    """
    print("\n" + "="*80)
    print("生成表 1：資料清理前後對比")
    print("="*80)

    # 篩選 Baseline 模型
    baseline_data = [exp for exp in experiments if exp['model_type'] == 'Baseline']

    if not baseline_data:
        print("[警告] 沒有 Baseline 模型數據")
        return pd.DataFrame()

    # 建立表格
    table_data = []
    for exp in baseline_data:
        table_data.append({
            '模型': exp['model_type'],
            '資料版本': exp['data_version'],
            'Accuracy': f"{exp['accuracy']:.4f}",
            'Macro-F1': f"{exp['macro_f1']:.4f}",
            'Precision': f"{exp['precision']:.4f}",
            'Recall': f"{exp['recall']:.4f}",
            '訓練時間 (秒)': f"{exp['training_time']:.1f}",
        })

    df = pd.DataFrame(table_data)
    print(f"\n生成的表格 (共 {len(df)} 行):")
    print(df.to_string(index=False))

    return df


def generate_table2_fusion_comparison(experiments: List[Dict]) -> pd.DataFrame:
    """
    表 2：不同融合策略比較（清理後資料）
    """
    print("\n" + "="*80)
    print("生成表 2：不同融合策略比較")
    print("="*80)

    # 建立表格
    table_data = []
    for exp in experiments:
        table_data.append({
            '模型': exp['model_type'],
            '層數': exp['num_layers'],
            'Accuracy': f"{exp['accuracy']:.4f}",
            'Macro-F1': f"{exp['macro_f1']:.4f}",
            'Precision': f"{exp['precision']:.4f}",
            'Recall': f"{exp['recall']:.4f}",
            '參數量': f"{exp['total_params']:,}" if exp['total_params'] > 0 else '-',
            '訓練時間 (秒)': f"{exp['training_time']:.1f}",
        })

    df = pd.DataFrame(table_data)

    # 按模型類型和層數排序
    model_order = {'Baseline': 0, 'Pre-Fusion': 1, 'Post-Fusion': 2}
    df['_sort_key'] = df['模型'].map(model_order)
    df = df.sort_values(['_sort_key', '層數']).drop('_sort_key', axis=1).reset_index(drop=True)

    print(f"\n生成的表格 (共 {len(df)} 行):")
    print(df.to_string(index=False))

    return df


def generate_table3_prefusion_vs_postfusion(experiments: List[Dict]) -> pd.DataFrame:
    """
    表 3：Pre-Fusion vs Post-Fusion 詳細比較
    """
    print("\n" + "="*80)
    print("生成表 3：Pre-Fusion vs Post-Fusion 詳細比較")
    print("="*80)

    # 篩選 Pre-Fusion 和 Post-Fusion
    fusion_data = [exp for exp in experiments if exp['model_type'] in ['Pre-Fusion', 'Post-Fusion']]

    if not fusion_data:
        print("[警告] 沒有 Fusion 模型數據")
        return pd.DataFrame()

    # 建立表格
    table_data = []
    for exp in fusion_data:
        table_data.append({
            '架構': exp['model_type'],
            '層數': exp['num_layers'],
            'Accuracy': f"{exp['accuracy']:.4f}",
            'Macro-F1': f"{exp['macro_f1']:.4f}",
            'Precision': f"{exp['precision']:.4f}",
            'Recall': f"{exp['recall']:.4f}",
            '參數量': f"{exp['total_params']:,}" if exp['total_params'] > 0 else '-',
            '最佳 Epoch': exp['best_epoch'],
            '訓練時間 (秒)': f"{exp['training_time']:.1f}",
        })

    df = pd.DataFrame(table_data)

    # 計算性能提升
    if len(df) > 0:
        print("\n性能分析:")

        # 按層數分組比較
        for num_layers in df['層數'].unique():
            layer_data = df[df['層數'] == num_layers]

            if len(layer_data) == 2:
                pre_fusion = layer_data[layer_data['架構'] == 'Pre-Fusion']
                post_fusion = layer_data[layer_data['架構'] == 'Post-Fusion']

                if not pre_fusion.empty and not post_fusion.empty:
                    pre_f1 = float(pre_fusion['Macro-F1'].values[0])
                    post_f1 = float(post_fusion['Macro-F1'].values[0])
                    improvement = ((post_f1 - pre_f1) / pre_f1) * 100

                    print(f"  {num_layers} 層: Post-Fusion F1 提升 {improvement:+.2f}%")

    print(f"\n生成的表格 (共 {len(df)} 行):")
    print(df.to_string(index=False))

    return df


def generate_summary_table(experiments: List[Dict]) -> pd.DataFrame:
    """
    完整結果總表
    """
    print("\n" + "="*80)
    print("生成完整結果總表")
    print("="*80)

    # 建立詳細表格
    table_data = []
    for exp in experiments:
        table_data.append({
            '模型': exp['model_type'],
            '層數': exp['num_layers'],
            '資料版本': exp['data_version'],
            'Accuracy': exp['accuracy'],
            'Macro-F1': exp['macro_f1'],
            'Precision': exp['precision'],
            'Recall': exp['recall'],
            '參數量': exp['total_params'],
            '最佳Epoch': exp['best_epoch'],
            '訓練時間(秒)': exp['training_time'],
        })

    df = pd.DataFrame(table_data)

    # 找出最佳模型
    best_model_idx = df['Macro-F1'].idxmax()
    best_model = df.loc[best_model_idx]

    print("\n[最佳模型]")
    print(f"  模型: {best_model['模型']} ({best_model['層數']} 層)")
    print(f"  Accuracy: {best_model['Accuracy']:.4f}")
    print(f"  Macro-F1: {best_model['Macro-F1']:.4f}")

    print(f"\n生成的總表 (共 {len(df)} 行):")
    print(df.to_string(index=False))

    return df


def generate_table_captions() -> str:
    """生成表格說明文字"""
    captions = """
表格說明 (Table Captions)
================================================================================

表 1：資料清理前後對比
Table 1: Comparison Before and After Data Cleaning
--------------------------------------------------------------------------------
說明：比較 Baseline 模型在清理前後資料集上的表現。
Note: Performance comparison of Baseline model on original vs. cleaned datasets.

表 2：不同融合策略比較（清理後資料）
Table 2: Comparison of Different Fusion Strategies (Cleaned Data)
--------------------------------------------------------------------------------
說明：在清理後的資料集上，比較 Baseline、Pre-Fusion 和 Post-Fusion 三種架構
      在不同層數下的表現。
Note: Performance comparison of Baseline, Pre-Fusion, and Post-Fusion
      architectures with different layer numbers on cleaned dataset.

表 3：前融合 vs 後融合詳細比較
Table 3: Detailed Comparison of Pre-Fusion vs Post-Fusion
--------------------------------------------------------------------------------
說明：詳細比較 Pre-Fusion (Attention → BiLSTM) 和 Post-Fusion (BiLSTM → Attention)
      兩種融合策略的性能差異。
Note: Detailed comparison between Pre-Fusion (Attention → BiLSTM) and
      Post-Fusion (BiLSTM → Attention) strategies.

完整結果總表
Complete Results Summary
--------------------------------------------------------------------------------
說明：所有實驗的完整結果匯總，包含詳細的性能指標和訓練資訊。
Note: Complete summary of all experiments with detailed performance metrics
      and training information.

================================================================================
"""
    return captions


def main():
    print("="*80)
    print("生成完整實驗對比表")
    print("="*80)

    # 1. 收集所有實驗結果
    print("\n1. 收集實驗結果...")
    experiments = collect_all_results()

    if not experiments:
        print("\n[錯誤] 沒有找到任何實驗結果")
        return

    print(f"\n共收集到 {len(experiments)} 個實驗結果")

    # 2. 建立輸出目錄
    output_dir = Path('outputs/paper_materials/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n輸出目錄: {output_dir}")

    # 3. 生成表 1
    table1 = generate_table1_data_cleaning(experiments)
    if not table1.empty:
        table1_path = output_dir / 'table1_data_cleaning_comparison.csv'
        table1.to_csv(table1_path, index=False, encoding='utf-8-sig')
        print(f"[OK] 已儲存: {table1_path}")

    # 4. 生成表 2
    table2 = generate_table2_fusion_comparison(experiments)
    if not table2.empty:
        table2_path = output_dir / 'table2_fusion_strategy_comparison.csv'
        table2.to_csv(table2_path, index=False, encoding='utf-8-sig')
        print(f"[OK] 已儲存: {table2_path}")

    # 5. 生成表 3
    table3 = generate_table3_prefusion_vs_postfusion(experiments)
    if not table3.empty:
        table3_path = output_dir / 'table3_prefusion_vs_postfusion.csv'
        table3.to_csv(table3_path, index=False, encoding='utf-8-sig')
        print(f"[OK] 已儲存: {table3_path}")

    # 6. 生成完整總表
    summary = generate_summary_table(experiments)
    if not summary.empty:
        summary_path = output_dir / 'all_results_summary.csv'
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"[OK] 已儲存: {summary_path}")

    # 7. 生成表格說明
    captions = generate_table_captions()
    captions_path = output_dir.parent / 'table_captions.txt'
    with open(captions_path, 'w', encoding='utf-8') as f:
        f.write(captions)
    print(f"[OK] 已儲存: {captions_path}")

    # 8. 總結
    print("\n" + "="*80)
    print("生成完成！")
    print("="*80)
    print(f"\n已生成檔案:")
    print(f"  - 表 1: table1_data_cleaning_comparison.csv")
    print(f"  - 表 2: table2_fusion_strategy_comparison.csv")
    print(f"  - 表 3: table3_prefusion_vs_postfusion.csv")
    print(f"  - 總表: all_results_summary.csv")
    print(f"  - 說明: table_captions.txt")
    print(f"\n輸出目錄: {output_dir.parent}")


if __name__ == "__main__":
    main()
