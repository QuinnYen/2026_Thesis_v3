"""
多維度實驗結果分析腳本

功能：
1. 自動收集所有實驗結果
2. 按維度分組統計
3. 生成比較表格
4. 繪製視覺化圖表
5. 執行統計顯著性檢驗
6. 生成論文用的 LaTeX 表格

使用範例：
    # 分析指定會話的結果
    python scripts/analyze_multi_dimensional_results.py --session 20231029_123456

    # 分析所有實驗結果
    python scripts/analyze_multi_dimensional_results.py --all

    # 生成論文表格
    python scripts/analyze_multi_dimensional_results.py --session 20231029_123456 --generate-latex
"""

import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from scipy import stats
from collections import defaultdict

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "configs"))

from experiment_dimensions import (
    get_dimension_config,
    DIMENSION_COMBINATIONS
)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 結果收集器
# ============================================================================

class ResultCollector:
    """實驗結果收集器"""

    def __init__(self, session_file: Path):
        self.session_file = session_file
        self.results = []
        self.load_results()

    def load_results(self):
        """載入實驗結果"""
        if not self.session_file.exists():
            raise FileNotFoundError(f"找不到會話檔案: {self.session_file}")

        with open(self.session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)

        self.session_id = session_data.get("session_id")
        self.results = session_data.get("results", [])

        print(f"已載入 {len(self.results)} 個實驗結果")

    def to_dataframe(self) -> pd.DataFrame:
        """轉換為 DataFrame"""
        if not self.results:
            return pd.DataFrame()

        # 展開結果
        rows = []
        for result in self.results:
            row = {
                'experiment_id': result.get('experiment_id'),
                'dataset': result.get('dataset'),
                'model_type': result.get('model_type'),
                'dimension': result.get('dimension'),
                'num_layers': result.get('num_layers'),
                'accuracy': result.get('final_metrics', {}).get('accuracy'),
                'precision': result.get('final_metrics', {}).get('precision'),
                'recall': result.get('final_metrics', {}).get('recall'),
                'macro_f1': result.get('final_metrics', {}).get('macro_f1'),
                'training_time': result.get('training_time_seconds')
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df


# ============================================================================
# 統計分析器
# ============================================================================

class StatisticalAnalyzer:
    """統計分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_dimension_statistics(self, dimension: int) -> pd.DataFrame:
        """計算指定維度的統計資料"""
        dim_df = self.df[self.df['dimension'] == dimension]

        if dim_df.empty:
            return pd.DataFrame()

        # 按模型類型分組
        stats = dim_df.groupby('model_type').agg({
            'macro_f1': ['mean', 'std', 'min', 'max', 'count'],
            'accuracy': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)

        return stats

    def compare_models(self, model1: str, model2: str, metric: str = 'macro_f1') -> Dict[str, Any]:
        """
        比較兩個模型

        執行 t 檢驗並返回結果
        """
        data1 = self.df[self.df['model_type'] == model1][metric].dropna()
        data2 = self.df[self.df['model_type'] == model2][metric].dropna()

        if len(data1) < 2 or len(data2) < 2:
            return {
                'error': '樣本數量不足',
                'n1': len(data1),
                'n2': len(data2)
            }

        # 執行 t 檢驗
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # 計算效應量（Cohen's d）
        pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'mean1': data1.mean(),
            'mean2': data2.mean(),
            'std1': data1.std(),
            'std2': data2.std(),
            'n1': len(data1),
            'n2': len(data2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        }

    def analyze_layer_effect(self, model_type: str) -> pd.DataFrame:
        """分析層數對性能的影響"""
        model_df = self.df[self.df['model_type'] == model_type]

        if model_df.empty or 'num_layers' not in model_df.columns:
            return pd.DataFrame()

        layer_stats = model_df.groupby('num_layers').agg({
            'macro_f1': ['mean', 'std', 'count'],
            'accuracy': ['mean', 'std']
        }).round(4)

        return layer_stats


# ============================================================================
# 視覺化器
# ============================================================================

class Visualizer:
    """結果視覺化器"""

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_dimension_comparison(self, dimension: int):
        """繪製維度內模型比較圖"""
        dim_df = self.df[self.df['dimension'] == dimension]

        if dim_df.empty:
            print(f"維度 {dimension} 沒有資料")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # F1 分數比較
        sns.boxplot(data=dim_df, x='model_type', y='macro_f1', ax=axes[0])
        axes[0].set_title(f'維度 {dimension} - Macro-F1 比較')
        axes[0].set_xlabel('模型類型')
        axes[0].set_ylabel('Macro-F1')
        axes[0].tick_params(axis='x', rotation=45)

        # 準確率比較
        sns.boxplot(data=dim_df, x='model_type', y='accuracy', ax=axes[1])
        axes[1].set_title(f'維度 {dimension} - Accuracy 比較')
        axes[1].set_xlabel('模型類型')
        axes[1].set_ylabel('Accuracy')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_file = self.output_dir / f'dimension_{dimension}_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已儲存圖表: {output_file}")

    def plot_layer_effect(self, model_type: str):
        """繪製層數影響圖"""
        model_df = self.df[self.df['model_type'] == model_type]

        if model_df.empty or 'num_layers' not in model_df.columns:
            print(f"模型 {model_type} 沒有層數資料")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # F1 vs 層數
        layer_stats = model_df.groupby('num_layers')['macro_f1'].agg(['mean', 'std']).reset_index()
        axes[0].plot(layer_stats['num_layers'], layer_stats['mean'], marker='o', linewidth=2)
        axes[0].fill_between(layer_stats['num_layers'],
                            layer_stats['mean'] - layer_stats['std'],
                            layer_stats['mean'] + layer_stats['std'],
                            alpha=0.3)
        axes[0].set_title(f'{model_type} - 層數對 F1 的影響')
        axes[0].set_xlabel('層數')
        axes[0].set_ylabel('Macro-F1')
        axes[0].grid(True, alpha=0.3)

        # Accuracy vs 層數
        layer_stats = model_df.groupby('num_layers')['accuracy'].agg(['mean', 'std']).reset_index()
        axes[1].plot(layer_stats['num_layers'], layer_stats['mean'], marker='o', linewidth=2, color='orange')
        axes[1].fill_between(layer_stats['num_layers'],
                            layer_stats['mean'] - layer_stats['std'],
                            layer_stats['mean'] + layer_stats['std'],
                            alpha=0.3, color='orange')
        axes[1].set_title(f'{model_type} - 層數對 Accuracy 的影響')
        axes[1].set_xlabel('層數')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / f'{model_type}_layer_effect.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已儲存圖表: {output_file}")

    def plot_overall_heatmap(self):
        """繪製整體熱力圖"""
        # 建立透視表
        pivot = self.df.pivot_table(
            values='macro_f1',
            index='model_type',
            columns='dataset',
            aggfunc='mean'
        )

        if pivot.empty:
            print("沒有足夠的資料繪製熱力圖")
            return

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Macro-F1'})
        plt.title('所有實驗 Macro-F1 熱力圖')
        plt.xlabel('數據集')
        plt.ylabel('模型類型')
        plt.tight_layout()

        output_file = self.output_dir / 'overall_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已儲存圖表: {output_file}")


# ============================================================================
# LaTeX 表格生成器
# ============================================================================

class LaTeXTableGenerator:
    """LaTeX 表格生成器"""

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dimension_table(self, dimension: int):
        """生成維度比較表格"""
        dim_df = self.df[self.df['dimension'] == dimension]

        if dim_df.empty:
            print(f"維度 {dimension} 沒有資料")
            return

        # 統計資料
        stats = dim_df.groupby('model_type').agg({
            'macro_f1': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(4)

        # 生成 LaTeX
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{維度 {dimension} 實驗結果比較}}")
        latex_lines.append("\\begin{tabular}{lcc}")
        latex_lines.append("\\toprule")
        latex_lines.append("模型 & Macro-F1 & Accuracy \\\\")
        latex_lines.append("\\midrule")

        for model in stats.index:
            f1_mean = stats.loc[model, ('macro_f1', 'mean')]
            f1_std = stats.loc[model, ('macro_f1', 'std')]
            acc_mean = stats.loc[model, ('accuracy', 'mean')]
            acc_std = stats.loc[model, ('accuracy', 'std')]

            latex_lines.append(f"{model} & ${f1_mean:.4f} \\pm {f1_std:.4f}$ & ${acc_mean:.4f} \\pm {acc_std:.4f}$ \\\\")

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # 儲存
        output_file = self.output_dir / f'dimension_{dimension}_table.tex'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))

        print(f"已儲存 LaTeX 表格: {output_file}")


# ============================================================================
# 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='多維度實驗結果分析')

    # 輸入選擇
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--session', type=str,
                      help='分析指定會話（提供 session_id）')
    group.add_argument('--all', action='store_true',
                      help='分析所有會話')

    # 輸出配置
    parser.add_argument('--checkpoint-dir', type=str,
                       default='outputs/experiment_checkpoints',
                       help='檢查點目錄')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/experiment_analysis',
                       help='分析結果輸出目錄')

    # 功能選項
    parser.add_argument('--generate-latex', action='store_true',
                       help='生成 LaTeX 表格')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='繪製視覺化圖表')

    args = parser.parse_args()

    print("=" * 80)
    print("多維度實驗結果分析")
    print("=" * 80)

    # 找到會話檔案
    checkpoint_dir = BASE_DIR / args.checkpoint_dir
    if args.session:
        session_files = [checkpoint_dir / f"session_{args.session}.json"]
    else:
        session_files = list(checkpoint_dir.glob("session_*.json"))

    if not session_files:
        print("找不到會話檔案")
        return

    print(f"\n找到 {len(session_files)} 個會話")

    # 分析每個會話
    for session_file in session_files:
        print(f"\n{'='*80}")
        print(f"分析會話: {session_file.stem}")
        print(f"{'='*80}")

        # 收集結果
        collector = ResultCollector(session_file)
        df = collector.to_dataframe()

        if df.empty:
            print("沒有實驗結果")
            continue

        print(f"\n資料摘要:")
        print(f"  總實驗數: {len(df)}")
        print(f"  數據集: {df['dataset'].unique().tolist()}")
        print(f"  模型類型: {df['model_type'].unique().tolist()}")
        print(f"  維度: {df['dimension'].unique().tolist()}")

        # 建立輸出目錄
        output_dir = BASE_DIR / args.output_dir / collector.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 統計分析
        analyzer = StatisticalAnalyzer(df)

        print(f"\n{'='*80}")
        print("統計分析")
        print(f"{'='*80}")

        for dimension in sorted(df['dimension'].unique()):
            print(f"\n--- 維度 {dimension} ---")
            stats = analyzer.compute_dimension_statistics(dimension)
            if not stats.empty:
                print(stats)

        # 視覺化
        if args.plot:
            print(f"\n{'='*80}")
            print("生成視覺化圖表")
            print(f"{'='*80}")

            visualizer = Visualizer(df, output_dir)

            # 各維度比較圖
            for dimension in sorted(df['dimension'].unique()):
                visualizer.plot_dimension_comparison(dimension)

            # 層數影響圖
            for model_type in df['model_type'].unique():
                if pd.notna(df[df['model_type'] == model_type]['num_layers']).any():
                    visualizer.plot_layer_effect(model_type)

            # 整體熱力圖
            visualizer.plot_overall_heatmap()

        # 生成 LaTeX 表格
        if args.generate_latex:
            print(f"\n{'='*80}")
            print("生成 LaTeX 表格")
            print(f"{'='*80}")

            latex_gen = LaTeXTableGenerator(df, output_dir)
            for dimension in sorted(df['dimension'].unique()):
                latex_gen.generate_dimension_table(dimension)

        # 儲存統計摘要
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("實驗結果統計摘要\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"會話ID: {collector.session_id}\n")
            f.write(f"總實驗數: {len(df)}\n\n")

            for dimension in sorted(df['dimension'].unique()):
                f.write(f"\n維度 {dimension}:\n")
                stats = analyzer.compute_dimension_statistics(dimension)
                f.write(stats.to_string())
                f.write("\n")

        print(f"\n統計摘要已儲存: {summary_file}")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
