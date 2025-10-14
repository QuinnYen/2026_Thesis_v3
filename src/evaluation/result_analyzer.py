"""
結果分析器（Result Analyzer）

用於分析和比較實驗結果，生成論文所需的表格和圖表。

功能：
1. 讀取所有實驗結果
2. 生成比較表格（CSV, Markdown, LaTeX）
3. 生成對比圖表（準確度、F1、訓練時間、參數量）
4. 統計顯著性檢驗
5. 生成分析報告
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 設置中文字體（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class ResultAnalyzer:
    """
    結果分析器

    讀取實驗結果並生成各種比較表格和圖表
    """

    def __init__(self, experiments_dir='outputs/experiments', output_dir='outputs/analysis'):
        """
        初始化結果分析器

        Args:
            experiments_dir: 實驗結果目錄
            output_dir: 分析輸出目錄
        """
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 創建子目錄
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        self.results_df = None

    def load_results(self) -> pd.DataFrame:
        """
        載入所有實驗結果

        Returns:
            包含所有實驗結果的 DataFrame
        """
        print("載入實驗結果...")

        results = []

        # 遍歷所有數據集
        for dataset_dir in self.experiments_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name == 'summary':
                continue

            dataset_name = dataset_dir.name

            # 遍歷所有模型
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name
                result_file = model_dir / 'results' / 'experiment_result.json'

                if result_file.exists():
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)

                        # 提取關鍵信息
                        results.append({
                            'dataset': dataset_name,
                            'model': model_name,
                            'model_type': self._get_model_type(model_name),
                            'num_layers': result.get('model_layers', 1),
                            'accuracy': result['test_metrics']['accuracy'],
                            'macro_f1': result['test_metrics']['macro_f1'],
                            'micro_f1': result['test_metrics']['micro_f1'],
                            'macro_precision': result['test_metrics']['macro_precision'],
                            'macro_recall': result['test_metrics']['macro_recall'],
                            'training_time': result['training_time'],
                            'best_epoch': result['best_epoch'],
                            'train_samples': result.get('train_samples', 0),
                            'test_samples': result.get('test_samples', 0),
                            'vocab_size': result.get('vocab_size', 0),
                        })

                    except Exception as e:
                        print(f"警告：無法載入 {result_file}: {e}")

        self.results_df = pd.DataFrame(results)
        print(f"成功載入 {len(self.results_df)} 個實驗結果")

        return self.results_df

    def _get_model_type(self, model_name: str) -> str:
        """獲取模型類型"""
        if 'post_fusion' in model_name:
            return 'Post-Fusion'
        elif 'pre_fusion' in model_name:
            return 'Pre-Fusion'
        elif 'baseline' in model_name:
            return 'Baseline'
        else:
            return 'Unknown'

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        生成比較表格

        Returns:
            比較表格 DataFrame
        """
        print("\n生成比較表格...")

        if self.results_df is None:
            self.load_results()

        # 創建比較表格
        table = self.results_df[[
            'dataset', 'model', 'model_type', 'num_layers',
            'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
            'training_time', 'best_epoch'
        ]].copy()

        # 格式化數值
        table['accuracy'] = table['accuracy'].apply(lambda x: f"{x*100:.2f}%")
        table['macro_f1'] = table['macro_f1'].apply(lambda x: f"{x:.4f}")
        table['macro_precision'] = table['macro_precision'].apply(lambda x: f"{x:.4f}")
        table['macro_recall'] = table['macro_recall'].apply(lambda x: f"{x:.4f}")
        table['training_time'] = table['training_time'].apply(lambda x: f"{x:.1f}s")

        # 重命名列
        table.columns = [
            'Dataset', 'Model', 'Type', 'Layers',
            'Accuracy', 'Macro-F1', 'Precision', 'Recall',
            'Time', 'Best Epoch'
        ]

        # 儲存為 CSV
        csv_path = self.tables_dir / 'comparison_table.csv'
        table.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV 表格已儲存: {csv_path}")

        # 儲存為 Markdown
        md_path = self.tables_dir / 'comparison_table.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 實驗結果比較表\n\n")
            f.write(table.to_markdown(index=False))
        print(f"Markdown 表格已儲存: {md_path}")

        return table

    def generate_latex_table(self, by_dataset=True):
        """
        生成 LaTeX 表格

        Args:
            by_dataset: 是否按數據集分組
        """
        print("\n生成 LaTeX 表格...")

        if self.results_df is None:
            self.load_results()

        latex_path = self.tables_dir / 'comparison_table.tex'

        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("% 實驗結果比較表（LaTeX格式）\n")
            f.write("% 可直接複製到論文中使用\n\n")

            if by_dataset:
                # 按數據集分組
                for dataset in self.results_df['dataset'].unique():
                    dataset_df = self.results_df[self.results_df['dataset'] == dataset]

                    f.write(f"% Dataset: {dataset}\n")
                    f.write("\\begin{table}[htbp]\n")
                    f.write("\\centering\n")
                    f.write(f"\\caption{{實驗結果比較 - {dataset}}}\n")
                    f.write("\\label{tab:results_" + dataset.lower().replace(' ', '_') + "}\n")
                    f.write("\\begin{tabular}{lcccccc}\n")
                    f.write("\\toprule\n")
                    f.write("Model & Layers & Accuracy & Macro-F1 & Precision & Recall & Time(s) \\\\\n")
                    f.write("\\midrule\n")

                    # 按模型類型和層數排序
                    dataset_df = dataset_df.sort_values(['model_type', 'num_layers'])

                    for _, row in dataset_df.iterrows():
                        f.write(
                            f"{row['model_type']} & "
                            f"{row['num_layers']} & "
                            f"{row['accuracy']*100:.2f}\\% & "
                            f"{row['macro_f1']:.4f} & "
                            f"{row['macro_precision']:.4f} & "
                            f"{row['macro_recall']:.4f} & "
                            f"{row['training_time']:.1f} \\\\\n"
                        )

                    f.write("\\bottomrule\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\end{table}\n\n")
            else:
                # 單一表格
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{實驗結果比較}\n")
                f.write("\\label{tab:results_all}\n")
                f.write("\\begin{tabular}{llcccccc}\n")
                f.write("\\toprule\n")
                f.write("Dataset & Model & Layers & Accuracy & Macro-F1 & Precision & Recall & Time(s) \\\\\n")
                f.write("\\midrule\n")

                # 按數據集、模型類型和層數排序
                sorted_df = self.results_df.sort_values(['dataset', 'model_type', 'num_layers'])

                for _, row in sorted_df.iterrows():
                    f.write(
                        f"{row['dataset']} & "
                        f"{row['model_type']} & "
                        f"{row['num_layers']} & "
                        f"{row['accuracy']*100:.2f}\\% & "
                        f"{row['macro_f1']:.4f} & "
                        f"{row['macro_precision']:.4f} & "
                        f"{row['macro_recall']:.4f} & "
                        f"{row['training_time']:.1f} \\\\\n"
                    )

                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")

        print(f"LaTeX 表格已儲存: {latex_path}")

    def plot_accuracy_comparison(self):
        """繪製準確度比較圖"""
        print("\n繪製準確度比較圖...")

        if self.results_df is None:
            self.load_results()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 為每個數據集繪製子圖
        for idx, dataset in enumerate(self.results_df['dataset'].unique()):
            ax = axes[idx] if len(self.results_df['dataset'].unique()) > 1 else axes

            dataset_df = self.results_df[self.results_df['dataset'] == dataset]

            # 分別繪製 Post-Fusion 和 Pre-Fusion
            for model_type in ['Post-Fusion', 'Pre-Fusion', 'Baseline']:
                model_df = dataset_df[dataset_df['model_type'] == model_type]
                if len(model_df) > 0:
                    model_df = model_df.sort_values('num_layers')
                    ax.plot(
                        model_df['num_layers'],
                        model_df['accuracy'] * 100,
                        marker='o',
                        label=model_type,
                        linewidth=2,
                        markersize=8
                    )

            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 6))

        plt.tight_layout()

        save_path = self.figures_dir / 'accuracy_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"準確度比較圖已儲存: {save_path}")
        plt.close()

    def plot_f1_comparison(self):
        """繪製 F1 分數比較圖"""
        print("\n繪製 F1 分數比較圖...")

        if self.results_df is None:
            self.load_results()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, dataset in enumerate(self.results_df['dataset'].unique()):
            ax = axes[idx] if len(self.results_df['dataset'].unique()) > 1 else axes

            dataset_df = self.results_df[self.results_df['dataset'] == dataset]

            for model_type in ['Post-Fusion', 'Pre-Fusion', 'Baseline']:
                model_df = dataset_df[dataset_df['model_type'] == model_type]
                if len(model_df) > 0:
                    model_df = model_df.sort_values('num_layers')
                    ax.plot(
                        model_df['num_layers'],
                        model_df['macro_f1'],
                        marker='s',
                        label=model_type,
                        linewidth=2,
                        markersize=8
                    )

            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Macro-F1 Score', fontsize=12)
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 6))

        plt.tight_layout()

        save_path = self.figures_dir / 'f1_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"F1 分數比較圖已儲存: {save_path}")
        plt.close()

    def plot_training_time_comparison(self):
        """繪製訓練時間比較圖"""
        print("\n繪製訓練時間比較圖...")

        if self.results_df is None:
            self.load_results()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, dataset in enumerate(self.results_df['dataset'].unique()):
            ax = axes[idx] if len(self.results_df['dataset'].unique()) > 1 else axes

            dataset_df = self.results_df[self.results_df['dataset'] == dataset]

            for model_type in ['Post-Fusion', 'Pre-Fusion', 'Baseline']:
                model_df = dataset_df[dataset_df['model_type'] == model_type]
                if len(model_df) > 0:
                    model_df = model_df.sort_values('num_layers')
                    ax.plot(
                        model_df['num_layers'],
                        model_df['training_time'],
                        marker='^',
                        label=model_type,
                        linewidth=2,
                        markersize=8
                    )

            ax.set_xlabel('Number of Layers', fontsize=12)
            ax.set_ylabel('Training Time (seconds)', fontsize=12)
            ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 6))

        plt.tight_layout()

        save_path = self.figures_dir / 'training_time_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"訓練時間比較圖已儲存: {save_path}")
        plt.close()

    def plot_heatmap(self):
        """繪製性能熱力圖"""
        print("\n繪製性能熱力圖...")

        if self.results_df is None:
            self.load_results()

        # 為每個數據集創建熱力圖
        for dataset in self.results_df['dataset'].unique():
            dataset_df = self.results_df[self.results_df['dataset'] == dataset]

            # 創建透視表
            pivot_acc = dataset_df.pivot_table(
                values='accuracy',
                index='model_type',
                columns='num_layers',
                aggfunc='mean'
            ) * 100

            pivot_f1 = dataset_df.pivot_table(
                values='macro_f1',
                index='model_type',
                columns='num_layers',
                aggfunc='mean'
            )

            # 繪製準確度熱力圖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            sns.heatmap(
                pivot_acc,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                ax=ax1,
                cbar_kws={'label': 'Accuracy (%)'}
            )
            ax1.set_title(f'{dataset} - Accuracy Heatmap', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Number of Layers', fontsize=12)
            ax1.set_ylabel('Model Type', fontsize=12)

            # 繪製 F1 分數熱力圖
            sns.heatmap(
                pivot_f1,
                annot=True,
                fmt='.4f',
                cmap='YlGnBu',
                ax=ax2,
                cbar_kws={'label': 'Macro-F1'}
            )
            ax2.set_title(f'{dataset} - Macro-F1 Heatmap', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Layers', fontsize=12)
            ax2.set_ylabel('Model Type', fontsize=12)

            plt.tight_layout()

            save_path = self.figures_dir / f'heatmap_{dataset.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"熱力圖已儲存: {save_path}")
            plt.close()

    def statistical_tests(self):
        """執行統計顯著性檢驗"""
        print("\n執行統計顯著性檢驗...")

        if self.results_df is None:
            self.load_results()

        report_path = self.output_dir / 'statistical_tests.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 統計顯著性檢驗報告\n\n")

            for dataset in self.results_df['dataset'].unique():
                f.write(f"## {dataset}\n\n")

                dataset_df = self.results_df[self.results_df['dataset'] == dataset]

                # Post-Fusion vs Pre-Fusion（所有層數）
                post_f1 = dataset_df[dataset_df['model_type'] == 'Post-Fusion']['macro_f1']
                pre_f1 = dataset_df[dataset_df['model_type'] == 'Pre-Fusion']['macro_f1']

                if len(post_f1) > 1 and len(pre_f1) > 1:
                    t_stat, p_value = stats.ttest_ind(post_f1, pre_f1)

                    f.write("### Post-Fusion vs Pre-Fusion\n\n")
                    f.write(f"- Post-Fusion 平均 F1: {post_f1.mean():.4f} ± {post_f1.std():.4f}\n")
                    f.write(f"- Pre-Fusion 平均 F1: {pre_f1.mean():.4f} ± {pre_f1.std():.4f}\n")
                    f.write(f"- t-statistic: {t_stat:.4f}\n")
                    f.write(f"- p-value: {p_value:.4f}\n")
                    f.write(f"- 結論: {'顯著差異' if p_value < 0.05 else '無顯著差異'} (α=0.05)\n\n")

                # 各層數之間的比較
                f.write("### 各層數之間的比較\n\n")

                for model_type in ['Post-Fusion', 'Pre-Fusion']:
                    model_df = dataset_df[dataset_df['model_type'] == model_type]

                    if len(model_df) > 1:
                        f.write(f"#### {model_type}\n\n")
                        f.write("| Layers | Accuracy | Macro-F1 |\n")
                        f.write("|--------|----------|----------|\n")

                        model_df = model_df.sort_values('num_layers')
                        for _, row in model_df.iterrows():
                            f.write(
                                f"| {row['num_layers']} | "
                                f"{row['accuracy']*100:.2f}% | "
                                f"{row['macro_f1']:.4f} |\n"
                            )
                        f.write("\n")

        print(f"統計檢驗報告已儲存: {report_path}")

    def generate_analysis_report(self):
        """生成完整的分析報告"""
        print("\n生成分析報告...")

        if self.results_df is None:
            self.load_results()

        report_path = self.output_dir / 'analysis_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 實驗結果分析報告\n\n")
            f.write(f"**生成時間：** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 總體統計
            f.write("## 總體統計\n\n")
            f.write(f"- 總實驗數：{len(self.results_df)}\n")
            f.write(f"- 數據集數：{self.results_df['dataset'].nunique()}\n")
            f.write(f"- 模型類型數：{self.results_df['model_type'].nunique()}\n\n")

            # 最佳結果
            f.write("## 最佳結果\n\n")

            for dataset in self.results_df['dataset'].unique():
                dataset_df = self.results_df[self.results_df['dataset'] == dataset]

                best_acc = dataset_df.loc[dataset_df['accuracy'].idxmax()]
                best_f1 = dataset_df.loc[dataset_df['macro_f1'].idxmax()]

                f.write(f"### {dataset}\n\n")
                f.write(f"**最高準確度：**\n")
                f.write(f"- 模型：{best_acc['model_type']} ({best_acc['num_layers']} 層)\n")
                f.write(f"- 準確度：{best_acc['accuracy']*100:.2f}%\n")
                f.write(f"- Macro-F1：{best_acc['macro_f1']:.4f}\n\n")

                f.write(f"**最高 Macro-F1：**\n")
                f.write(f"- 模型：{best_f1['model_type']} ({best_f1['num_layers']} 層)\n")
                f.write(f"- 準確度：{best_f1['accuracy']*100:.2f}%\n")
                f.write(f"- Macro-F1：{best_f1['macro_f1']:.4f}\n\n")

            # 平均性能
            f.write("## 平均性能\n\n")

            avg_by_type = self.results_df.groupby('model_type').agg({
                'accuracy': 'mean',
                'macro_f1': 'mean',
                'training_time': 'mean'
            })

            f.write("### 按模型類型\n\n")
            f.write("| 模型類型 | 平均準確度 | 平均 Macro-F1 | 平均訓練時間 |\n")
            f.write("|----------|------------|---------------|---------------|\n")

            for model_type, row in avg_by_type.iterrows():
                f.write(
                    f"| {model_type} | "
                    f"{row['accuracy']*100:.2f}% | "
                    f"{row['macro_f1']:.4f} | "
                    f"{row['training_time']:.1f}s |\n"
                )
            f.write("\n")

            # 關鍵發現
            f.write("## 關鍵發現\n\n")

            # 比較 Post-Fusion 和 Pre-Fusion
            post_avg = self.results_df[self.results_df['model_type'] == 'Post-Fusion']['macro_f1'].mean()
            pre_avg = self.results_df[self.results_df['model_type'] == 'Pre-Fusion']['macro_f1'].mean()

            f.write(f"1. **架構比較：**\n")
            f.write(f"   - Post-Fusion 平均 F1：{post_avg:.4f}\n")
            f.write(f"   - Pre-Fusion 平均 F1：{pre_avg:.4f}\n")
            if post_avg > pre_avg:
                f.write(f"   - Post-Fusion 優於 Pre-Fusion {(post_avg-pre_avg)*100:.2f}%\n\n")
            else:
                f.write(f"   - Pre-Fusion 優於 Post-Fusion {(pre_avg-post_avg)*100:.2f}%\n\n")

            # 層數影響
            f.write(f"2. **層數影響：**\n")
            for model_type in ['Post-Fusion', 'Pre-Fusion']:
                model_df = self.results_df[self.results_df['model_type'] == model_type]
                if len(model_df) > 0:
                    best_layers = model_df.loc[model_df['macro_f1'].idxmax()]['num_layers']
                    f.write(f"   - {model_type} 最佳層數：{int(best_layers)}\n")
            f.write("\n")

            # 效率分析
            f.write(f"3. **效率分析：**\n")
            fastest = self.results_df.loc[self.results_df['training_time'].idxmin()]
            f.write(f"   - 最快訓練：{fastest['model_type']} ({fastest['num_layers']} 層)\n")
            f.write(f"   - 訓練時間：{fastest['training_time']:.1f}s\n")
            f.write(f"   - F1 分數：{fastest['macro_f1']:.4f}\n\n")

        print(f"分析報告已儲存: {report_path}")

    def run_full_analysis(self):
        """運行完整分析流程"""
        print("="*80)
        print("開始完整分析")
        print("="*80)

        # 1. 載入結果
        self.load_results()

        # 2. 生成表格
        self.generate_comparison_table()
        self.generate_latex_table(by_dataset=True)
        self.generate_latex_table(by_dataset=False)

        # 3. 生成圖表
        self.plot_accuracy_comparison()
        self.plot_f1_comparison()
        self.plot_training_time_comparison()
        self.plot_heatmap()

        # 4. 統計檢驗
        self.statistical_tests()

        # 5. 分析報告
        self.generate_analysis_report()

        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print(f"\n輸出目錄: {self.output_dir}")
        print(f"- 表格: {self.tables_dir}")
        print(f"- 圖表: {self.figures_dir}")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='分析實驗結果')
    parser.add_argument(
        '--experiments_dir',
        type=str,
        default='outputs/experiments',
        help='實驗結果目錄'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/analysis',
        help='分析輸出目錄'
    )

    args = parser.parse_args()

    # 創建分析器並運行
    analyzer = ResultAnalyzer(
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir
    )

    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
