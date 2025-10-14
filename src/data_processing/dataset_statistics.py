"""
資料集統計分析腳本
用於分析和視覺化 SemEval 資料集
"""

import os
import sys

# 支援直接執行此腳本
if __name__ == "__main__":
    # 將專案根目錄加入 Python 路徑
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

try:
    from src.data_processing.xml_parser import SemEvalDatasetLoader
except ImportError:
    from xml_parser import SemEvalDatasetLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DatasetStatistics:
    """資料集統計分析類別"""
    
    def __init__(self, loader: SemEvalDatasetLoader):
        self.loader = loader
        self.stats_summary = {}
    
    def collect_all_statistics(self):
        """收集所有資料集的統計資訊"""
        for name, dataset in self.loader.datasets.items():
            self.stats_summary[name] = dataset.get_statistics()
    
    def create_summary_table(self) -> pd.DataFrame:
        """建立統計摘要表格"""
        data = []
        for name, stats in self.stats_summary.items():
            row = {
                '資料集': name,
                '句子數': stats['total_sentences'],
                '面向詞數': stats['total_aspects'],
                '平均面向詞/句': f"{stats['avg_aspects_per_sentence']:.2f}",
                '正面': stats['sentiment_distribution'].get('positive', 0),
                '中性': stats['sentiment_distribution'].get('neutral', 0),
                '負面': stats['sentiment_distribution'].get('negative', 0),
                '衝突': stats['sentiment_distribution'].get('conflict', 0)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_summary_to_csv(self, output_path: str = "outputs/results/dataset_summary.csv"):
        """儲存統計摘要到 CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = self.create_summary_table()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 統計摘要已儲存至: {output_path}")
    
    def plot_sentiment_distribution(self, output_path: str = "outputs/visualizations/sentiment_distribution.png"):
        """繪製情感分佈圖"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 準備資料
        data = []
        for name, stats in self.stats_summary.items():
            for sentiment, count in stats['sentiment_distribution'].items():
                data.append({
                    '資料集': name,
                    '情感': sentiment,
                    '數量': count
                })
        
        if not data:
            print("⚠ 無資料可繪製")
            return
        
        df = pd.DataFrame(data)
        
        # 繪圖
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 子圖 1: 堆疊長條圖
        pivot_df = df.pivot_table(index='資料集', columns='情感', values='數量', fill_value=0)
        pivot_df.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set3')
        axes[0].set_title('各資料集情感分佈 (堆疊)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('資料集')
        axes[0].set_ylabel('面向詞數量')
        axes[0].legend(title='情感')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 子圖 2: 分組長條圖
        pivot_df.plot(kind='bar', ax=axes[1], colormap='Set2')
        axes[1].set_title('各資料集情感分佈 (分組)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('資料集')
        axes[1].set_ylabel('面向詞數量')
        axes[1].legend(title='情感')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 情感分佈圖已儲存至: {output_path}")
        plt.close()
    
    def plot_dataset_comparison(self, output_path: str = "outputs/visualizations/dataset_comparison.png"):
        """繪製資料集比較圖"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 準備資料
        datasets = []
        sentences = []
        aspects = []
        
        for name, stats in self.stats_summary.items():
            datasets.append(name)
            sentences.append(stats['total_sentences'])
            aspects.append(stats['total_aspects'])
        
        if not datasets:
            print("⚠ 無資料可繪製")
            return
        
        # 繪圖
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子圖 1: 句子數比較
        axes[0].bar(range(len(datasets)), sentences, color='skyblue', edgecolor='navy')
        axes[0].set_xticks(range(len(datasets)))
        axes[0].set_xticklabels(datasets, rotation=45, ha='right')
        axes[0].set_title('各資料集句子數比較', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('句子數')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 子圖 2: 面向詞數比較
        axes[1].bar(range(len(datasets)), aspects, color='lightcoral', edgecolor='darkred')
        axes[1].set_xticks(range(len(datasets)))
        axes[1].set_xticklabels(datasets, rotation=45, ha='right')
        axes[1].set_title('各資料集面向詞數比較', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('面向詞數')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 資料集比較圖已儲存至: {output_path}")
        plt.close()
    
    def generate_full_report(self):
        """生成完整統計報告"""
        print("\n" + "="*80)
        print(" "*25 + "資料集完整統計報告")
        print("="*80)
        
        # 列印摘要表格
        df = self.create_summary_table()
        print("\n摘要表格:")
        print(df.to_string(index=False))
        
        # 儲存結果
        self.save_summary_to_csv()
        self.plot_sentiment_distribution()
        self.plot_dataset_comparison()
        
        print("\n" + "="*80)
        print("✓ 報告生成完成！")
        print("="*80 + "\n")


def main():
    """主函數"""
    print("開始載入和分析 SemEval 資料集...\n")
    
    # 載入資料集
    loader = SemEvalDatasetLoader(base_path="data/raw")
    
    print("正在載入 SemEval-2014 資料集...")
    loader.load_semeval_2014(domain='both')
    
    print("正在載入 SemEval-2016 資料集...")
    loader.load_semeval_2016(domain='both')
    
    # 列印基本統計
    loader.print_all_statistics()
    
    # 生成詳細報告
    stats = DatasetStatistics(loader)
    stats.collect_all_statistics()
    stats.generate_full_report()


if __name__ == "__main__":
    main()

