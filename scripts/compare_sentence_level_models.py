"""
比較 Pre-Fusion vs Post-Fusion Sentence-Level 模型

讀取實驗結果並生成比較報告
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_result(result_path: Path):
    """載入實驗結果"""
    if not result_path.exists():
        return None

    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models():
    """比較前融合 vs 後融合模型"""
    print("\n" + "="*80)
    print("Sentence-Level 模型比較: Pre-Fusion vs Post-Fusion")
    print("="*80)

    # 實驗路徑
    base_dir = Path('outputs/experiments/sentence_level/SemEval2014_Restaurant')

    experiments = {
        'Pre-Fusion (128)': base_dir / 'pre_fusion' / 'results' / 'experiment_result.json',
        'Post-Fusion (128)': base_dir / 'post_fusion' / 'results' / 'experiment_result.json',
    }

    # 載入結果
    results = {}
    for name, path in experiments.items():
        result = load_experiment_result(path)
        if result:
            results[name] = result
        else:
            print(f"警告：找不到 {name} 的結果")

    if not results:
        print("錯誤：沒有找到任何實驗結果")
        return

    # 比較表
    print("\n" + "-"*80)
    print("模型比較")
    print("-"*80)

    comparison_data = []

    for name, result in results.items():
        comparison_data.append({
            '模型': name,
            '參數量': f"{result.get('hidden_size', 128)} hidden",
            '訓練樣本': result['train_samples'],
            '最佳 Epoch': result['best_epoch'],
            '訓練時間 (秒)': f"{result['training_time']:.1f}",
            '測試準確度': f"{result['test_accuracy']:.4f}",
            'F1-Macro': f"{result['test_f1_macro']:.4f}",
            'F1-Micro': f"{result['test_f1_micro']:.4f}"
        })

    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))

    # 繪製訓練曲線
    print("\n" + "-"*80)
    print("繪製訓練曲線")
    print("-"*80)

    output_dir = Path('outputs/comparisons/sentence_level')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. F1-Macro 曲線
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, result in results.items():
        history = result.get('history', {})
        f1_scores = history.get('test_f1_macro', [])
        epochs = range(1, len(f1_scores) + 1)

        ax.plot(epochs, f1_scores, marker='o', label=name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Macro Score')
    ax.set_title('Sentence-Level Models: F1-Macro Comparison')
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = output_dir / 'f1_macro_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  儲存: {save_path}")
    plt.close()

    # 2. 準確度曲線
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, result in results.items():
        history = result.get('history', {})
        test_acc = history.get('test_acc', [])
        epochs = range(1, len(test_acc) + 1)

        ax.plot(epochs, test_acc, marker='o', label=name, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Sentence-Level Models: Accuracy Comparison')
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  儲存: {save_path}")
    plt.close()

    # 3. 損失曲線
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for name, result in results.items():
        history = result.get('history', {})
        train_loss = history.get('train_loss', [])
        test_loss = history.get('test_loss', [])
        epochs = range(1, len(train_loss) + 1)

        ax1.plot(epochs, train_loss, marker='o', label=name, linewidth=2)
        ax2.plot(epochs, test_loss, marker='o', label=name, linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'loss_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  儲存: {save_path}")
    plt.close()

    # 4. 最終效能比較（條形圖）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    accuracies = [results[m]['test_accuracy'] for m in models]
    f1_macros = [results[m]['test_f1_macro'] for m in models]

    x = range(len(models))

    axes[0].bar(x, accuracies, color=['steelblue', 'coral'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy')
    axes[0].grid(axis='y', alpha=0.3)

    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    axes[1].bar(x, f1_macros, color=['steelblue', 'coral'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylabel('F1-Macro')
    axes[1].set_title('Test F1-Macro')
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(f1_macros):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    save_path = output_dir / 'final_performance_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  儲存: {save_path}")
    plt.close()

    # 儲存比較 CSV
    csv_path = output_dir / 'baseline_sentence_attention_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n比較表已儲存: {csv_path}")

    # 總結
    print("\n" + "="*80)
    print("比較總結")
    print("="*80)

    best_f1 = max(results.items(), key=lambda x: x[1]['test_f1_macro'])
    best_acc = max(results.items(), key=lambda x: x[1]['test_accuracy'])

    print(f"\n最佳 F1-Macro: {best_f1[0]} ({best_f1[1]['test_f1_macro']:.4f})")
    print(f"最佳準確度:   {best_acc[0]} ({best_acc[1]['test_accuracy']:.4f})")

    print(f"\n架構差異分析:")
    print(f"  Pre-Fusion:  Embedding → Attention → BiLSTM → Pooling → Classifier")
    print(f"  Post-Fusion: Embedding → BiLSTM → Attention → Classifier")

    print(f"\n輸出文件:")
    print(f"  比較表: {csv_path}")
    print(f"  圖表:   {output_dir}")

    print("\n" + "="*80)


if __name__ == '__main__':
    compare_models()
