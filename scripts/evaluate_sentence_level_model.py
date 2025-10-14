"""
評估 Sentence-Level 模型並生成詳細報告

包含：
1. 混淆矩陣
2. 分類報告
3. 錯誤分析
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse

from src.models.sentence_level import (
    BiLSTM_SentenceLevel,
    BiLSTM_SentenceLevel_PreFusion,
    BiLSTM_SentenceLevel_PostFusion
)
from src.data_processing.sentence_level_dataset import create_sentence_level_dataloaders


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='評估 Sentence-Level 模型')

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['base', 'pre_fusion', 'post_fusion'],
                        help='模型類型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型檢查點路徑')
    parser.add_argument('--dataset', type=str, default='SemEval2014_Restaurant',
                        help='資料集名稱')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/evaluations',
                        help='輸出目錄')

    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, save_path):
    """繪製混淆矩陣"""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  儲存混淆矩陣: {save_path}")
    plt.close()


def evaluate_model(args):
    """評估模型"""
    print("\n" + "="*80)
    print(f"評估 Sentence-Level 模型: {args.model_type}")
    print("="*80)

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"設備: {device}")

    # 創建輸出目錄
    output_dir = Path(args.output_dir) / args.dataset / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    print("\n[步驟 1] 載入資料")
    print("-"*80)

    dataloaders = create_sentence_level_dataloaders(
        dataset_name=args.dataset,
        batch_size=32,
        max_len=100,
        min_freq=2
    )

    test_loader = dataloaders['test']
    preprocessor = dataloaders['preprocessor']

    vocab_size = preprocessor.vocab_size
    num_classes = 3

    # 載入模型
    print("\n[步驟 2] 載入模型")
    print("-"*80)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 從 checkpoint 獲取模型配置
    model_args = checkpoint.get('args', {})
    hidden_size = model_args.get('hidden_size', 128)
    embedding_dim = model_args.get('embedding_dim', 300)
    num_lstm_layers = model_args.get('num_lstm_layers', 2)
    dropout = model_args.get('dropout', 0.3)

    # 創建模型
    model_classes = {
        'base': BiLSTM_SentenceLevel,
        'pre_fusion': BiLSTM_SentenceLevel_PreFusion,
        'post_fusion': BiLSTM_SentenceLevel_PostFusion
    }

    model = model_classes[args.model_type](
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        num_classes=num_classes,
        dropout=dropout
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型已載入: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  F1-Macro: {checkpoint.get('test_f1_macro', 'N/A'):.4f}")

    # 評估
    print("\n[步驟 3] 評估模型")
    print("-"*80)

    all_predictions = []
    all_labels = []
    all_texts = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']

            # 處理標籤維度
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            # 前向傳播
            if args.model_type == 'base':
                logits = model(input_ids)
            else:
                logits, _ = model(input_ids, attention_mask)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 計算指標
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

    print(f"\n測試集結果:")
    print(f"  準確度:     {accuracy:.4f}")
    print(f"  F1-Macro:   {f1_macro:.4f}")
    print(f"  F1-Micro:   {f1_micro:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")

    # 分類報告
    print("\n[步驟 4] 生成分類報告")
    print("-"*80)

    class_names = ['Negative', 'Neutral', 'Positive']
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(report)

    # 儲存分類報告
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"\nTest Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Macro: {f1_macro:.4f}\n")
        f.write(f"F1-Micro: {f1_micro:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n\n")
        f.write(report)

    print(f"\n分類報告已儲存: {report_path}")

    # 混淆矩陣
    print("\n[步驟 5] 生成混淆矩陣")
    print("-"*80)

    cm = confusion_matrix(all_labels, all_predictions)
    print("\n混淆矩陣:")
    print(cm)

    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)

    # 錯誤分析
    print("\n[步驟 6] 錯誤分析")
    print("-"*80)

    errors = []
    for i, (pred, true, text) in enumerate(zip(all_predictions, all_labels, all_texts)):
        if pred != true:
            errors.append({
                'index': i,
                'text': text,
                'true_label': class_names[true],
                'pred_label': class_names[pred]
            })

    print(f"\n錯誤樣本數: {len(errors)} / {len(all_labels)} ({len(errors)/len(all_labels)*100:.2f}%)")

    # 儲存錯誤樣本
    if errors:
        error_path = output_dir / 'error_analysis.txt'
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Error Analysis - {args.model_type}\n")
            f.write(f"Total Errors: {len(errors)} / {len(all_labels)}\n\n")

            for i, err in enumerate(errors[:20], 1):  # 顯示前20個錯誤
                f.write(f"{i}. [{err['true_label']} → {err['pred_label']}]\n")
                f.write(f"   Text: {err['text']}\n\n")

        print(f"錯誤分析已儲存: {error_path}")

        # 列印前5個錯誤
        print("\n前5個錯誤樣本:")
        for i, err in enumerate(errors[:5], 1):
            print(f"{i}. [{err['true_label']} → {err['pred_label']}]")
            print(f"   {err['text'][:80]}...")

    print("\n" + "="*80)
    print("評估完成")
    print("="*80)
    print(f"\n輸出目錄: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args)
