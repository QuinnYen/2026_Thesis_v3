"""
訓練 Sentence-Level Baseline 模型

支援三種架構：
1. Base (無 Attention)
2. Pre-Fusion (Attention → BiLSTM)
3. Post-Fusion (BiLSTM → Attention)
"""

import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import time
from datetime import datetime
from collections import Counter

from src.models.sentence_level import (
    BiLSTM_SentenceLevel,
    BiLSTM_SentenceLevel_PreFusion,
    BiLSTM_SentenceLevel_PostFusion
)
from src.data_processing.sentence_level_dataset import create_sentence_level_dataloaders
from src.training.trainer import ABSATrainer
from src.evaluation.evaluator import ABSAEvaluator


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='訓練 Sentence-Level Baseline 模型')

    # 模型參數
    parser.add_argument('--model_type', type=str, default='post_fusion',
                        choices=['base', 'pre_fusion', 'post_fusion'],
                        help='模型類型')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='詞嵌入維度')
    parser.add_argument('--hidden_size', type=int, default=128,
                        choices=[128, 256],
                        help='LSTM 隱藏層大小（也用作 Attention dimension）')
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                        help='LSTM 層數')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout 比例')

    # 訓練參數
    parser.add_argument('--dataset', type=str, default='SemEval2014_Restaurant',
                        choices=['SemEval2014_Restaurant', 'SemEval2014_Laptop'],
                        help='資料集名稱')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='學習率')
    parser.add_argument('--max_len', type=int, default=100,
                        help='最大序列長度')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='最小詞頻')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='學習率衰減率')

    # 早停參數
    parser.add_argument('--patience', type=int, default=10,
                        help='早停的耐心值')

    # 輸出路徑
    parser.add_argument('--output_dir', type=str,
                        default='outputs/experiments/sentence_level',
                        help='輸出目錄')

    # 隨機種子
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    """設置隨機種子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def create_model(model_type: str,
                vocab_size: int,
                embedding_dim: int,
                hidden_size: int,
                num_lstm_layers: int,
                num_classes: int,
                dropout: float):
    """
    創建模型

    Args:
        model_type: 模型類型 ('base', 'pre_fusion', 'post_fusion')
        vocab_size: 詞彙表大小
        embedding_dim: 詞嵌入維度
        hidden_size: LSTM 隱藏層大小
        num_lstm_layers: LSTM 層數
        num_classes: 分類數量
        dropout: Dropout 比例

    Returns:
        模型實例
    """
    model_classes = {
        'base': BiLSTM_SentenceLevel,
        'pre_fusion': BiLSTM_SentenceLevel_PreFusion,
        'post_fusion': BiLSTM_SentenceLevel_PostFusion
    }

    model_class = model_classes[model_type]

    model = model_class(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        num_classes=num_classes,
        dropout=dropout
    )

    return model


def train_epoch(model, train_loader, criterion, optimizer, device, model_type):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 處理標籤維度
        if labels.dim() > 1:
            labels = labels.squeeze(-1)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        optimizer.zero_grad()

        # 前向傳播
        if model_type == 'base':
            logits = model(input_ids)
        else:
            logits, _ = model(input_ids, attention_mask)

        # 計算損失
        loss = criterion(logits, labels)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 統計
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, model_type):
    """評估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 處理標籤維度
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            # 前向傳播
            if model_type == 'base':
                logits = model(input_ids)
            else:
                logits, _ = model(input_ids, attention_mask)

            # 計算損失
            loss = criterion(logits, labels)

            # 統計
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    # 計算 F1 分數
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)

    return avg_loss, accuracy, f1_macro, f1_micro


def main():
    args = parse_args()

    # 設置隨機種子
    set_seed(args.seed)

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print(f"訓練 Sentence-Level Baseline: {args.model_type}")
    print("="*80)
    print(f"設備: {device}")
    print(f"資料集: {args.dataset}")
    print(f"模型類型: {args.model_type}")
    print(f"隱藏層大小: {args.hidden_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"學習率: {args.learning_rate}")
    print(f"訓練輪數: {args.num_epochs}")

    # 創建輸出目錄
    output_dir = Path(args.output_dir) / args.dataset / args.model_type
    checkpoint_dir = output_dir / 'checkpoints'
    results_dir = output_dir / 'results'

    for d in [checkpoint_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 載入資料
    print("\n" + "-"*80)
    print("[步驟 1] 載入資料")
    print("-"*80)

    dataloaders = create_sentence_level_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq
    )

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    preprocessor = dataloaders['preprocessor']

    vocab_size = preprocessor.vocab_size
    num_classes = 3  # positive, negative, neutral

    print(f"訓練樣本: {len(train_loader.dataset)}")
    print(f"測試樣本: {len(test_loader.dataset)}")
    print(f"詞彙表大小: {vocab_size:,}")

    # 創建模型
    print("\n" + "-"*80)
    print("[步驟 2] 創建模型")
    print("-"*80)

    model = create_model(
        model_type=args.model_type,
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        num_classes=num_classes,
        dropout=args.dropout
    )

    model = model.to(device)
    model.print_model_info()

    # 計算類別權重（處理不平衡）
    print("\n" + "-"*80)
    print("[步驟 3] 計算類別權重")
    print("-"*80)

    label_counts = Counter()
    for batch in train_loader:
        labels = batch['label'].squeeze()
        if labels.dim() == 0:  # 單個樣本
            labels = labels.unsqueeze(0)
        label_counts.update(labels.numpy().tolist())

    print(f"類別分佈: {dict(label_counts)}")

    total_samples = sum(label_counts.values())
    class_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"類別權重: {class_weights.cpu().numpy()}")

    # 創建優化器和損失函數
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 學習率調度器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # 訓練模型
    print("\n" + "-"*80)
    print("[步驟 4] 訓練模型")
    print("-"*80)

    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1_macro': [],
        'test_f1_micro': []
    }

    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_start = time.time()

        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.model_type
        )

        # 評估
        test_loss, test_acc, test_f1_macro, test_f1_micro = evaluate(
            model, test_loader, criterion, device, args.model_type
        )

        epoch_time = time.time() - epoch_start

        # 記錄
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1_macro'].append(test_f1_macro)
        history['test_f1_micro'].append(test_f1_micro)

        # 列印進度
        print(f"Epoch {epoch+1:02d}/{args.num_epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} "
              f"F1-Macro: {test_f1_macro:.4f}")

        # 儲存最佳模型
        if test_f1_macro > best_f1:
            best_f1 = test_f1_macro
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_f1_macro': test_f1_macro,
                'test_acc': test_acc,
                'args': vars(args)
            }, checkpoint_dir / 'best_model.pt')

            print(f"  → 儲存最佳模型 (F1-Macro: {best_f1:.4f})")
        else:
            patience_counter += 1

        # 學習率衰減
        scheduler.step()

        # 早停
        if patience_counter >= args.patience:
            print(f"\n早停於 Epoch {epoch+1} (連續 {args.patience} 輪無改善)")
            break

    training_time = time.time() - start_time

    print(f"\n訓練完成！")
    print(f"  最佳 Epoch: {best_epoch}")
    print(f"  最佳 F1-Macro: {best_f1:.4f}")
    print(f"  總訓練時間: {training_time:.1f}s ({training_time/60:.1f}m)")

    # 載入最佳模型並最終評估
    print("\n" + "-"*80)
    print("[步驟 5] 最終評估")
    print("-"*80)

    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1_macro, test_f1_micro = evaluate(
        model, test_loader, criterion, device, args.model_type
    )

    print(f"\n測試集結果:")
    print(f"  準確度:     {test_acc:.4f}")
    print(f"  F1-Macro:   {test_f1_macro:.4f}")
    print(f"  F1-Micro:   {test_f1_micro:.4f}")

    # 儲存結果
    results = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'hidden_size': args.hidden_size,
        'num_lstm_layers': args.num_lstm_layers,
        'embedding_dim': args.embedding_dim,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'vocab_size': vocab_size,
        'train_samples': len(train_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }

    result_file = results_dir / 'experiment_result.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存到: {result_file}")

    print("\n" + "="*80)
    print(f"實驗完成: {args.model_type}")
    print("="*80)


if __name__ == '__main__':
    main()
