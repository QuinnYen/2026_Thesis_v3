"""
訓練 Baseline 模型（清理後資料）

配置:
- 模型: Baseline (BiLSTM + Attention)
- 資料: 清理後的數據集
- 支持: 從 configs/experiment_config.py 讀取配置
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import argparse
from tqdm import tqdm

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "models"))
sys.path.insert(0, str(BASE_DIR / "src" / "data_processing"))
sys.path.insert(0, str(BASE_DIR / "configs"))

from models.baseline import BaselineModel
from data_processing.cleaned_data_loader import create_cleaned_data_loaders
from experiment_config import DATASETS, BASELINE_CONFIG, get_output_dir, get_dataset_path

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EarlyStopping:
    """Early Stopping 機制"""

    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

        return self.early_stop


def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='訓練中', leave=False, ncols=100)

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        aspect_mask = batch['aspect_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, aspect_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(model, val_loader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(input_ids, aspect_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return avg_loss, acc, precision, recall, f1, all_preds, all_labels


def plot_training_curves(history, output_dir):
    """繪製訓練曲線"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Precision & Recall
    axes[1, 1].plot(history['val_precision'], label='Precision', color='blue')
    axes[1, 1].plot(history['val_recall'], label='Recall', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision and Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, preds, output_dir):
    """繪製混淆矩陣"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative', 'Neutral'],
                yticklabels=['Positive', 'Negative', 'Neutral'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='訓練 Baseline 模型')
    parser.add_argument('--dataset', type=str, default=None,
                       help='數據集名稱（預設：使用 configs/experiment_config.py 中的第一個數據集）')
    args = parser.parse_args()

    print("=" * 80)
    print("訓練 Baseline 模型（清理後資料）")
    print("=" * 80)

    # 設定隨機種子
    torch.manual_seed(42)
    np.random.seed(42)

    # 設定 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}\n")

    # 確定要訓練的數據集
    dataset_name = args.dataset if args.dataset else (DATASETS[0] if DATASETS else "SemEval2014_Restaurant")
    print(f"數據集: {dataset_name}\n")

    # 載入資料
    print("載入資料...")
    CSV_PATH = get_dataset_path(dataset_name, "train")

    train_loader, val_loader, vocab, label_map, class_weights = create_cleaned_data_loaders(
        train_csv=CSV_PATH,
        batch_size=BASELINE_CONFIG.get('batch_size', 32),
        val_split=0.2,
        max_length=BASELINE_CONFIG.get('max_length', 100),
        min_freq=2,
        random_seed=42
    )

    print(f"訓練樣本: {len(train_loader.dataset)}")
    print(f"驗證樣本: {len(val_loader.dataset)}")
    print(f"詞彙量: {len(vocab)}")
    print(f"類別: {label_map}\n")

    # 輸出目錄
    output_dir = get_output_dir(dataset_name, "baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 建立模型
    print("建立模型...")
    model = BaselineModel(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_size=128,
        num_classes=3,
        dropout=BASELINE_CONFIG.get('dropout', 0.3),
        pretrained_embeddings=None,
        freeze_embeddings=False
    )

    model = model.to(device)

    total_params, trainable_params = model.get_num_params()
    print(f"模型參數量: {total_params:,} (可訓練: {trainable_params:,})\n")

    # 訓練設定
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=BASELINE_CONFIG.get('learning_rate', 1e-3))
    early_stopping = EarlyStopping(patience=5, mode='max')

    # 訓練循環
    print("\n" + "=" * 80)
    print(f"開始訓練: {dataset_name}")
    print("=" * 80)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    best_f1 = 0
    best_epoch = 0
    num_epochs = BASELINE_CONFIG.get('num_epochs', 30)

    start_time = time.time()

    epoch_pbar = tqdm(range(1, num_epochs + 1), desc=f'{dataset_name} 訓練進度', ncols=120)

    for epoch in epoch_pbar:
        epoch_start = time.time()

        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 評估
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # 記錄
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        epoch_time = time.time() - epoch_start

        # 更新進度條
        epoch_pbar.set_postfix({
            'Loss': f'{val_loss:.4f}',
            'Acc': f'{val_acc:.4f}',
            'F1': f'{val_f1:.4f}'
        })

        tqdm.write(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.2f}s)")
        tqdm.write(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        tqdm.write(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        tqdm.write(f"  Val   P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")

        # 儲存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_model_path = checkpoint_dir / 'best_model.pt'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, best_model_path)

            tqdm.write(f"  [新最佳模型] F1: {val_f1:.4f} (已儲存)")

        # Early Stopping
        if early_stopping(val_f1):
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    tqdm.write(f"\n訓練完成！總時間: {total_time:.2f}s")
    tqdm.write(f"最佳 Epoch: {best_epoch} | 最佳 F1: {best_f1:.4f}")

    # 載入最佳模型並最終評估
    print("\n" + "=" * 80)
    print("最終評估（使用最佳模型）")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc, final_precision, final_recall, final_f1, final_preds, final_labels = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\n最終結果:")
    print(f"  Accuracy:  {final_acc:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall:    {final_recall:.4f}")
    print(f"  Macro-F1:  {final_f1:.4f}")

    # 分類報告
    report = classification_report(
        final_labels, final_preds,
        target_names=['Positive', 'Negative', 'Neutral'],
        digits=4,
        zero_division=0
    )
    print(f"\n分類報告:\n{report}")

    # 儲存結果
    results = {
        'dataset': dataset_name,
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'final_metrics': {
            'accuracy': final_acc,
            'precision': final_precision,
            'recall': final_recall,
            'macro_f1': final_f1
        },
        'training_time_seconds': total_time,
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'classification_report': report
    }

    with open(results_dir / 'experiment_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {results_dir / 'experiment_result.json'}")

    # 繪製圖表
    print("\n繪製圖表...")
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(final_labels, final_preds, output_dir)
    print(f"圖表已儲存至: {output_dir}")

    print("\n" + "=" * 80)
    print(f"[成功] {dataset_name} Baseline 訓練完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
