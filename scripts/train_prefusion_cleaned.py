"""
Task 2.3: 訓練 Aspect Pre-Fusion 模型（清理後資料）

配置:
- 模型: Pre-Fusion (Attention → BiLSTM)
- 層數: 先測試 2 層，效果好再測試 3-5 層
- 資料: 清理後的 Restaurant 2014
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

from models.pre_fusion import PreFusionModel
from data_processing.cleaned_data_loader import create_cleaned_data_loaders
from experiment_config import DATASETS, PREFUSION_CONFIG, get_output_dir, get_dataset_path

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

    # 使用 tqdm 顯示進度條
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

        # 更新進度條
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 使用 tqdm 顯示驗證進度
        pbar = tqdm(val_loader, desc='驗證中', leave=False, ncols=100)

        for batch in pbar:
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
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def plot_training_curves(history, output_dir):
    """繪製訓練曲線"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    epochs = range(1, len(history['train_loss']) + 1)

    # 損失曲線
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='訓練損失', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='驗證損失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('訓練/驗證損失曲線', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 準確率曲線
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='訓練準確率', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='驗證準確率', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('訓練/驗證準確率曲線', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # F1 曲線
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='驗證 Macro-F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('驗證 Macro-F1 曲線', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Precision & Recall
    axes[1, 1].plot(epochs, history['val_precision'], 'b-', label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Precision & Recall 曲線', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Positive', 'Negative', 'Neutral'],
        yticklabels=['Positive', 'Negative', 'Neutral'],
        cbar_kws={'label': '樣本數'}
    )
    plt.xlabel('預測標籤', fontsize=12, fontweight='bold')
    plt.ylabel('真實標籤', fontsize=12, fontweight='bold')
    plt.title('混淆矩陣', fontsize=14, fontweight='bold')

    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f}',
            ha='center', transform=plt.gca().transAxes,
            fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_model(num_layers, train_loader, val_loader, class_weights, device, base_output_dir):
    """訓練指定層數的模型"""

    print("\n" + "=" * 80)
    print(f"訓練 Pre-Fusion {num_layers} 層模型")
    print("=" * 80)

    # 配置
    CONFIG = {
        'model_type': 'PreFusion',
        'num_lstm_layers': num_layers,
        'embedding_dim': 300,
        'hidden_size': 128,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'early_stopping_patience': 5,
        'use_class_weights': True
    }

    # 輸出目錄
    OUTPUT_DIR = base_output_dir / f"pre_fusion_{num_layers}layer_cleaned"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    RESULTS_DIR = OUTPUT_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    # 儲存配置
    with open(RESULTS_DIR / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print("\n配置:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # 建立模型
    print("\n建立模型...")

    # 取得詞彙表大小（從 train_loader 推斷）
    sample_batch = next(iter(train_loader))
    vocab_size = sample_batch['input_ids'].max().item() + 1

    model = PreFusionModel(
        vocab_size=vocab_size + 100,  # 加一些 buffer
        embedding_dim=CONFIG['embedding_dim'],
        hidden_size=CONFIG['hidden_size'],
        num_lstm_layers=num_layers,
        num_classes=3,
        dropout=CONFIG['dropout'],
        pretrained_embeddings=None,
        freeze_embeddings=False
    )

    model = model.to(device)
    total_params, trainable_params = model.get_num_params()
    print(f"模型參數量: {total_params:,} (可訓練: {trainable_params:,})")

    # 訓練設定
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'], mode='max')

    # 訓練循環
    print("\n" + "=" * 80)
    print("開始訓練")
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
    start_time = time.time()

    # 使用 tqdm 顯示 epoch 進度
    epoch_pbar = tqdm(range(1, CONFIG['num_epochs'] + 1), desc='總體進度', ncols=120)

    for epoch in epoch_pbar:
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        epoch_time = time.time() - epoch_start

        # 更新 epoch 進度條
        epoch_pbar.set_postfix({
            'Loss': f'{val_loss:.4f}',
            'Acc': f'{val_acc:.4f}',
            'F1': f'{val_f1:.4f}'
        })

        # 使用 tqdm.write() 避免打斷進度條
        tqdm.write(f"\nEpoch {epoch}/{CONFIG['num_epochs']} ({epoch_time:.2f}s)")
        tqdm.write(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        tqdm.write(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        tqdm.write(f"  Val   P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': CONFIG
            }, CHECKPOINT_DIR / 'best_model.pt')
            tqdm.write(f"  [新最佳模型] F1: {val_f1:.4f} (已儲存)")

        if early_stopping(val_f1):
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    tqdm.write(f"\n訓練完成！總時間: {total_time:.2f}s")
    tqdm.write(f"最佳 Epoch: {best_epoch} | 最佳 F1: {best_f1:.4f}")

    # 載入最佳模型並評估
    checkpoint = torch.load(CHECKPOINT_DIR / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc, final_precision, final_recall, final_f1, final_preds, final_labels = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\n最終結果:")
    print(f"  Accuracy:  {final_acc:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall:    {final_recall:.4f}")
    print(f"  Macro-F1:  {final_f1:.4f}")

    # 儲存結果
    results = {
        'num_layers': num_layers,
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'final_metrics': {
            'accuracy': final_acc,
            'precision': final_precision,
            'recall': final_recall,
            'macro_f1': final_f1
        },
        'training_time_seconds': total_time,
        'config': CONFIG
    }

    with open(RESULTS_DIR / 'experiment_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 繪製圖表
    plot_training_curves(history, OUTPUT_DIR)
    plot_confusion_matrix(final_labels, final_preds, OUTPUT_DIR)

    print(f"\n結果已儲存至: {OUTPUT_DIR}")

    return final_f1, final_acc


def main():
    parser = argparse.ArgumentParser(description='訓練 Pre-Fusion 模型')
    parser.add_argument('--dataset', type=str, default=None,
                       help='數據集名稱（預設：使用 configs/experiment_config.py 中的第一個數據集）')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='要訓練的層數（預設：使用 configs/experiment_config.py 中的配置）')
    parser.add_argument('--auto', action='store_true',
                       help='自動模式：如果 2 層 F1>0.68，則訓練 3-5 層')
    args = parser.parse_args()

    # 使用配置文件中的層數設定（如果命令行未指定）
    if args.layers is None:
        args.layers = PREFUSION_CONFIG.get('layers', [2])

    print("=" * 80)
    print("Task 2.3: 訓練 Aspect Pre-Fusion 模型（清理後資料）")
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
        batch_size=PREFUSION_CONFIG.get('batch_size', 32),
        val_split=0.2,
        max_length=PREFUSION_CONFIG.get('max_length', 100),
        min_freq=2,
        random_seed=42
    )

    base_output_dir = get_output_dir(dataset_name, "pre_fusion").parent

    # 儲存所有結果
    all_results = {}

    # 訓練指定層數的模型
    layers_to_train = args.layers

    for num_layers in layers_to_train:
        f1, acc = train_model(num_layers, train_loader, val_loader, class_weights, device, base_output_dir)
        all_results[num_layers] = {'f1': f1, 'accuracy': acc}

    # 自動模式：如果 2 層效果好，則訓練 3-5 層
    if args.auto and 2 in layers_to_train:
        if all_results[2]['f1'] > 0.68:
            print("\n" + "=" * 80)
            print(f"2 層模型 F1={all_results[2]['f1']:.4f} > 0.68")
            print("自動訓練 3-5 層模型...")
            print("=" * 80)

            for num_layers in [3, 4, 5]:
                if num_layers not in all_results:
                    f1, acc = train_model(num_layers, train_loader, val_loader, class_weights, device, base_output_dir)
                    all_results[num_layers] = {'f1': f1, 'accuracy': acc}
        else:
            print("\n" + "=" * 80)
            print(f"2 層模型 F1={all_results[2]['f1']:.4f} <= 0.68")
            print("不訓練更多層數")
            print("=" * 80)

    # 輸出總結
    print("\n" + "=" * 80)
    print("訓練總結")
    print("=" * 80)
    print(f"\n{'層數':<10} {'Accuracy':<15} {'Macro-F1':<15}")
    print("-" * 40)
    for num_layers in sorted(all_results.keys()):
        res = all_results[num_layers]
        print(f"{num_layers:<10} {res['accuracy']:<15.4f} {res['f1']:<15.4f}")

    # 找出最佳模型
    best_layers = max(all_results.keys(), key=lambda k: all_results[k]['f1'])
    print(f"\n最佳模型: {best_layers} 層 (F1={all_results[best_layers]['f1']:.4f})")

    print("\n" + "=" * 80)
    print("[成功] 所有訓練完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
