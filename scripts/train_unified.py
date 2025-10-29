"""
統一訓練腳本 - 支援多維度實驗配置

此腳本整合了 Baseline、Pre-Fusion、Post-Fusion 三種模型的訓練邏輯，
支援在一次運行中完成多個維度的實驗：
- 多個數據集
- 多種模型類型
- 多種層數配置
- 多種軟遮罩配置
- 多種詞向量配置

使用範例：
    # 單一實驗
    python scripts/train_unified.py --dataset SemEval2014_Restaurant --model baseline

    # 批次實驗
    python scripts/train_unified.py \
        --datasets SemEval2014_Restaurant SemEval2014_Laptop \
        --models baseline prefusion postfusion \
        --layers 2 3 4 5 \
        --soft_mask_configs hard_mask soft_mask_normalized
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
from itertools import product
from typing import List, Dict, Optional, Tuple

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "models"))
sys.path.insert(0, str(BASE_DIR / "src" / "data_processing"))
sys.path.insert(0, str(BASE_DIR / "configs"))

from models.baseline import BaselineModel
from models.pre_fusion import PreFusionModel
from models.post_fusion import PostFusionModel
from data_processing.cleaned_data_loader import create_cleaned_data_loaders, get_vocab_words
from data_processing.embedding_loader import load_glove_embeddings
from experiment_config import (
    DATASETS, BASELINE_CONFIG, PREFUSION_CONFIG, POSTFUSION_CONFIG,
    SOFT_MASK_EXPERIMENTS, get_output_dir, get_dataset_path
)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    axes[0, 0].set_title('訓練/驗證損失曲線')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('訓練/驗證準確率曲線')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('驗證 F1 分數曲線')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Precision & Recall
    axes[1, 1].plot(history['val_precision'], label='Precision', color='blue')
    axes[1, 1].plot(history['val_recall'], label='Recall', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision 與 Recall 曲線')
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
    plt.title('混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_model(model_type: str, vocab_size: int, config: dict,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 num_layers: int = None):
    """
    建立模型實例

    Args:
        model_type: 模型類型 ("baseline", "prefusion", "postfusion")
        vocab_size: 詞彙表大小
        config: 模型配置字典
        pretrained_embeddings: 預訓練詞向量
        num_layers: LSTM 層數（僅用於 pre/post-fusion）

    Returns:
        model: 模型實例
    """
    if model_type == "baseline":
        model = BaselineModel(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 300),
            hidden_size=config.get('hidden_size', 128),
            num_classes=3,
            dropout=config.get('dropout', 0.3),
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=config.get('freeze_embeddings', False),
            use_soft_mask=config.get('use_soft_mask', False),
            penalty_weight=config.get('penalty_weight', 5.0),
            context_window=config.get('context_window', 2),
            normalize_penalty=config.get('normalize_penalty', False)
        )
    elif model_type == "prefusion":
        model = PreFusionModel(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 300),
            hidden_size=config.get('hidden_size', 128),
            num_lstm_layers=num_layers or 2,
            num_classes=3,
            dropout=config.get('dropout', 0.3),
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=config.get('freeze_embeddings', False)
        )
    elif model_type == "postfusion":
        model = PostFusionModel(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 300),
            hidden_size=config.get('hidden_size', 128),
            num_lstm_layers=num_layers or 2,
            num_classes=3,
            dropout=config.get('dropout', 0.3),
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=config.get('freeze_embeddings', False)
        )
    else:
        raise ValueError(f"未知的模型類型: {model_type}")

    return model


def train_single_experiment(
    dataset_name: str,
    model_type: str,
    train_loader,
    val_loader,
    vocab,
    class_weights,
    device,
    config: dict,
    num_layers: Optional[int] = None,
    soft_mask_config: Optional[str] = None
):
    """
    訓練單一實驗配置

    Args:
        dataset_name: 數據集名稱
        model_type: 模型類型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
        vocab: 詞彙表
        class_weights: 類別權重
        device: 計算裝置
        config: 配置字典
        num_layers: LSTM 層數（可選）
        soft_mask_config: 軟遮罩配置名稱（可選）

    Returns:
        results: 實驗結果字典
    """
    # 建立實驗標識
    exp_id = f"{model_type}"
    if num_layers is not None:
        exp_id += f"_{num_layers}layer"
    if soft_mask_config is not None:
        exp_id += f"_{soft_mask_config}"

    print("\n" + "=" * 80)
    print(f"訓練實驗: {dataset_name} - {exp_id}")
    print("=" * 80)

    # 如果有軟遮罩配置，合併到 config 中
    if soft_mask_config and soft_mask_config in SOFT_MASK_EXPERIMENTS:
        config = {**config, **SOFT_MASK_EXPERIMENTS[soft_mask_config]}

    # 載入預訓練詞向量（如果啟用）
    pretrained_embeddings = None
    if config.get('use_pretrained_embeddings', False):
        print("\n載入 GloVe 預訓練詞向量...")
        try:
            glove_dir = BASE_DIR / "data" / "embeddings"
            glove_embedding = load_glove_embeddings(
                embedding_dim=300,
                data_dir=str(glove_dir),
                use_cache=True
            )

            vocab_words = get_vocab_words(vocab)
            embedding_matrix, _ = glove_embedding.get_embedding_matrix(
                vocab=vocab_words,
                oov_strategy=config.get('oov_strategy', 'random'),
                seed=42
            )

            pretrained_embeddings = torch.FloatTensor(embedding_matrix)
            print(f"GloVe 詞向量載入完成！形狀: {pretrained_embeddings.shape}")
        except Exception as e:
            print(f"警告：無法載入 GloVe 詞向量: {e}")
            print("將使用隨機初始化的詞向量")

    # 建立輸出目錄
    if model_type == "baseline":
        if soft_mask_config:
            output_dir = BASE_DIR / "outputs" / "experiments" / "aspect_level_v2" / dataset_name / f"baseline_{soft_mask_config}"
        else:
            output_dir = get_output_dir(dataset_name, "baseline")
    else:
        output_dir = get_output_dir(dataset_name, model_type, layers=num_layers)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 建立模型
    print("\n建立模型...")
    model = create_model(
        model_type=model_type,
        vocab_size=len(vocab),
        config=config,
        pretrained_embeddings=pretrained_embeddings,
        num_layers=num_layers
    )
    model = model.to(device)

    total_params, trainable_params = model.get_num_params()
    print(f"模型參數量: {total_params:,} (可訓練: {trainable_params:,})")

    # 訓練設定
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))
    early_stopping = EarlyStopping(patience=5, mode='max')

    # 訓練循環
    print("\n開始訓練...")
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
    num_epochs = config.get('num_epochs', 30)
    start_time = time.time()

    epoch_pbar = tqdm(range(1, num_epochs + 1), desc=f'{exp_id} 訓練進度', ncols=120)

    for epoch in epoch_pbar:
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

        # 更新進度條
        postfix = {
            'Loss': f'{val_loss:.4f}',
            'Acc': f'{val_acc:.4f}',
            'F1': f'{val_f1:.4f}'
        }

        # 儲存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, checkpoint_dir / 'best_model.pt')

            postfix['Best'] = '✓'

        epoch_pbar.set_postfix(postfix)

        # Early Stopping
        if early_stopping(val_f1):
            print(f"\nEarly stopping 於 epoch {epoch}")
            break

    total_time = time.time() - start_time

    # 載入最佳模型並最終評估
    print("\n最終評估（使用最佳模型）...")
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

    # 儲存結果
    results = {
        'dataset': dataset_name,
        'model_type': model_type,
        'experiment_id': exp_id,
        'num_layers': num_layers,
        'soft_mask_config': soft_mask_config,
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
        'config': config
    }

    with open(results_dir / 'experiment_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 繪製圖表
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(final_labels, final_preds, output_dir)

    print(f"\n結果已儲存至: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='統一訓練腳本 - 支援多維度實驗')

    # 數據集配置
    parser.add_argument('--dataset', type=str, default=None,
                       help='單一數據集名稱')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='多個數據集名稱（優先於 --dataset）')

    # 模型配置
    parser.add_argument('--model', type=str, default=None,
                       choices=['baseline', 'prefusion', 'postfusion'],
                       help='單一模型類型')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       choices=['baseline', 'prefusion', 'postfusion'],
                       help='多個模型類型（優先於 --model）')

    # 層數配置（用於 pre/post-fusion）
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='LSTM 層數列表（例如：2 3 4 5）')

    # 軟遮罩配置（用於 baseline）
    parser.add_argument('--soft_mask_configs', type=str, nargs='+', default=None,
                       help='軟遮罩配置名稱列表（例如：hard_mask soft_mask_normalized）')

    # 其他配置
    parser.add_argument('--use_config_datasets', action='store_true',
                       help='使用 experiment_config.py 中配置的數據集')

    args = parser.parse_args()

    print("=" * 80)
    print("統一訓練腳本 - 多維度實驗訓練")
    print("=" * 80)

    # 確定要訓練的數據集
    if args.use_config_datasets:
        datasets = DATASETS
    elif args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        # 預設使用配置檔案中的第一個數據集
        datasets = [DATASETS[0]] if DATASETS else ["SemEval2014_Restaurant"]

    # 確定要訓練的模型
    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        # 預設訓練所有模型
        models = ['baseline', 'prefusion', 'postfusion']

    print(f"\n將訓練以下配置：")
    print(f"  數據集 ({len(datasets)}): {datasets}")
    print(f"  模型類型 ({len(models)}): {models}")
    if args.layers:
        print(f"  層數配置: {args.layers}")
    if args.soft_mask_configs:
        print(f"  軟遮罩配置: {args.soft_mask_configs}")

    # 設定隨機種子
    torch.manual_seed(42)
    np.random.seed(42)

    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用裝置: {device}\n")

    # 儲存所有實驗結果
    all_results = []

    # 遍歷所有數據集
    for dataset_name in datasets:
        print("\n" + "=" * 80)
        print(f"數據集: {dataset_name}")
        print("=" * 80)

        # 載入資料（所有模型共用同一份資料）
        print("\n載入資料...")
        csv_path = get_dataset_path(dataset_name, "train")

        train_loader, val_loader, vocab, label_map, class_weights = create_cleaned_data_loaders(
            train_csv=csv_path,
            batch_size=32,
            val_split=0.2,
            max_length=128,
            min_freq=2,
            random_seed=42
        )

        print(f"訓練樣本: {len(train_loader.dataset)}")
        print(f"驗證樣本: {len(val_loader.dataset)}")
        print(f"詞彙量: {len(vocab)}")

        # 訓練每個模型
        for model_type in models:
            if model_type == 'baseline':
                # Baseline 模型配置
                config = BASELINE_CONFIG.copy()

                # 如果指定了軟遮罩配置，訓練多個版本
                if args.soft_mask_configs:
                    for soft_mask_config in args.soft_mask_configs:
                        try:
                            results = train_single_experiment(
                                dataset_name=dataset_name,
                                model_type=model_type,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                vocab=vocab,
                                class_weights=class_weights,
                                device=device,
                                config=config,
                                soft_mask_config=soft_mask_config
                            )
                            all_results.append(results)
                        except Exception as e:
                            print(f"\n[錯誤] 實驗失敗: {dataset_name} - {model_type} - {soft_mask_config}")
                            print(f"錯誤訊息: {str(e)}")
                            continue
                else:
                    # 訓練單一 baseline 配置
                    try:
                        results = train_single_experiment(
                            dataset_name=dataset_name,
                            model_type=model_type,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            vocab=vocab,
                            class_weights=class_weights,
                            device=device,
                            config=config
                        )
                        all_results.append(results)
                    except Exception as e:
                        print(f"\n[錯誤] 實驗失敗: {dataset_name} - {model_type}")
                        print(f"錯誤訊息: {str(e)}")
                        continue

            elif model_type in ['prefusion', 'postfusion']:
                # Pre/Post-Fusion 模型配置
                config = PREFUSION_CONFIG.copy() if model_type == 'prefusion' else POSTFUSION_CONFIG.copy()

                # 確定要訓練的層數
                layers_list = args.layers if args.layers else config.get('layers', [2, 3, 4, 5])

                # 訓練每個層數配置
                for num_layers in layers_list:
                    try:
                        results = train_single_experiment(
                            dataset_name=dataset_name,
                            model_type=model_type,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            vocab=vocab,
                            class_weights=class_weights,
                            device=device,
                            config=config,
                            num_layers=num_layers
                        )
                        all_results.append(results)
                    except Exception as e:
                        print(f"\n[錯誤] 實驗失敗: {dataset_name} - {model_type} - {num_layers}層")
                        print(f"錯誤訊息: {str(e)}")
                        continue

    # 輸出總結
    print("\n" + "=" * 80)
    print("所有實驗完成！")
    print("=" * 80)

    if all_results:
        print(f"\n總共完成 {len(all_results)} 個實驗")
        print("\n實驗結果摘要：")
        print("-" * 80)
        print(f"{'數據集':<25} {'實驗ID':<30} {'F1':<10} {'Acc':<10}")
        print("-" * 80)

        for result in all_results:
            print(f"{result['dataset']:<25} {result['experiment_id']:<30} "
                  f"{result['final_metrics']['macro_f1']:<10.4f} "
                  f"{result['final_metrics']['accuracy']:<10.4f}")

        # 找出最佳模型
        best_result = max(all_results, key=lambda x: x['final_metrics']['macro_f1'])
        print("\n" + "=" * 80)
        print("🏆 最佳實驗：")
        print(f"  數據集: {best_result['dataset']}")
        print(f"  實驗ID: {best_result['experiment_id']}")
        print(f"  Macro-F1: {best_result['final_metrics']['macro_f1']:.4f}")
        print(f"  Accuracy: {best_result['final_metrics']['accuracy']:.4f}")
        print("=" * 80)
    else:
        print("\n[警告] 沒有成功完成的實驗")


if __name__ == "__main__":
    main()
