"""
ABSA 訓練器模組

此模組提供統一的模型訓練介面，包含：
- 訓練循環
- 驗證循環
- Early Stopping
- 檢查點管理
- 記錄與視覺化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


class ABSATrainer:
    """
    ABSA 模型訓練器

    提供完整的訓練流程，包括訓練循環、驗證、Early Stopping、
    檢查點管理和視覺化功能。

    參數:
        model (nn.Module): 要訓練的模型
        train_loader (DataLoader): 訓練資料載入器
        val_loader (DataLoader): 驗證資料載入器
        optimizer (optim.Optimizer, optional): 優化器，預設為 Adam
        criterion (nn.Module, optional): 損失函數，預設為 CrossEntropyLoss
        learning_rate (float): 學習率，預設 0.001
        max_grad_norm (float): 梯度裁剪的最大範數，預設 5.0
        device (str, optional): 訓練裝置，預設自動選擇
        patience (int): Early Stopping 的耐心值，預設 5
        save_dir (str): 檢查點儲存目錄，預設 'checkpoints'
        scheduler (optional): 學習率調度器，預設 None
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        learning_rate: float = 0.001,
        max_grad_norm: float = 5.0,
        device: Optional[str] = None,
        patience: int = 5,
        save_dir: str = 'checkpoints',
        scheduler: Optional[object] = None
    ):
        # 模型和資料
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 裝置設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # 優化器和損失函數
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # 訓練配置
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler

        # Early Stopping
        self.patience = patience
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        # 檢查點管理
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 訓練記錄
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }

        # 批次記錄
        self.batch_history = {
            'losses': [],
            'steps': []
        }

        print(f"訓練器初始化完成")
        print(f"  - 裝置: {self.device}")
        print(f"  - 優化器: {type(self.optimizer).__name__}")
        print(f"  - 學習率: {self.learning_rate}")
        print(f"  - 損失函數: {type(self.criterion).__name__}")
        print(f"  - 梯度裁剪: {self.max_grad_norm}")
        print(f"  - Early Stopping 耐心值: {self.patience}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        訓練一個 epoch

        參數:
            epoch (int): 當前 epoch 編號

        返回:
            Dict[str, float]: 包含訓練損失、準確度和 F1 的字典
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # 進度條
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [訓練]')

        for batch_idx, batch in enumerate(pbar):
            # 取得批次資料
            input_ids = batch['input_ids'].to(self.device)
            aspect_mask = batch['aspect_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向傳播
            logits, _ = self.model(input_ids, aspect_mask)

            # 計算損失
            loss = self.criterion(logits, labels)

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 更新參數
            self.optimizer.step()

            # 記錄
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 記錄批次損失
            step = epoch * len(self.train_loader) + batch_idx
            self.batch_history['losses'].append(loss.item())
            self.batch_history['steps'].append(step)

            # 更新進度條
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 計算 epoch 平均指標
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        驗證模型

        參數:
            epoch (int): 當前 epoch 編號

        返回:
            Dict[str, float]: 包含驗證損失、準確度、F1、精確度和召回率的字典
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # 進度條
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [驗證]')

        with torch.no_grad():
            for batch in pbar:
                # 取得批次資料
                input_ids = batch['input_ids'].to(self.device)
                aspect_mask = batch['aspect_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向傳播
                logits, _ = self.model(input_ids, aspect_mask)

                # 計算損失
                loss = self.criterion(logits, labels)

                # 記錄
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新進度條
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 計算指標
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, num_epochs: int, verbose: bool = True) -> Dict:
        """
        完整訓練流程

        參數:
            num_epochs (int): 訓練的 epoch 數量
            verbose (bool): 是否顯示詳細訊息

        返回:
            Dict: 訓練歷史記錄
        """
        print(f"\n{'='*60}")
        print(f"開始訓練")
        print(f"{'='*60}")
        print(f"訓練資料批次數: {len(self.train_loader)}")
        print(f"驗證資料批次數: {len(self.val_loader)}")
        print(f"總 Epoch 數: {num_epochs}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # 訓練一個 epoch
            train_metrics = self.train_epoch(epoch)

            # 驗證
            val_metrics = self.validate(epoch)

            # 更新學習率調度器
            if self.scheduler is not None:
                # ReduceLROnPlateau 需要傳遞監控的指標
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()

            # 記錄當前學習率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 記錄歷史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(current_lr)

            epoch_time = time.time() - epoch_start_time

            # 顯示訓練訊息
            if verbose:
                print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
                print(f"  訓練 - Loss: {train_metrics['loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}, "
                      f"F1: {train_metrics['f1']:.4f}")
                print(f"  驗證 - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, "
                      f"Precision: {val_metrics['precision']:.4f}, "
                      f"Recall: {val_metrics['recall']:.4f}")
                print(f"  學習率: {current_lr:.6f}")

            # Early Stopping 檢查
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                if verbose:
                    print(f"  *** 新的最佳驗證 F1: {self.best_val_f1:.4f} - 模型已儲存 ***")
            else:
                self.patience_counter += 1
                if verbose:
                    print(f"  Early Stopping 計數器: {self.patience_counter}/{self.patience}")

            # Early Stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly Stopping 觸發！最佳驗證 F1: {self.best_val_f1:.4f}")
                break

            print("-" * 60)

        # 訓練結束
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"訓練完成！")
        print(f"{'='*60}")
        print(f"總訓練時間: {total_time:.2f}s ({total_time/60:.2f} 分鐘)")
        print(f"最佳驗證 F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")

        # 儲存最後的模型
        self.save_checkpoint('last_model.pt', epoch, val_metrics)

        # 儲存訓練歷史
        self.save_history()

        # 繪製訓練曲線
        self.plot_training_curves()

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """
        儲存模型檢查點

        參數:
            filename (str): 檔案名稱
            epoch (int): 當前 epoch
            metrics (Dict[str, float]): 當前指標
        """
        checkpoint_path = self.save_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_val_f1': self.best_val_f1
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str) -> Dict:
        """
        載入模型檢查點

        參數:
            filename (str): 檔案名稱

        返回:
            Dict: 檢查點資訊
        """
        checkpoint_path = self.save_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"檢查點檔案不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint['history']
        self.best_val_f1 = checkpoint['best_val_f1']

        print(f"檢查點載入成功: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - 驗證 F1: {checkpoint['metrics']['f1']:.4f}")

        return checkpoint

    def save_history(self):
        """儲存訓練歷史到 JSON 檔案"""
        history_path = self.save_dir / 'training_history.json'

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        print(f"訓練歷史已儲存: {history_path}")

    def plot_training_curves(self):
        """
        繪製訓練曲線

        包括：
        1. 訓練和驗證損失
        2. 訓練和驗證準確度
        3. 訓練和驗證 F1
        4. 批次損失變化
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 1. 損失曲線
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='訓練損失', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='驗證損失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('訓練和驗證損失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 準確度曲線
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='訓練準確度', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='驗證準確度', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('訓練和驗證準確度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. F1 曲線
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', label='訓練 F1', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='驗證 F1', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('訓練和驗證 F1 分數')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 批次損失變化
        if len(self.batch_history['losses']) > 0:
            axes[1, 1].plot(self.batch_history['steps'], self.batch_history['losses'],
                           'g-', alpha=0.6, linewidth=1)
            axes[1, 1].set_xlabel('訓練步數')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('批次損失變化')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 儲存圖表
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"訓練曲線已儲存: {plot_path}")

        plt.close()


# 測試程式碼
if __name__ == "__main__":
    """
    測試訓練器功能
    """
    print("="*60)
    print("測試 ABSA 訓練器")
    print("="*60)

    # 建立假資料集
    from torch.utils.data import TensorDataset

    # 參數設定
    num_samples = 100
    vocab_size = 1000
    seq_len = 50
    batch_size = 16
    num_epochs = 3

    # 建立假資料
    train_input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    train_aspect_mask = torch.zeros(num_samples, seq_len)
    train_aspect_mask[:, 10:15] = 1
    train_labels = torch.randint(0, 3, (num_samples,))

    val_input_ids = torch.randint(0, vocab_size, (20, seq_len))
    val_aspect_mask = torch.zeros(20, seq_len)
    val_aspect_mask[:, 10:15] = 1
    val_labels = torch.randint(0, 3, (20,))

    # 建立 DataLoader
    train_dataset = TensorDataset(train_input_ids, train_aspect_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_aspect_mask, val_labels)

    # 自訂 collate_fn
    def collate_fn(batch):
        input_ids, aspect_mask, labels = zip(*batch)
        return {
            'input_ids': torch.stack(input_ids),
            'aspect_mask': torch.stack(aspect_mask),
            'label': torch.stack(labels)
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 建立模型（需要先導入）
    import sys
    sys.path.append('..')
    from models.baseline import BaselineModel

    model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=100,  # 使用較小的維度加快測試
        hidden_size=64,
        num_classes=3,
        dropout=0.3
    )

    # 建立訓練器
    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.001,
        max_grad_norm=5.0,
        patience=3,
        save_dir='test_checkpoints'
    )

    print("\n[PASS] 訓練器初始化成功")

    # 訓練模型
    print("\n" + "="*60)
    print("開始訓練測試")
    print("="*60)

    history = trainer.train(num_epochs=num_epochs)

    print("\n[PASS] 訓練完成")

    # 測試檢查點載入
    print("\n" + "="*60)
    print("測試檢查點載入")
    print("="*60)

    checkpoint = trainer.load_checkpoint('best_model.pt')
    print(f"[PASS] 檢查點載入成功")

    print("\n" + "="*60)
    print("所有測試通過！")
    print("="*60)
