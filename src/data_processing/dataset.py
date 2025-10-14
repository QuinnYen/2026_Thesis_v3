"""
PyTorch Dataset 和 DataLoader
提供標準的 PyTorch 資料介面
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import numpy as np

from .data_loader import SemEvalDataLoader
from .preprocessor import ABSAPreprocessor


class ABSADataset(Dataset):
    """
    ABSA PyTorch Dataset

    繼承 torch.utils.data.Dataset，提供標準的 PyTorch 資料介面
    """

    def __init__(self, processed_data: List[Dict]):
        """
        初始化 Dataset

        Args:
            processed_data: 預處理後的資料列表
        """
        self.data = processed_data

    def __len__(self) -> int:
        """返回資料集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        取得單個樣本

        Args:
            idx: 索引

        Returns:
            包含 PyTorch Tensor 的字典
        """
        item = self.data[idx]

        return {
            'input_ids': torch.LongTensor(item['input_ids']),
            'aspect_mask': torch.BoolTensor(item['aspect_mask']),
            'label': torch.LongTensor([item['label']]),
            'seq_length': item['seq_length']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    處理變長序列的批次化

    Args:
        batch: 批次資料列表

    Returns:
        批次化後的字典
    """
    # 取得批次中最大的序列長度
    max_len = max(item['input_ids'].size(0) for item in batch)
    batch_size = len(batch)

    # 取得 PAD token index（假設為 0）
    pad_idx = 0

    # 初始化批次 tensors
    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    aspect_masks = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    # 填充資料
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        aspect_masks[i, :seq_len] = item['aspect_mask']
        labels[i] = item['label']
        seq_lengths[i] = item['seq_length']

    return {
        'input_ids': input_ids,
        'aspect_mask': aspect_masks,
        'label': labels,
        'seq_length': seq_lengths
    }


def create_data_loaders(
    dataset_name: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 0,
    data_dir: str = 'data/raw',
    use_official_split: bool = False,
    min_freq: int = 2
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    建立 train/val/test DataLoaders

    Args:
        dataset_name: 資料集名稱（如 'SemEval2014_Laptop'）
        batch_size: 批次大小
        val_split: 驗證集比例（如果不使用官方分割）
        random_seed: 隨機種子
        num_workers: DataLoader 工作進程數
        data_dir: 資料目錄
        use_official_split: 是否使用官方測試集作為驗證集
        min_freq: 最小詞頻

    Returns:
        (train_loader, val_loader, test_loader)
        如果 use_official_split=False，test_loader 為 None
    """
    print("=" * 70)
    print(f" 建立 DataLoaders: {dataset_name}")
    print("=" * 70)
    print()

    # 載入資料
    print("[步驟 1] 載入資料")
    print("-" * 70)
    loader = SemEvalDataLoader(data_dir=data_dir)
    train_data = loader.load(dataset_name, 'train')
    print(f"訓練集樣本數: {len(train_data)}")

    # 建立預處理器
    print("\n[步驟 2] 建立預處理器並建立詞彙表")
    print("-" * 70)
    preprocessor = ABSAPreprocessor(min_freq=min_freq, lower=True)
    preprocessor.build_vocab(train_data)
    print()

    # 預處理訓練資料
    print("[步驟 3] 預處理資料")
    print("-" * 70)
    print("處理訓練集...")
    processed_train = preprocessor.preprocess_batch(train_data)
    print(f"完成：{len(processed_train)} 個樣本")

    # 分割訓練/驗證集
    if use_official_split:
        # 使用官方測試集作為驗證集
        print("\n[步驟 4] 使用官方測試集作為驗證集")
        print("-" * 70)
        try:
            test_data = loader.load(dataset_name, 'test', include_unlabeled=False)
            if len(test_data) > 0:
                print(f"測試集樣本數: {len(test_data)}")
                print("處理測試集...")
                processed_val = preprocessor.preprocess_batch(test_data)
                processed_test = None
                print(f"完成：{len(processed_val)} 個樣本")
            else:
                print("警告：官方測試集無標籤，改用訓練集分割")
                use_official_split = False
        except:
            print("警告：無法載入測試集，改用訓練集分割")
            use_official_split = False

    if not use_official_split:
        # 從訓練集分割
        print(f"\n[步驟 4] 從訓練集分割驗證集（比例: {val_split}）")
        print("-" * 70)
        train_indices, val_indices = train_test_split(
            range(len(processed_train)),
            test_size=val_split,
            random_state=random_seed,
            stratify=[item['label'] for item in processed_train]  # 保持標籤分佈
        )

        processed_train_split = [processed_train[i] for i in train_indices]
        processed_val = [processed_train[i] for i in val_indices]
        processed_train = processed_train_split
        processed_test = None

        print(f"訓練集: {len(processed_train)} 樣本")
        print(f"驗證集: {len(processed_val)} 樣本")

    # 建立 Datasets
    print("\n[步驟 5] 建立 PyTorch Datasets")
    print("-" * 70)
    train_dataset = ABSADataset(processed_train)
    val_dataset = ABSADataset(processed_val)
    test_dataset = ABSADataset(processed_test) if processed_test else None

    print(f"訓練 Dataset: {len(train_dataset)} 樣本")
    print(f"驗證 Dataset: {len(val_dataset)} 樣本")
    if test_dataset:
        print(f"測試 Dataset: {len(test_dataset)} 樣本")

    # 建立 DataLoaders
    print("\n[步驟 6] 建立 DataLoaders")
    print("-" * 70)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 訓練集打亂
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 驗證集不打亂
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    print(f"批次大小: {batch_size}")
    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    if test_loader:
        print(f"測試批次數: {len(test_loader)}")

    print("\n" + "=" * 70)
    print(" DataLoaders 建立完成！")
    print("=" * 70)
    print()

    return train_loader, val_loader, test_loader, preprocessor


def create_simple_data_loaders(
    preprocessor: ABSAPreprocessor,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    從已預處理的資料建立 DataLoaders（簡化版）

    Args:
        preprocessor: 預處理器
        train_data: 原始訓練資料
        val_data: 原始驗證資料（可選）
        batch_size: 批次大小
        val_split: 驗證集比例（如果 val_data 為 None）
        random_seed: 隨機種子
        num_workers: DataLoader 工作進程數

    Returns:
        (train_loader, val_loader)
    """
    # 預處理
    processed_train = preprocessor.preprocess_batch(train_data)

    # 分割或使用提供的驗證集
    if val_data is not None:
        processed_val = preprocessor.preprocess_batch(val_data)
    else:
        train_indices, val_indices = train_test_split(
            range(len(processed_train)),
            test_size=val_split,
            random_state=random_seed,
            stratify=[item['label'] for item in processed_train]
        )

        processed_train_split = [processed_train[i] for i in train_indices]
        processed_val = [processed_train[i] for i in val_indices]
        processed_train = processed_train_split

    # 建立 Datasets
    train_dataset = ABSADataset(processed_train)
    val_dataset = ABSADataset(processed_val)

    # 建立 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 簡單測試
    print("ABSADataset 和 DataLoader 測試")
    print("=" * 70)

    # 建立 DataLoaders
    train_loader, val_loader, test_loader, preprocessor = create_data_loaders(
        dataset_name='SemEval2014_Laptop',
        batch_size=16,
        val_split=0.2,
        use_official_split=False
    )

    # 測試迭代
    print("\n測試批次迭代:")
    print("-" * 70)
    for i, batch in enumerate(train_loader):
        print(f"\n批次 {i+1}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  aspect_mask shape: {batch['aspect_mask'].shape}")
        print(f"  label shape: {batch['label'].shape}")
        print(f"  seq_length shape: {batch['seq_length'].shape}")

        if i >= 2:  # 只顯示前 3 個批次
            break

    print("\n測試完成！")
