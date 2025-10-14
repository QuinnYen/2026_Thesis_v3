"""
Sentence-Level 資料集載入器

用於載入和處理轉換後的 sentence-level 資料
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.preprocessor import ABSAPreprocessor


class SentenceLevelDataset(Dataset):
    """Sentence-Level 情感分析資料集"""

    def __init__(self,
                 csv_path: str,
                 preprocessor: ABSAPreprocessor,
                 max_len: int = 100):
        """
        初始化資料集

        Args:
            csv_path: CSV 檔案路徑
            preprocessor: 預處理器（已經 fit）
            max_len: 最大序列長度
        """
        self.csv_path = csv_path
        self.preprocessor = preprocessor
        self.max_len = max_len

        # 載入資料
        self.data = pd.read_csv(csv_path)

        # 過濾掉無效的資料
        self.data = self.data[self.data['sentence_polarity'].notna()]

        # 將 conflict 映射為 neutral（3 類分類）
        self.polarity_map = {
            'positive': 2,
            'negative': 0,
            'neutral': 1,
            'conflict': 1  # conflict 視為 neutral
        }

        print(f"載入資料集: {csv_path}")
        print(f"  總樣本數: {len(self.data)}")
        self._print_statistics()

    def _print_statistics(self):
        """列印資料集統計資訊"""
        polarity_counts = self.data['sentence_polarity'].value_counts()
        print(f"  情感分佈:")
        for polarity, count in polarity_counts.items():
            percentage = (count / len(self.data) * 100) if len(self.data) > 0 else 0
            mapped_label = self.polarity_map.get(polarity, -1)
            print(f"    {polarity:10s} ({mapped_label}): {count:5d} ({percentage:5.2f}%)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        獲取單個樣本

        Returns:
            {
                'input_ids': torch.LongTensor,  # (seq_len,)
                'attention_mask': torch.FloatTensor,  # (seq_len,)
                'label': torch.LongTensor,  # scalar
                'sentence_id': str,
                'text': str
            }
        """
        row = self.data.iloc[idx]

        # 文本
        text = row['text']
        sentence_id = row['sentence_id']

        # 情感標籤
        polarity = row['sentence_polarity']
        label = self.polarity_map[polarity]

        # 預處理文本
        tokens = self.preprocessor.tokenize(self.preprocessor.clean_text(text))

        # 數值化
        token_ids = self.preprocessor.tokens_to_indices(tokens)

        # Truncation
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]

        # Padding
        seq_len = len(token_ids)
        padding_length = self.max_len - seq_len

        input_ids = token_ids + [0] * padding_length  # 0 is PAD_TOKEN index
        attention_mask = [1] * seq_len + [0] * padding_length

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.FloatTensor(attention_mask),
            'label': torch.LongTensor([label]),
            'sentence_id': sentence_id,
            'text': text
        }


def create_sentence_level_dataloaders(
    dataset_name: str = 'SemEval2014_Restaurant',
    batch_size: int = 32,
    max_len: int = 100,
    min_freq: int = 2,
    data_dir: str = 'data/processed/sentence_level',
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    創建 Sentence-Level 資料載入器

    Args:
        dataset_name: 資料集名稱 ('SemEval2014_Restaurant' 或 'SemEval2014_Laptop')
        batch_size: 批次大小
        max_len: 最大序列長度
        min_freq: 最小詞頻
        data_dir: 資料目錄
        num_workers: 工作執行緒數

    Returns:
        {
            'train': DataLoader,
            'test': DataLoader,
            'preprocessor': ABSAPreprocessor
        }
    """
    print(f"\n{'='*80}")
    print(f"創建 Sentence-Level 資料載入器: {dataset_name}")
    print(f"{'='*80}")

    data_dir = Path(data_dir)

    # 檔案路徑
    train_csv = data_dir / f"{dataset_name}_train_sentence_level.csv"
    test_csv = data_dir / f"{dataset_name}_test_sentence_level.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"找不到訓練集檔案: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"找不到測試集檔案: {test_csv}")

    # 載入訓練資料並建立詞彙表
    print(f"\n載入訓練集並建立詞彙表...")
    train_df = pd.read_csv(train_csv)

    # 過濾無效資料
    train_df = train_df[train_df['sentence_polarity'].notna()]

    # 建立預處理器
    preprocessor = ABSAPreprocessor(min_freq=min_freq, lower=True)

    # 從訓練集建立詞彙表
    train_data = [{'text': text} for text in train_df['text'].tolist()]
    preprocessor.build_vocab(train_data)

    print(f"詞彙表大小: {preprocessor.vocab_size:,}")

    # 創建資料集
    train_dataset = SentenceLevelDataset(
        csv_path=str(train_csv),
        preprocessor=preprocessor,
        max_len=max_len
    )

    test_dataset = SentenceLevelDataset(
        csv_path=str(test_csv),
        preprocessor=preprocessor,
        max_len=max_len
    )

    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n資料載入器創建完成:")
    print(f"  訓練集批次數: {len(train_loader)}")
    print(f"  測試集批次數: {len(test_loader)}")
    print(f"  批次大小:     {batch_size}")
    print(f"  最大序列長度: {max_len}")
    print(f"{'='*80}\n")

    return {
        'train': train_loader,
        'test': test_loader,
        'preprocessor': preprocessor
    }


if __name__ == '__main__':
    # 測試資料載入器
    print("\n" + "="*80)
    print("測試 Sentence-Level 資料載入器")
    print("="*80)

    # 測試 Restaurant 資料集
    try:
        print("\n[1] 測試 Restaurant 資料集")
        dataloaders = create_sentence_level_dataloaders(
            dataset_name='SemEval2014_Restaurant',
            batch_size=32,
            max_len=100,
            min_freq=2
        )

        # 檢查一個批次
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))

        print(f"\n批次內容:")
        print(f"  input_ids shape:     {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  label shape:         {batch['label'].shape}")
        print(f"  label values:        {batch['label'].squeeze().tolist()[:5]}")

        print(f"\n範例句子:")
        for i in range(min(3, len(batch['text']))):
            print(f"  [{i}] {batch['text'][i][:80]}...")
            print(f"      Label: {batch['label'][i].item()}")

        print(f"\n[OK] Restaurant 資料集測試通過")

    except Exception as e:
        print(f"\n[FAIL] Restaurant 資料集測試失敗: {e}")

    # 測試 Laptop 資料集
    try:
        print("\n" + "-"*80)
        print("[2] 測試 Laptop 資料集")
        dataloaders = create_sentence_level_dataloaders(
            dataset_name='SemEval2014_Laptop',
            batch_size=32,
            max_len=100,
            min_freq=2
        )

        print(f"[OK] Laptop 資料集測試通過")

    except Exception as e:
        print(f"\n[FAIL] Laptop 資料集測試失敗: {e}")

    print("\n" + "="*80)
    print("測試完成！")
    print("="*80)
