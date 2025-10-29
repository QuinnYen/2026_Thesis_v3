"""
清理後資料載入器

專門用於載入 Task 1.2 清理後的 CSV 格式資料
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import numpy as np


class CleanedABSADataset(Dataset):
    """
    清理後的 ABSA Dataset

    從 CSV 檔案載入清理後的 aspect-level 資料
    """

    def __init__(self, csv_path, vocab, label_map, max_length=100):
        """
        初始化 Dataset

        Args:
            csv_path: CSV 檔案路徑
            vocab: 詞彙表字典 {word: id}
            label_map: 標籤映射 {polarity: id}
            max_length: 最大序列長度
        """
        self.csv_path = csv_path
        self.vocab = vocab
        self.label_map = label_map
        self.max_length = max_length

        # 載入資料
        self.samples = self._load_data()

        print(f"從 {csv_path} 載入了 {len(self.samples)} 個樣本")

    def _load_data(self):
        """載入 CSV 資料並轉換為樣本"""
        df = pd.read_csv(self.csv_path)

        # 按句子分組
        grouped = df.groupby('sentence_id')

        samples = []

        for sent_id, group in grouped:
            # 取得句子文本
            sentence_text = group.iloc[0]['sentence_text']

            # 對每個 aspect 建立一個樣本
            for _, row in group.iterrows():
                aspect_term = row['aspect_term']
                aspect_polarity = row['aspect_polarity']
                from_pos = row['from_position']
                to_pos = row['to_position']

                # 將文本轉換為 token（簡單的空格分割）
                tokens = sentence_text.split()

                # 建立 aspect mask（基於字符位置）
                aspect_mask = self._create_aspect_mask(
                    sentence_text, from_pos, to_pos
                )

                # 轉換為 ID
                input_ids = [self.vocab.get(token.lower(), self.vocab.get('<UNK>', 1))
                            for token in tokens]

                # 標籤
                label = self.label_map.get(aspect_polarity, -1)

                # 跳過 conflict 類別（清理後應該很少，但可能還有殘留）
                if label == 3 or label == -1:
                    continue

                samples.append({
                    'input_ids': input_ids,
                    'aspect_mask': aspect_mask,
                    'label': label,
                    'sentence_text': sentence_text,
                    'aspect_term': aspect_term,
                    'aspect_polarity': aspect_polarity
                })

        return samples

    def _create_aspect_mask(self, sentence, from_pos, to_pos):
        """
        建立 aspect mask

        基於字符位置找出對應的 token 位置
        """
        tokens = sentence.split()
        mask = []

        char_pos = 0
        for token in tokens:
            # 跳過前面的空格
            while char_pos < len(sentence) and sentence[char_pos] == ' ':
                char_pos += 1

            # 檢查 token 是否在 aspect 範圍內
            token_start = char_pos
            token_end = char_pos + len(token)

            # 如果 token 與 aspect 有重疊，標記為 True
            if not (token_end <= from_pos or token_start >= to_pos):
                mask.append(True)
            else:
                mask.append(False)

            char_pos = token_end

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 截斷或填充到 max_length
        input_ids = sample['input_ids'][:self.max_length]
        aspect_mask = sample['aspect_mask'][:self.max_length]

        # 記錄實際長度
        seq_length = len(input_ids)

        # Padding
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length
            aspect_mask = aspect_mask + [False] * padding_length

        return {
            'input_ids': torch.LongTensor(input_ids),
            'aspect_mask': torch.BoolTensor(aspect_mask),
            'label': torch.LongTensor([sample['label']]),
            'seq_length': seq_length
        }


def build_vocab_from_csv(csv_paths, min_freq=2):
    """
    從 CSV 檔案建立詞彙表

    Args:
        csv_paths: CSV 檔案路徑列表
        min_freq: 最小詞頻

    Returns:
        vocab: 詞彙表 {word: id}
    """
    word_counter = Counter()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # 統計詞頻
        for text in df['sentence_text'].unique():
            tokens = text.split()
            for token in tokens:
                word_counter[token.lower()] += 1

    # 建立詞彙表
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
    }

    # 加入高頻詞
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"詞彙表大小: {len(vocab)} (最小詞頻: {min_freq})")
    print(f"總詞數: {len(word_counter)}")

    return vocab


def get_vocab_words(vocab):
    """
    從 vocab 字典獲取詞彙列表（按 ID 排序）

    Args:
        vocab: 詞彙表字典 {word: id}

    Returns:
        vocab_words: 詞彙列表（按 ID 排序）
    """
    # 按 ID 排序，返回詞彙列表
    vocab_words = [''] * len(vocab)
    for word, idx in vocab.items():
        vocab_words[idx] = word
    return vocab_words


def compute_class_weights(train_dataset, num_classes=3):
    """
    計算類別權重（用於處理不平衡）

    Args:
        train_dataset: 訓練資料集
        num_classes: 類別數量

    Returns:
        class_weights: 類別權重 tensor
    """
    # 統計每個類別的樣本數
    label_counts = Counter()

    for i in range(len(train_dataset)):
        label = train_dataset[i]['label'].item()
        label_counts[label] += 1

    print("\n類別分佈:")
    for label in range(num_classes):
        count = label_counts.get(label, 0)
        print(f"  類別 {label}: {count} 樣本")

    # 計算權重：inverse frequency
    total_samples = len(train_dataset)
    class_weights = []

    for label in range(num_classes):
        count = label_counts.get(label, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights)

    print("\n類別權重:")
    for label, weight in enumerate(class_weights):
        print(f"  類別 {label}: {weight:.4f}")

    return class_weights


def create_cleaned_data_loaders(
    train_csv,
    batch_size=32,
    val_split=0.2,
    max_length=100,
    min_freq=2,
    random_seed=42
):
    """
    建立 DataLoader

    Args:
        train_csv: 訓練資料 CSV 路徑
        batch_size: 批次大小
        val_split: 驗證集比例
        max_length: 最大序列長度
        min_freq: 最小詞頻
        random_seed: 隨機種子

    Returns:
        train_loader, val_loader, vocab, label_map, class_weights
    """
    print("=" * 80)
    print("建立清理後資料的 DataLoader")
    print("=" * 80)

    # 1. 建立詞彙表
    print("\n1. 建立詞彙表...")
    vocab = build_vocab_from_csv([train_csv], min_freq=min_freq)

    # 2. 標籤映射（清理後只有 3 個類別）
    label_map = {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }

    # 3. 建立完整資料集
    print("\n2. 載入資料集...")
    full_dataset = CleanedABSADataset(
        csv_path=train_csv,
        vocab=vocab,
        label_map=label_map,
        max_length=max_length
    )

    # 4. 分割訓練集和驗證集
    print(f"\n3. 分割資料集（驗證集比例: {val_split}）...")
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")

    # 5. 計算類別權重
    print("\n4. 計算類別權重...")
    class_weights = compute_class_weights(train_dataset, num_classes=3)

    # 6. 建立 DataLoader
    print("\n5. 建立 DataLoader...")

    def collate_fn(batch):
        """批次化函數"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        aspect_mask = torch.stack([item['aspect_mask'] for item in batch])
        labels = torch.cat([item['label'] for item in batch])
        seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])

        return {
            'input_ids': input_ids,
            'aspect_mask': aspect_mask,
            'label': labels,
            'seq_length': seq_lengths
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows 相容性
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")

    print("\n" + "=" * 80)
    print("DataLoader 建立完成！")
    print("=" * 80)

    return train_loader, val_loader, vocab, label_map, class_weights


# 測試程式碼
if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).parent.parent.parent
    CSV_PATH = BASE_DIR / "data" / "processed" / "aspect_level_cleaned" / "SemEval2014_Restaurant_train_cleaned.csv"

    print("測試清理後資料載入器")
    print("=" * 80)

    if CSV_PATH.exists():
        train_loader, val_loader, vocab, label_map, class_weights = create_cleaned_data_loaders(
            train_csv=CSV_PATH,
            batch_size=4,
            val_split=0.2,
            max_length=50,
            min_freq=2
        )

        # 測試取得一個批次
        print("\n測試批次:")
        for batch in train_loader:
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  aspect_mask shape: {batch['aspect_mask'].shape}")
            print(f"  label shape: {batch['label'].shape}")
            print(f"  Labels: {batch['label']}")
            break

        print("\n[成功] 資料載入器測試通過！")
    else:
        print(f"[錯誤] 找不到檔案: {CSV_PATH}")
