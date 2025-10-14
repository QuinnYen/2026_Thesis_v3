"""
ABSA 資料預處理器
負責文本清理、分詞、詞彙表建立、數值化和批次處理
"""

import re
import pickle
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import numpy as np


class ABSAPreprocessor:
    """
    ABSA (Aspect-Based Sentiment Analysis) 資料預處理器

    功能：
    1. 文本清理和分詞
    2. 詞彙表建立和管理
    3. 文本數值化
    4. 面向詞遮罩生成
    5. 序列 Padding
    """

    # 特殊 token
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    # 情感標籤映射
    POLARITY_MAP = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
        'conflict': 1  # 衝突標籤映射為 neutral
    }

    def __init__(self,
                 min_freq: int = 2,
                 max_vocab_size: Optional[int] = None,
                 lower: bool = True):
        """
        初始化預處理器

        Args:
            min_freq: 最小詞頻（低於此頻率的詞會被替換為 UNK）
            max_vocab_size: 最大詞彙表大小（None 表示不限制）
            lower: 是否轉換為小寫
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.lower = lower

        # 詞彙表
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # 統計資訊
        self.vocab_size = 0
        self.is_fitted = False

    def clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            清理後的文本
        """
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)

        # 移除 HTML 標籤
        text = re.sub(r'<[^>]+>', '', text)

        # 處理縮寫（保留撇號）
        # 例如：don't -> do n't
        text = re.sub(r"n't", " n't", text)
        text = re.sub(r"'re", " 're", text)
        text = re.sub(r"'s", " 's", text)
        text = re.sub(r"'ll", " 'll", text)
        text = re.sub(r"'ve", " 've", text)
        text = re.sub(r"'m", " 'm", text)

        # 轉小寫
        if self.lower:
            text = text.lower()

        # 去除首尾空白
        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        分詞

        Args:
            text: 文本

        Returns:
            token 列表
        """
        # 清理文本
        text = self.clean_text(text)

        # 分離標點符號（提高 GloVe 覆蓋率）
        # 在標點前後添加空格
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # 移除多餘空白

        # 簡單的空格分詞
        tokens = text.split()

        return tokens

    def build_vocab(self, data: List[Dict]) -> None:
        """
        從資料建立詞彙表

        Args:
            data: 資料列表，每個元素包含 'text' 欄位
        """
        print("建立詞彙表...")

        # 統計詞頻
        for item in data:
            tokens = self.tokenize(item['text'])
            self.word_freq.update(tokens)

        # 過濾低頻詞
        if self.min_freq > 1:
            filtered_words = [
                word for word, freq in self.word_freq.items()
                if freq >= self.min_freq
            ]
        else:
            filtered_words = list(self.word_freq.keys())

        # 按詞頻排序
        sorted_words = sorted(
            filtered_words,
            key=lambda w: self.word_freq[w],
            reverse=True
        )

        # 限制詞彙表大小
        if self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size - 2]  # 保留空間給特殊 token

        # 建立映射（特殊 token 放在前面）
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1
        }

        for idx, word in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx

        # 反向映射
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.vocab_size = len(self.word2idx)
        self.is_fitted = True

        # 統計資訊
        total_words = sum(self.word_freq.values())
        filtered_count = len([w for w, f in self.word_freq.items() if f < self.min_freq])

        print(f"詞彙表建立完成！")
        print(f"  總詞數（token）: {total_words:,}")
        print(f"  唯一詞數: {len(self.word_freq):,}")
        print(f"  過濾低頻詞: {filtered_count:,} 個 (freq < {self.min_freq})")
        print(f"  詞彙表大小: {self.vocab_size:,}")
        print(f"  Top 10 常見詞: {self.word_freq.most_common(10)}")

    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        將 token 列表轉換為索引列表

        Args:
            tokens: token 列表

        Returns:
            索引列表
        """
        if not self.is_fitted:
            raise RuntimeError("請先呼叫 build_vocab() 建立詞彙表")

        unk_idx = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(token, unk_idx) for token in tokens]

    def get_aspect_mask(self,
                       tokens: List[str],
                       aspect: str,
                       aspect_position: List[int],
                       text: str) -> List[int]:
        """
        生成面向詞遮罩

        Args:
            tokens: token 列表
            aspect: 面向詞
            aspect_position: 字元級別的位置 [start, end]
            text: 原始文本

        Returns:
            遮罩列表，面向詞位置為 1，其他為 0
        """
        mask = [0] * len(tokens)

        # 清理面向詞和文本（保持一致）
        aspect_clean = self.clean_text(aspect)
        text_clean = self.clean_text(text)

        # 調整字元位置（因為清理可能改變位置）
        start_char, end_char = aspect_position

        # 如果原始文本和清理後不同，需要重新定位
        if text != text_clean:
            # 嘗試在清理後的文本中找到面向詞
            aspect_in_clean = text_clean.find(aspect_clean)
            if aspect_in_clean != -1:
                start_char = aspect_in_clean
                end_char = aspect_in_clean + len(aspect_clean)

        # 將字元位置轉換為 token 位置
        current_pos = 0
        for i, token in enumerate(tokens):
            token_start = text_clean.find(token, current_pos)
            token_end = token_start + len(token)

            # 檢查 token 是否在面向詞範圍內
            if token_start != -1:
                # 判斷重疊
                if not (token_end <= start_char or token_start >= end_char):
                    mask[i] = 1
                current_pos = token_end

        # 如果沒有找到，嘗試基於面向詞本身匹配
        if sum(mask) == 0:
            aspect_tokens = self.tokenize(aspect)
            # 在 tokens 中尋找連續匹配
            for i in range(len(tokens) - len(aspect_tokens) + 1):
                if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                    mask[i:i+len(aspect_tokens)] = [1] * len(aspect_tokens)
                    break

        return mask

    def encode_polarity(self, polarity: str) -> int:
        """
        將情感極性編碼為數字標籤

        Args:
            polarity: 情感極性字串

        Returns:
            數字標籤
        """
        return self.POLARITY_MAP.get(polarity.lower(), self.POLARITY_MAP['neutral'])

    def preprocess_item(self, item: Dict) -> Dict:
        """
        預處理單個資料項

        Args:
            item: 原始資料項，包含 text, aspect, polarity, aspect_position

        Returns:
            預處理後的資料，包含 input_ids, aspect_mask, label, seq_length
        """
        # 分詞
        tokens = self.tokenize(item['text'])

        # 轉換為索引
        input_ids = self.tokens_to_indices(tokens)

        # 生成面向詞遮罩
        aspect_mask = self.get_aspect_mask(
            tokens,
            item['aspect'],
            item['aspect_position'],
            item['text']
        )

        # 編碼標籤
        label = self.encode_polarity(item['polarity'])

        # 序列長度
        seq_length = len(input_ids)

        return {
            'input_ids': input_ids,
            'aspect_mask': aspect_mask,
            'label': label,
            'seq_length': seq_length,
            # 保留原始資訊用於除錯
            'text': item['text'],
            'aspect': item['aspect'],
            'tokens': tokens
        }

    def preprocess_batch(self, data: List[Dict]) -> List[Dict]:
        """
        預處理批次資料

        Args:
            data: 資料列表

        Returns:
            預處理後的資料列表
        """
        return [self.preprocess_item(item) for item in data]

    def pad_sequences(self,
                     batch: List[Dict],
                     max_length: Optional[int] = None,
                     padding_value: int = 0) -> Dict[str, np.ndarray]:
        """
        對批次資料進行 padding

        Args:
            batch: 預處理後的資料批次
            max_length: 最大長度（None 則使用批次中最長的序列）
            padding_value: padding 值

        Returns:
            包含 numpy 陣列的字典
        """
        # 決定最大長度
        if max_length is None:
            max_length = max(item['seq_length'] for item in batch)

        batch_size = len(batch)

        # 初始化陣列
        input_ids = np.full((batch_size, max_length), self.word2idx[self.PAD_TOKEN], dtype=np.int64)
        aspect_masks = np.zeros((batch_size, max_length), dtype=np.int64)
        labels = np.zeros(batch_size, dtype=np.int64)
        seq_lengths = np.zeros(batch_size, dtype=np.int64)

        # 填充資料
        for i, item in enumerate(batch):
            seq_len = min(item['seq_length'], max_length)
            input_ids[i, :seq_len] = item['input_ids'][:seq_len]
            aspect_masks[i, :seq_len] = item['aspect_mask'][:seq_len]
            labels[i] = item['label']
            seq_lengths[i] = seq_len

        return {
            'input_ids': input_ids,
            'aspect_mask': aspect_masks,
            'label': labels,
            'seq_length': seq_lengths
        }

    def get_statistics(self, processed_data: List[Dict]) -> Dict:
        """
        獲取預處理資料的統計資訊

        Args:
            processed_data: 預處理後的資料

        Returns:
            統計資訊字典
        """
        seq_lengths = [item['seq_length'] for item in processed_data]
        labels = [item['label'] for item in processed_data]

        label_counts = Counter(labels)

        stats = {
            'total_samples': len(processed_data),
            'avg_seq_length': np.mean(seq_lengths),
            'max_seq_length': np.max(seq_lengths),
            'min_seq_length': np.min(seq_lengths),
            'median_seq_length': np.median(seq_lengths),
            'label_distribution': dict(label_counts),
            'vocab_size': self.vocab_size
        }

        return stats

    def print_statistics(self, processed_data: List[Dict]) -> None:
        """列印統計資訊"""
        stats = self.get_statistics(processed_data)

        print("\n" + "=" * 70)
        print(" 預處理資料統計")
        print("=" * 70)
        print(f"總樣本數: {stats['total_samples']:,}")
        print(f"詞彙表大小: {stats['vocab_size']:,}")
        print(f"\n序列長度統計:")
        print(f"  平均: {stats['avg_seq_length']:.2f}")
        print(f"  最大: {stats['max_seq_length']}")
        print(f"  最小: {stats['min_seq_length']}")
        print(f"  中位數: {stats['median_seq_length']:.2f}")

        print(f"\n標籤分佈:")
        polarity_names = {v: k for k, v in self.POLARITY_MAP.items()}
        total = stats['total_samples']
        for label, count in sorted(stats['label_distribution'].items()):
            percentage = (count / total) * 100
            name = polarity_names.get(label, f'label_{label}')
            bar = "█" * int(percentage / 2)
            print(f"  {name:12s} ({label}): {count:5d} ({percentage:5.2f}%) {bar}")

        print("=" * 70)

    def save(self, filepath: str) -> None:
        """
        儲存預處理器狀態

        Args:
            filepath: 儲存路徑
        """
        state = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'max_vocab_size': self.max_vocab_size,
            'lower': self.lower,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"預處理器已儲存到: {filepath}")

    def load(self, filepath: str) -> None:
        """
        載入預處理器狀態

        Args:
            filepath: 載入路徑
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.word2idx = state['word2idx']
        self.idx2word = state['idx2word']
        self.word_freq = state['word_freq']
        self.vocab_size = state['vocab_size']
        self.min_freq = state['min_freq']
        self.max_vocab_size = state['max_vocab_size']
        self.lower = state['lower']
        self.is_fitted = state['is_fitted']

        print(f"預處理器已從 {filepath} 載入")
        print(f"  詞彙表大小: {self.vocab_size:,}")


if __name__ == "__main__":
    # 簡單測試
    print("ABSAPreprocessor 測試")
    print("=" * 70)

    # 建立預處理器
    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)

    # 測試資料
    test_data = [
        {
            'text': 'The food was great but the service was terrible.',
            'aspect': 'food',
            'polarity': 'positive',
            'aspect_position': [4, 8]
        },
        {
            'text': 'The food was great but the service was terrible.',
            'aspect': 'service',
            'polarity': 'negative',
            'aspect_position': [27, 34]
        }
    ]

    # 建立詞彙表
    preprocessor.build_vocab(test_data)

    # 預處理
    processed = preprocessor.preprocess_batch(test_data)

    # 顯示結果
    for item in processed:
        print(f"\n文本: {item['text']}")
        print(f"面向詞: {item['aspect']}")
        print(f"Tokens: {item['tokens']}")
        print(f"Input IDs: {item['input_ids']}")
        print(f"Aspect Mask: {item['aspect_mask']}")
        print(f"Label: {item['label']}")

    # Padding 測試
    padded = preprocessor.pad_sequences(processed)
    print(f"\nPadded shapes:")
    for key, value in padded.items():
        print(f"  {key}: {value.shape}")
