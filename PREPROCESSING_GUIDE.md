# ABSA 資料預處理指南

## 概覽

`ABSAPreprocessor` 是一個完整的 ABSA (Aspect-Based Sentiment Analysis) 資料預處理器，負責將原始文本轉換為模型可用的數值格式。

## 功能特性

### 1. 文本清理
- ✅ 小寫化
- ✅ 移除多餘空白
- ✅ 處理縮寫（don't -> do n't）
- ✅ 移除 HTML 標籤

### 2. 分詞（Tokenization）
- 基於空格的簡單分詞
- 保留標點符號
- 可擴展為使用更複雜的分詞器（如 NLTK, spaCy）

### 3. 詞彙表建立
- **特殊 Token**: `<PAD>` (索引 0), `<UNK>` (索引 1)
- **詞頻過濾**: 支援 `min_freq` 參數過濾低頻詞
- **大小限制**: 支援 `max_vocab_size` 參數限制詞彙表大小
- **統計資訊**: 自動顯示詞彙統計

### 4. 數值化
- 文本 → Token Indices
- 面向詞位置 → Token-level 遮罩
- 情感極性 → 數字標籤（0: negative, 1: neutral, 2: positive）

### 5. 面向詞遮罩
- 自動將字元級位置轉換為 Token 級位置
- 面向詞位置標記為 1，其他為 0
- 支援多詞面向詞（如 "battery life"）

### 6. Padding
- 批次序列對齊
- 支援動態長度（自動取批次最長）
- 支援固定長度（手動指定 max_length）
- PAD token 正確填充

## 使用範例

### 基本使用流程

```python
from src.data_processing.data_loader import SemEvalDataLoader
from src.data_processing.preprocessor import ABSAPreprocessor

# 1. 載入資料
loader = SemEvalDataLoader()
train_data = loader.load('SemEval2014_Laptop', 'train')

# 2. 建立預處理器
preprocessor = ABSAPreprocessor(
    min_freq=2,          # 最小詞頻
    max_vocab_size=None, # 不限制詞彙表大小
    lower=True           # 轉小寫
)

# 3. 建立詞彙表
preprocessor.build_vocab(train_data)

# 4. 預處理資料
processed_data = preprocessor.preprocess_batch(train_data)

# 5. 批次 Padding
batch = processed_data[:32]
padded_batch = preprocessor.pad_sequences(batch)
```

### 輸出格式

#### 單個樣本

```python
{
    'input_ids': [7, 373, 9, 31, 265, ...],      # Token 索引列表
    'aspect_mask': [0, 0, 0, 0, 0, 1, 1, ...],   # 面向詞遮罩
    'label': 2,                                   # 情感標籤
    'seq_length': 18,                             # 實際序列長度

    # 額外保留（用於除錯）
    'text': '原始文本',
    'aspect': '面向詞',
    'tokens': ['token', 'list', ...]
}
```

#### Padding 後的批次

```python
{
    'input_ids': np.ndarray,      # shape: (batch_size, max_length)
    'aspect_mask': np.ndarray,    # shape: (batch_size, max_length)
    'label': np.ndarray,          # shape: (batch_size,)
    'seq_length': np.ndarray      # shape: (batch_size,)
}
```

### 標籤映射

```python
POLARITY_MAP = {
    'negative': 0,
    'neutral': 1,
    'positive': 2,
    'conflict': 3
}
```

### 特殊 Token 索引

```python
PAD_TOKEN = '<PAD>'  # 索引 0
UNK_TOKEN = '<UNK>'  # 索引 1
```

## 參數配置建議

### 訓練完整模型

```python
preprocessor = ABSAPreprocessor(
    min_freq=2,              # 保留出現 2 次以上的詞
    max_vocab_size=15000,    # 限制詞彙表大小
    lower=True               # 小寫化
)
```

**預期結果**:
- 詞彙表大小: 5,000 - 15,000
- 平均序列長度: 15-25
- OOV 率: < 5%

### 快速實驗

```python
preprocessor = ABSAPreprocessor(
    min_freq=5,              # 更高的詞頻要求
    max_vocab_size=5000,     # 較小的詞彙表
    lower=True
)
```

**預期結果**:
- 詞彙表大小: 2,000 - 5,000
- 訓練速度更快
- 可能犧牲一些準確度

## 統計資訊範例

```
預處理資料統計
======================================================================
總樣本數: 6,051
詞彙表大小: 6,198

序列長度統計:
  平均: 18.49
  最大: 79
  最小: 1
  中位數: 17.00

標籤分佈:
  negative     (0):  1671 (27.62%)
  neutral      (1):  1093 (18.06%)
  positive     (2):  3151 (52.07%)
  conflict     (3):   136 ( 2.25%)
```

## 儲存和載入

### 儲存預處理器

```python
preprocessor.save('data/processed/preprocessor.pkl')
```

**儲存內容**:
- 詞彙表映射 (word2idx, idx2word)
- 詞頻統計
- 配置參數

### 載入預處理器

```python
preprocessor = ABSAPreprocessor()
preprocessor.load('data/processed/preprocessor.pkl')

# 現在可以直接用於預處理新資料
new_processed = preprocessor.preprocess_batch(new_data)
```

## 進階使用

### 處理測試資料

```python
# 訓練階段：建立詞彙表
train_preprocessor = ABSAPreprocessor(min_freq=2)
train_preprocessor.build_vocab(train_data)
train_preprocessor.save('preprocessor.pkl')

# 測試階段：載入詞彙表
test_preprocessor = ABSAPreprocessor()
test_preprocessor.load('preprocessor.pkl')
test_processed = test_preprocessor.preprocess_batch(test_data)
```

### 自定義批次處理

```python
# 固定長度 Padding（用於 LSTM）
padded = preprocessor.pad_sequences(batch, max_length=50)

# 動態長度 Padding（用於 Transformer）
padded = preprocessor.pad_sequences(batch)  # 自動取批次最長
```

### 查詢詞彙

```python
# Word -> Index
word_idx = preprocessor.word2idx.get('food', preprocessor.word2idx['<UNK>'])

# Index -> Word
word = preprocessor.idx2word[word_idx]

# 詞頻
freq = preprocessor.word_freq['food']
```

## 驗證檢查清單

使用 `verify_preprocessor.py` 驗證實作：

- ✅ 文本清理功能正常
- ✅ 分詞功能正常
- ✅ 詞彙表包含特殊 token
- ✅ 詞彙表大小在合理範圍（5000-15000）
- ✅ 數值化功能正常
- ✅ 標籤編碼正確（至少 3 類）
- ✅ 面向詞遮罩正確率 > 95%
- ✅ Padding 功能正常
- ✅ 序列長度正確記錄

## 常見問題

### Q: 詞彙表太小怎麼辦？

**A**: 降低 `min_freq` 參數：
```python
preprocessor = ABSAPreprocessor(min_freq=1)  # 保留所有詞
```

### Q: 記憶體不足？

**A**: 減小詞彙表大小：
```python
preprocessor = ABSAPreprocessor(max_vocab_size=5000)
```

### Q: 面向詞遮罩不準確？

**A**: 預處理器會自動處理字元位置到 token 位置的轉換，如果遇到問題：
1. 檢查原始資料的 `aspect_position` 是否正確
2. 查看 `tokens` 和 `aspect_mask` 欄位進行除錯

### Q: 如何處理未知詞？

**A**: 預處理器自動將未知詞映射到 `<UNK>` token（索引 1）：
```python
tokens = ['known', 'word', 'UNKNOWN']
indices = preprocessor.tokens_to_indices(tokens)
# 輸出: [123, 456, 1]  # UNKNOWN -> 1 (<UNK>)
```

## 性能優化建議

1. **批次處理**: 使用 `preprocess_batch()` 而非逐個處理
2. **快取**: 預處理後儲存結果，避免重複處理
3. **詞彙表大小**: 在準確度和效率間取得平衡（推薦 5000-10000）
4. **Padding 策略**:
   - 靜態模型（LSTM）: 使用固定長度
   - 動態模型（Transformer）: 使用動態長度

## 與模型整合

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ABSADataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'aspect_mask': torch.tensor(item['aspect_mask'], dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

# 使用
dataset = ABSADataset(processed_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 相關檔案

- 實作: [preprocessor.py](src/data_processing/preprocessor.py)
- 測試: [test_preprocessor.py](scripts/test_preprocessor.py)
- 驗證: [verify_preprocessor.py](scripts/verify_preprocessor.py)
