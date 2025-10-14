"""
Sentence-Level 情感分析模型

這些模型用於句子級別的情感分類，不使用 aspect 資訊。
包含三種架構：
1. BiLSTM_SentenceLevel: 基礎 BiLSTM（無 Attention）
2. BiLSTM_SentenceLevel_PreFusion: 前融合（Attention → BiLSTM）
3. BiLSTM_SentenceLevel_PostFusion: 後融合（BiLSTM → Attention）
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BiLSTM_SentenceLevel(nn.Module):
    """
    基礎 Sentence-Level BiLSTM 模型（無 Attention）

    架構：
    Embedding → BiLSTM → Mean Pooling → Classifier

    特點：
    - 不使用 aspect 資訊
    - 不使用 attention 機制
    - 使用 mean pooling 聚合 BiLSTM 輸出
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_size: int = 128,
                 num_lstm_layers: int = 2,
                 num_classes: int = 3,  # positive, negative, neutral
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        初始化模型

        Args:
            vocab_size: 詞彙表大小
            embedding_dim: 詞嵌入維度
            hidden_size: LSTM 隱藏層大小
            num_lstm_layers: LSTM 層數
            num_classes: 分類數量（預設 3: positive, negative, neutral）
            dropout: Dropout 比例
            bidirectional: 是否使用雙向 LSTM
        """
        super(BiLSTM_SentenceLevel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # 詞嵌入層
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # BiLSTM 層
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout 層
        self.dropout = nn.Dropout(dropout)

        # 分類器
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            input_ids: 輸入 token IDs, shape: (batch_size, seq_len)

        Returns:
            logits: 分類 logits, shape: (batch_size, num_classes)
        """
        # 詞嵌入: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # BiLSTM: (batch_size, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(embeddings)

        # Mean Pooling: (batch_size, hidden_size * 2)
        # 對序列維度取平均
        pooled = torch.mean(lstm_out, dim=1)

        # 分類: (batch_size, num_classes)
        logits = self.classifier(pooled)

        return logits

    def get_num_params(self) -> Tuple[int, int]:
        """
        計算模型參數量

        Returns:
            (總參數量, 可訓練參數量)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def print_model_info(self):
        """列印模型資訊"""
        total_params, trainable_params = self.get_num_params()

        print(f"\n{'='*60}")
        print(f"模型: BiLSTM_SentenceLevel (無 Attention)")
        print(f"{'='*60}")
        print(f"架構: Embedding → BiLSTM({self.num_lstm_layers}層) → Mean Pooling → Classifier")
        print(f"\n超參數:")
        print(f"  詞彙表大小:     {self.vocab_size:,}")
        print(f"  詞嵌入維度:     {self.embedding_dim}")
        print(f"  隱藏層大小:     {self.hidden_size}")
        print(f"  LSTM 層數:      {self.num_lstm_layers}")
        print(f"  分類數量:       {self.num_classes}")
        print(f"  Dropout:        {self.dropout_rate}")
        print(f"  雙向 LSTM:      {self.bidirectional}")
        print(f"\n參數量:")
        print(f"  總參數:         {total_params:,}")
        print(f"  可訓練參數:     {trainable_params:,}")
        print(f"{'='*60}\n")


class SentenceAttentionLayer(nn.Module):
    """
    Sentence-Level Attention 層

    計算句子中每個詞的重要性權重（不使用 aspect 資訊）
    """

    def __init__(self, hidden_size: int):
        """
        初始化 Attention 層

        Args:
            hidden_size: LSTM 隱藏層大小（如果是雙向則為 hidden_size * 2）
        """
        super(SentenceAttentionLayer, self).__init__()

        self.hidden_size = hidden_size

        # Attention 權重計算
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            hidden_states: LSTM 輸出, shape: (batch_size, seq_len, hidden_size)
            mask: Padding mask, shape: (batch_size, seq_len), 1 表示有效位置

        Returns:
            context_vector: 加權後的上下文向量, shape: (batch_size, hidden_size)
            attention_weights: Attention 權重, shape: (batch_size, seq_len)
        """
        # 計算 attention scores
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, 1)
        scores = self.v(torch.tanh(self.W(hidden_states)))
        scores = scores.squeeze(-1)  # (batch_size, seq_len)

        # 應用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 計算 attention 權重
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)

        # 計算加權上下文向量
        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_size) -> (batch_size, 1, hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)

        return context_vector, attention_weights


class BiLSTM_SentenceLevel_PreFusion(nn.Module):
    """
    前融合 Sentence-Level BiLSTM 模型（Attention → BiLSTM）

    架構：
    Embedding → Attention → BiLSTM → Mean Pooling → Classifier

    特點：
    - Attention 在 BiLSTM 之前
    - Attention 對詞嵌入進行加權
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_size: int = 128,
                 num_lstm_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """初始化模型"""
        super(BiLSTM_SentenceLevel_PreFusion, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # 詞嵌入層
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Attention 層（作用於詞嵌入）
        self.attention = SentenceAttentionLayer(embedding_dim)

        # BiLSTM 層
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout 層
        self.dropout = nn.Dropout(dropout)

        # 分類器
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            input_ids: 輸入 token IDs, shape: (batch_size, seq_len)
            padding_mask: Padding mask, shape: (batch_size, seq_len)

        Returns:
            logits: 分類 logits, shape: (batch_size, num_classes)
            attention_weights: Attention 權重, shape: (batch_size, seq_len)
        """
        # 詞嵌入: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # Attention: (batch_size, embedding_dim)
        context_vector, attention_weights = self.attention(embeddings, padding_mask)

        # 將 context vector 與原始 embeddings 結合
        # 方法：將 context vector 擴展並與 embeddings 相加
        context_expanded = context_vector.unsqueeze(1).expand_as(embeddings)
        attended_embeddings = embeddings + context_expanded

        # BiLSTM: (batch_size, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(attended_embeddings)

        # Mean Pooling: (batch_size, hidden_size * 2)
        pooled = torch.mean(lstm_out, dim=1)

        # 分類: (batch_size, num_classes)
        logits = self.classifier(pooled)

        return logits, attention_weights

    def get_num_params(self) -> Tuple[int, int]:
        """計算模型參數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def print_model_info(self):
        """列印模型資訊"""
        total_params, trainable_params = self.get_num_params()

        print(f"\n{'='*60}")
        print(f"模型: BiLSTM_SentenceLevel_PreFusion")
        print(f"{'='*60}")
        print(f"架構: Embedding → Attention → BiLSTM({self.num_lstm_layers}層) → Pooling → Classifier")
        print(f"\n超參數:")
        print(f"  詞彙表大小:     {self.vocab_size:,}")
        print(f"  詞嵌入維度:     {self.embedding_dim}")
        print(f"  隱藏層大小:     {self.hidden_size}")
        print(f"  LSTM 層數:      {self.num_lstm_layers}")
        print(f"  分類數量:       {self.num_classes}")
        print(f"  Dropout:        {self.dropout_rate}")
        print(f"\n參數量:")
        print(f"  總參數:         {total_params:,}")
        print(f"  可訓練參數:     {trainable_params:,}")
        print(f"{'='*60}\n")


class BiLSTM_SentenceLevel_PostFusion(nn.Module):
    """
    後融合 Sentence-Level BiLSTM 模型（BiLSTM → Attention）

    架構：
    Embedding → BiLSTM → Attention → Classifier

    特點：
    - Attention 在 BiLSTM 之後
    - Attention 對 LSTM 輸出進行加權
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_size: int = 128,
                 num_lstm_layers: int = 2,
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """初始化模型"""
        super(BiLSTM_SentenceLevel_PostFusion, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # 詞嵌入層
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # BiLSTM 層
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout 層
        self.dropout = nn.Dropout(dropout)

        # Attention 層（作用於 LSTM 輸出）
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = SentenceAttentionLayer(lstm_output_size)

        # 分類器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            input_ids: 輸入 token IDs, shape: (batch_size, seq_len)
            padding_mask: Padding mask, shape: (batch_size, seq_len)

        Returns:
            logits: 分類 logits, shape: (batch_size, num_classes)
            attention_weights: Attention 權重, shape: (batch_size, seq_len)
        """
        # 詞嵌入: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # BiLSTM: (batch_size, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        # Attention: (batch_size, hidden_size * 2)
        context_vector, attention_weights = self.attention(lstm_out, padding_mask)

        # 分類: (batch_size, num_classes)
        logits = self.classifier(context_vector)

        return logits, attention_weights

    def get_num_params(self) -> Tuple[int, int]:
        """計算模型參數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def print_model_info(self):
        """列印模型資訊"""
        total_params, trainable_params = self.get_num_params()

        print(f"\n{'='*60}")
        print(f"模型: BiLSTM_SentenceLevel_PostFusion")
        print(f"{'='*60}")
        print(f"架構: Embedding → BiLSTM({self.num_lstm_layers}層) → Attention → Classifier")
        print(f"\n超參數:")
        print(f"  詞彙表大小:     {self.vocab_size:,}")
        print(f"  詞嵌入維度:     {self.embedding_dim}")
        print(f"  隱藏層大小:     {self.hidden_size}")
        print(f"  LSTM 層數:      {self.num_lstm_layers}")
        print(f"  分類數量:       {self.num_classes}")
        print(f"  Dropout:        {self.dropout_rate}")
        print(f"\n參數量:")
        print(f"  總參數:         {total_params:,}")
        print(f"  可訓練參數:     {trainable_params:,}")
        print(f"{'='*60}\n")


# 工廠函數：方便創建不同層數的模型

def create_sentence_level_model(model_type: str,
                                 vocab_size: int,
                                 num_lstm_layers: int = 2,
                                 embedding_dim: int = 300,
                                 hidden_size: int = 128,
                                 num_classes: int = 3,
                                 dropout: float = 0.3):
    """
    工廠函數：創建 Sentence-Level 模型

    Args:
        model_type: 模型類型 ('base', 'pre_fusion', 'post_fusion')
        vocab_size: 詞彙表大小
        num_lstm_layers: LSTM 層數
        embedding_dim: 詞嵌入維度
        hidden_size: 隱藏層大小
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

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}")

    model_class = model_classes[model_type]

    return model_class(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == '__main__':
    # 測試模型
    print("="*80)
    print("測試 Sentence-Level 模型")
    print("="*80)

    vocab_size = 5000
    batch_size = 16
    seq_len = 50

    # 創建測試資料
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    padding_mask = torch.ones(batch_size, seq_len)
    padding_mask[:, 40:] = 0  # 模擬 padding

    # 測試三種模型
    models = {
        'Base (無 Attention)': BiLSTM_SentenceLevel(vocab_size=vocab_size),
        'Pre-Fusion': BiLSTM_SentenceLevel_PreFusion(vocab_size=vocab_size),
        'Post-Fusion': BiLSTM_SentenceLevel_PostFusion(vocab_size=vocab_size)
    }

    for name, model in models.items():
        print(f"\n測試模型: {name}")
        print("-"*80)
        model.print_model_info()

        # 前向傳播
        if 'Base' in name:
            logits = model(input_ids)
            print(f"輸入形狀:  {list(input_ids.shape)}")
            print(f"輸出形狀:  {list(logits.shape)}")
        else:
            logits, attn = model(input_ids, padding_mask)
            print(f"輸入形狀:      {list(input_ids.shape)}")
            print(f"Mask 形狀:     {list(padding_mask.shape)}")
            print(f"Logits 形狀:   {list(logits.shape)}")
            print(f"Attention 形狀: {list(attn.shape)}")

        print(f"[OK] 測試通過")

    print("\n" + "="*80)
    print("所有模型測試完成！")
    print("="*80)
