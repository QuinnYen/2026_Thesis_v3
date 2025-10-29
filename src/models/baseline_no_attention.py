"""
Baseline (無 Attention) 模型

專為維度1-A1實驗設計：測試不使用attention機制的基礎模型
架構：Input → Embedding → BiLSTM → Mean Pooling → Classifier

不同層數版本：
- 1層：單層 BiLSTM
- 2-5層：多層堆疊 BiLSTM
"""

import torch
import torch.nn as nn

try:
    from .components import BiLSTMLayer, Classifier
except ImportError:
    from components import BiLSTMLayer, Classifier


class BaselineNoAttention(nn.Module):
    """
    Baseline 模型（無 Attention）

    架構流程:
        1. 詞嵌入層：將詞 ID 轉換為詞向量
        2. BiLSTM 層：編碼序列的上下文信息（可多層）
        3. Mean Pooling：對序列取平均（不使用 Attention）
        4. 分類器：輸出情感類別機率

    參數:
        vocab_size (int): 詞彙表大小
        embedding_dim (int): 詞嵌入維度，預設 300
        hidden_size (int): LSTM 隱藏層維度，預設 128
        num_layers (int): LSTM 層數，預設 1
        num_classes (int): 分類類別數量，預設 3
        dropout (float): Dropout 比率，預設 0.3
        pretrained_embeddings (Tensor, optional): 預訓練的詞嵌入矩陣
        freeze_embeddings (bool): 是否凍結詞嵌入層，預設 False
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_layers=1,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super(BaselineNoAttention, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # 1. 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 如果提供預訓練的詞嵌入，則載入
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # 是否凍結詞嵌入層
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # 2. BiLSTM 層（可多層）
        self.bilstm = BiLSTMLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Mean Pooling（在 forward 中實現）
        # 不需要額外的層，直接對 BiLSTM 輸出取平均

        # 4. 分類器
        self.classifier = Classifier(
            input_size=hidden_size * 2,  # BiLSTM 輸出是雙向的
            num_classes=num_classes,
            dropout=dropout
        )

        # 參數初始化
        self._init_weights()

    def _init_weights(self):
        """初始化權重"""
        if self.embedding.weight.requires_grad:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.data[0].fill_(0)  # Padding token

    def forward(self, input_ids, aspect_mask=None):
        """
        前向傳播

        參數:
            input_ids (Tensor): 詞 ID 序列，形狀為 (batch_size, seq_len)
            aspect_mask (Tensor, optional): aspect 遮罩（此模型不使用，但保留接口一致性）

        返回:
            logits (Tensor): 情感類別機率分布，形狀為 (batch_size, num_classes)
            pooled_output (Tensor): 池化後的輸出（用於分析），形狀為 (batch_size, hidden_size*2)
        """
        batch_size, seq_len = input_ids.size()

        # 1. 詞嵌入
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # 2. BiLSTM 編碼
        lstm_output, _ = self.bilstm(embedded)  # (batch_size, seq_len, hidden_size*2)

        # 3. Mean Pooling - 對序列維度取平均
        # 創建 padding mask（input_ids != 0）
        padding_mask = (input_ids != 0).unsqueeze(-1)  # (batch_size, seq_len, 1)

        # 對非 padding 位置取平均
        masked_output = lstm_output * padding_mask.float()  # 遮蔽 padding
        sum_output = torch.sum(masked_output, dim=1)  # (batch_size, hidden_size*2)
        seq_lengths = torch.sum(padding_mask.float(), dim=1).clamp(min=1)  # 避免除以 0
        pooled_output = sum_output / seq_lengths  # (batch_size, hidden_size*2)

        # 4. 分類
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)

        return logits, pooled_output

    def get_num_params(self):
        """計算模型參數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# ============================================================================
# 測試程式碼
# ============================================================================

if __name__ == "__main__":
    # 測試模型
    print("測試 BaselineNoAttention 模型\n")

    # 測試不同層數
    for num_layers in [1, 2, 3]:
        print(f"--- 測試 {num_layers} 層 ---")

        model = BaselineNoAttention(
            vocab_size=5000,
            embedding_dim=300,
            hidden_size=128,
            num_layers=num_layers,
            num_classes=3,
            dropout=0.3
        )

        # 測試前向傳播
        batch_size = 4
        seq_len = 20
        input_ids = torch.randint(1, 5000, (batch_size, seq_len))
        aspect_mask = torch.zeros(batch_size, seq_len).bool()
        aspect_mask[:, 5:8] = True  # 模擬 aspect 位置

        logits, pooled = model(input_ids, aspect_mask)

        print(f"輸入形狀: {input_ids.shape}")
        print(f"輸出 logits 形狀: {logits.shape}")
        print(f"輸出 pooled 形狀: {pooled.shape}")

        total_params, trainable_params = model.get_num_params()
        print(f"總參數量: {total_params:,}")
        print(f"可訓練參數: {trainable_params:,}\n")
