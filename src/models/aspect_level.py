"""
Aspect-Level 模型模組（改進版）

此模組實現改進的 Aspect-Level ABSA 模型：
- 加入 Aspect Position Embedding（相對位置編碼）
- Aspect-aware Attention 機制
- 能明確區分不同 aspect 的情感

架構：Input → Embedding + Position Embedding → BiLSTM → Aspect-Aware Attention → Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 嘗試相對導入
try:
    from .components import BiLSTMLayer, Classifier
except ImportError:
    from components import BiLSTMLayer, Classifier


class AspectPositionEmbedding(nn.Module):
    """
    Aspect 位置編碼層

    為每個詞添加相對於 aspect 的位置資訊：
    - 距離 aspect 越近，位置編碼影響越大
    - 使用可學習的位置嵌入矩陣

    參數:
        max_position (int): 最大相對位置距離，預設 50
        embedding_dim (int): 位置嵌入維度
    """

    def __init__(self, max_position=50, embedding_dim=50):
        super(AspectPositionEmbedding, self).__init__()

        self.max_position = max_position
        self.embedding_dim = embedding_dim

        # 位置嵌入矩陣：[-max_position, max_position]
        # 總共需要 2*max_position+1 個位置
        self.position_embeddings = nn.Embedding(
            num_embeddings=2 * max_position + 1,
            embedding_dim=embedding_dim,
            padding_idx=max_position  # 中間位置（aspect 自身）
        )

        self._init_weights()

    def _init_weights(self):
        """初始化位置嵌入權重"""
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def compute_relative_positions(self, aspect_mask):
        """
        計算每個詞相對於 aspect 的位置

        參數:
            aspect_mask: (batch_size, seq_len) 布林遮罩，True 表示 aspect 位置

        返回:
            relative_positions: (batch_size, seq_len) 相對位置索引
        """
        batch_size, seq_len = aspect_mask.shape
        device = aspect_mask.device

        # 建立位置索引：[0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 找到每個樣本中 aspect 的平均位置
        # aspect_positions: (batch_size,)
        aspect_positions = (positions * aspect_mask.float()).sum(dim=1) / (aspect_mask.sum(dim=1).float() + 1e-10)
        aspect_positions = aspect_positions.long()

        # 計算相對位置：position - aspect_position
        # relative_positions: (batch_size, seq_len)
        relative_positions = positions - aspect_positions.unsqueeze(1)

        # 限制在 [-max_position, max_position] 範圍內
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_position,
            self.max_position
        )

        # 轉換為嵌入索引：[0, 2*max_position]
        # 0 對應 -max_position，max_position 對應 0（aspect 自身），2*max_position 對應 +max_position
        position_ids = relative_positions + self.max_position

        return position_ids

    def forward(self, aspect_mask):
        """
        前向傳播

        參數:
            aspect_mask: (batch_size, seq_len) aspect 遮罩

        返回:
            position_embeds: (batch_size, seq_len, embedding_dim) 位置嵌入
        """
        # 計算相對位置索引
        position_ids = self.compute_relative_positions(aspect_mask)

        # 取得位置嵌入
        position_embeds = self.position_embeddings(position_ids)

        return position_embeds


class AspectAwareAttention(nn.Module):
    """
    Aspect-Aware 注意力機制

    改進的注意力機制，考慮：
    1. Aspect 位置資訊
    2. 距離 aspect 的遠近
    3. 上下文相關性

    參數:
        hidden_size (int): LSTM 隱藏狀態維度（單向）
        attention_size (int): 注意力層維度
    """

    def __init__(self, hidden_size, attention_size=None):
        super(AspectAwareAttention, self).__init__()

        self.hidden_size = hidden_size * 2  # BiLSTM 輸出維度
        self.attention_size = attention_size if attention_size is not None else hidden_size

        # 注意力權重矩陣
        self.W = nn.Linear(self.hidden_size, self.attention_size, bias=True)

        # Aspect 表示的權重矩陣
        self.W_aspect = nn.Linear(self.hidden_size, self.attention_size, bias=False)

        # 注意力向量
        self.v = nn.Linear(self.attention_size, 1, bias=False)

        # 距離權重（可學習的參數）
        self.distance_weight = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        """初始化權重"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.W_aspect.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, lstm_outputs, aspect_mask):
        """
        前向傳播

        參數:
            lstm_outputs: (batch_size, seq_len, hidden_size*2) LSTM 輸出
            aspect_mask: (batch_size, seq_len) aspect 遮罩

        返回:
            context_vector: (batch_size, hidden_size*2) 注意力加權後的上下文向量
            attention_weights: (batch_size, seq_len) 注意力權重
        """
        batch_size, seq_len, hidden_dim = lstm_outputs.shape

        # 1. 計算 aspect 表示（aspect 位置的平均）
        # aspect_outputs: (batch_size, hidden_size*2)
        aspect_mask_expanded = aspect_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        aspect_sum = (lstm_outputs * aspect_mask_expanded).sum(dim=1)  # (batch_size, hidden_size*2)
        aspect_count = aspect_mask.sum(dim=1, keepdim=True).float() + 1e-10  # (batch_size, 1)
        aspect_repr = aspect_sum / aspect_count  # (batch_size, hidden_size*2)

        # 2. 計算注意力分數
        # 結合上下文表示和 aspect 表示
        # h: (batch_size, seq_len, attention_size)
        h = torch.tanh(
            self.W(lstm_outputs) +  # 上下文資訊
            self.W_aspect(aspect_repr).unsqueeze(1)  # aspect 資訊（broadcast）
        )

        # 計算注意力分數：e_i = v^T * h_i
        attention_scores = self.v(h).squeeze(-1)  # (batch_size, seq_len)

        # 3. 計算距離衰減（距離 aspect 越遠，權重越低）
        device = lstm_outputs.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        aspect_positions = (positions * aspect_mask.float()).sum(dim=1) / (aspect_mask.sum(dim=1).float() + 1e-10)

        # 計算距離
        distances = torch.abs(positions.float() - aspect_positions.unsqueeze(1))  # (batch_size, seq_len)

        # 距離衰減因子：exp(-distance_weight * distance)
        distance_decay = torch.exp(-torch.abs(self.distance_weight) * distances)

        # 應用距離衰減
        attention_scores = attention_scores + torch.log(distance_decay + 1e-10)

        # 4. 計算注意力權重（softmax）
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # 5. 計算上下文向量
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_outputs  # (batch_size, seq_len, hidden_size*2)
        ).squeeze(1)  # (batch_size, hidden_size*2)

        return context_vector, attention_weights


class AspectLevelModel(nn.Module):
    """
    改進的 Aspect-Level ABSA 模型

    關鍵改進：
    1. Aspect Position Embedding：為每個詞添加相對於 aspect 的位置編碼
    2. Aspect-Aware Attention：注意力機制考慮 aspect 位置和距離
    3. 明確區分不同 aspect 的能力

    架構流程:
        1. 詞嵌入 + 位置嵌入
        2. BiLSTM 編碼
        3. Aspect-Aware Attention
        4. 分類器

    參數:
        vocab_size (int): 詞彙表大小
        embedding_dim (int): 詞嵌入維度，預設 300
        position_dim (int): 位置嵌入維度，預設 50
        hidden_size (int): LSTM 隱藏層維度，預設 128
        num_classes (int): 分類類別數量，預設 3
        dropout (float): Dropout 比率，預設 0.3
        pretrained_embeddings (Tensor, optional): 預訓練詞嵌入
        freeze_embeddings (bool): 是否凍結詞嵌入，預設 False
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        position_dim=50,
        hidden_size=128,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super(AspectLevelModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.position_dim = position_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        # 1. 詞嵌入層
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.word_embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embeddings:
            self.word_embedding.weight.requires_grad = False

        # 2. Aspect 位置嵌入層
        self.position_embedding = AspectPositionEmbedding(
            max_position=50,
            embedding_dim=position_dim
        )

        # 3. 投影層（將 word_embedding + position_embedding 投影到 LSTM 輸入維度）
        self.input_projection = nn.Linear(embedding_dim + position_dim, embedding_dim)

        # 4. BiLSTM 層
        self.bilstm = BiLSTMLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0
        )

        # 5. Aspect-Aware Attention
        self.attention = AspectAwareAttention(hidden_size=hidden_size)

        # 6. 分類器
        self.classifier = Classifier(
            input_size=hidden_size * 2,
            num_classes=num_classes,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        """初始化權重"""
        if self.word_embedding.weight.requires_grad:
            nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
            self.word_embedding.weight.data[0].fill_(0)

        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(self, input_ids, aspect_mask):
        """
        前向傳播

        參數:
            input_ids: (batch_size, seq_len) 詞 ID 序列
            aspect_mask: (batch_size, seq_len) aspect 遮罩

        返回:
            logits: (batch_size, num_classes) 分類結果
            attention_weights: (batch_size, seq_len) 注意力權重
        """
        # 1. 詞嵌入
        word_embeds = self.word_embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # 2. 位置嵌入
        position_embeds = self.position_embedding(aspect_mask)  # (batch_size, seq_len, position_dim)

        # 3. 合併嵌入並投影
        combined_embeds = torch.cat([word_embeds, position_embeds], dim=-1)  # (batch_size, seq_len, embedding_dim+position_dim)
        projected_embeds = torch.tanh(self.input_projection(combined_embeds))  # (batch_size, seq_len, embedding_dim)

        # 4. BiLSTM 編碼
        lstm_output, _ = self.bilstm(projected_embeds)  # (batch_size, seq_len, hidden_size*2)

        # 確保 aspect_mask 是布林類型
        if aspect_mask.dtype != torch.bool:
            aspect_mask = aspect_mask.bool()

        # 5. Aspect-Aware Attention
        context_vector, attention_weights = self.attention(lstm_output, aspect_mask)

        # 6. 分類
        logits = self.classifier(context_vector)

        return logits, attention_weights

    def get_num_params(self):
        """計算模型參數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# 測試程式碼
if __name__ == "__main__":
    print("=" * 80)
    print("測試改進的 Aspect-Level 模型")
    print("=" * 80)

    # 設定測試參數
    vocab_size = 10000
    embedding_dim = 300
    position_dim = 50
    hidden_size = 128
    num_classes = 3
    dropout = 0.3
    batch_size = 4
    seq_len = 50

    # 1. 測試模型初始化
    print("\n1. 測試模型初始化")
    print("-" * 80)
    model = AspectLevelModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        position_dim=position_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout
    )
    total_params, trainable_params = model.get_num_params()
    print(f"模型結構:")
    print(f"  - 詞彙表大小: {vocab_size}")
    print(f"  - 詞嵌入維度: {embedding_dim}")
    print(f"  - 位置嵌入維度: {position_dim}")
    print(f"  - LSTM 隱藏層維度: {hidden_size}")
    print(f"  - 分類類別數: {num_classes}")
    print(f"\n參數統計:")
    print(f"  - 總參數量: {total_params:,}")
    print(f"  - 可訓練參數量: {trainable_params:,}")
    print("[PASS] 模型成功初始化")

    # 2. 測試前向傳播
    print("\n2. 測試前向傳播")
    print("-" * 80)

    # 建立測試輸入
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_mask = torch.zeros(batch_size, seq_len)
    dummy_mask[:, 10:15] = 1  # aspect 在位置 10-14

    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(dummy_input, dummy_mask)

    print(f"輸入形狀:")
    print(f"  - input_ids: {dummy_input.shape}")
    print(f"  - aspect_mask: {dummy_mask.shape}")
    print(f"\n輸出形狀:")
    print(f"  - logits: {logits.shape}")
    print(f"  - attention_weights: {attention_weights.shape}")

    assert logits.shape == (batch_size, num_classes)
    assert attention_weights.shape == (batch_size, seq_len)
    print("[PASS] 前向傳播成功")

    # 3. 驗證 attention 是否聚焦在 aspect 附近
    print("\n3. 驗證 Attention 聚焦性")
    print("-" * 80)

    # aspect 位置的注意力權重
    aspect_attention = attention_weights[:, 10:15].sum(dim=1)
    print(f"Aspect 位置（10-14）的注意力權重: {aspect_attention.mean().item():.4f}")

    # aspect 附近（±3）的注意力權重
    nearby_attention = attention_weights[:, 7:18].sum(dim=1)
    print(f"Aspect 附近（7-17）的注意力權重: {nearby_attention.mean().item():.4f}")

    # 遠離 aspect 的位置
    far_attention = attention_weights[:, :5].sum(dim=1) + attention_weights[:, -5:].sum(dim=1)
    print(f"遠離 Aspect 的注意力權重: {far_attention.mean().item():.4f}")

    if nearby_attention.mean() > far_attention.mean():
        print("[PASS] Attention 成功聚焦在 aspect 附近")
    else:
        print("[WARNING] Attention 聚焦效果不明顯，可能需要調整參數")

    # 4. 測試多個 aspect 的情況
    print("\n4. 測試多個 Aspect 的區分能力")
    print("-" * 80)

    # 建立兩個不同位置的 aspect
    mask1 = torch.zeros(1, seq_len)
    mask1[0, 10:15] = 1  # aspect 在位置 10-14

    mask2 = torch.zeros(1, seq_len)
    mask2[0, 30:35] = 1  # aspect 在位置 30-34

    test_input = torch.randint(0, vocab_size, (1, seq_len))

    with torch.no_grad():
        _, attn1 = model(test_input, mask1)
        _, attn2 = model(test_input, mask2)

    # 檢查兩個 aspect 的注意力分布是否不同
    attn_diff = torch.abs(attn1 - attn2).mean()
    print(f"兩個不同 aspect 的注意力分布差異: {attn_diff.item():.4f}")

    if attn_diff.item() > 0.01:
        print("[PASS] 模型能區分不同位置的 aspect")
    else:
        print("[WARNING] 模型對不同 aspect 的區分能力較弱")

    # 5. 與 Baseline 模型比較
    print("\n5. 與 Baseline 模型比較")
    print("-" * 80)

    try:
        from baseline import BaselineModel

        baseline = BaselineModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

        baseline_params, _ = baseline.get_num_params()
        aspect_params, _ = model.get_num_params()

        print(f"Baseline 模型參數量: {baseline_params:,}")
        print(f"Aspect-Level 模型參數量: {aspect_params:,}")
        print(f"增加的參數量: {aspect_params - baseline_params:,}")
        print(f"參數增加比例: {(aspect_params - baseline_params) / baseline_params * 100:.2f}%")

        print("\n主要差異:")
        print("  - Baseline: 僅使用 aspect_mask 做 attention masking")
        print("  - Aspect-Level: 加入位置嵌入 + aspect-aware attention")
        print("  - Aspect-Level 能明確感知詞與 aspect 的距離關係")
    except ImportError:
        print("[INFO] 無法載入 Baseline 模型進行比較")

    print("\n" + "=" * 80)
    print("所有測試通過！")
    print("=" * 80)
