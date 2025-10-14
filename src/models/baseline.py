"""
Baseline 模型模組

此模組實現基礎的 ABSA 模型：
架構：Input → Embedding → BiLSTM(1層) → Attention → Classifier
"""

import torch
import torch.nn as nn

# 嘗試相對導入，如果失敗則使用絕對導入
try:
    from .components import BiLSTMLayer, AttentionLayer, Classifier
except ImportError:
    from components import BiLSTMLayer, AttentionLayer, Classifier


class BaselineModel(nn.Module):
    """
    Baseline ABSA 模型

    此模型使用單層雙向 LSTM 和注意力機制進行面向詞感知的情感分析。

    架構流程:
        1. 詞嵌入層：將詞 ID 轉換為詞向量
        2. BiLSTM 層：編碼序列的上下文信息
        3. 注意力層：聚焦於面向詞相關的上下文
        4. 分類器：輸出情感類別機率

    參數:
        vocab_size (int): 詞彙表大小
        embedding_dim (int): 詞嵌入維度，預設 300（GloVe 維度）
        hidden_size (int): LSTM 隱藏層維度，預設 128
        num_classes (int): 分類類別數量，預設 3（負面、中性、正面）
        dropout (float): Dropout 比率，預設 0.3
        pretrained_embeddings (Tensor, optional): 預訓練的詞嵌入矩陣
        freeze_embeddings (bool): 是否凍結詞嵌入層，預設 False

    輸入形狀:
        - input_ids: (batch_size, seq_len) 詞 ID 序列
        - aspect_mask: (batch_size, seq_len) 面向詞遮罩，True 表示面向詞位置

    輸出形狀:
        - logits: (batch_size, num_classes) 情感類別機率分布
        - attention_weights: (batch_size, seq_len) 注意力權重（用於視覺化）
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super(BaselineModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
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

        # 2. BiLSTM 層（單層）
        self.bilstm = BiLSTMLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0  # 單層 LSTM 不使用 dropout
        )

        # 3. 注意力層
        self.attention = AttentionLayer(hidden_size=hidden_size)

        # 4. 分類器
        self.classifier = Classifier(
            input_size=hidden_size * 2,  # BiLSTM 輸出是雙向的
            num_classes=num_classes,
            dropout=dropout
        )

        # 參數初始化（embedding 層除外）
        self._init_weights()

    def _init_weights(self):
        """初始化詞嵌入層權重（如果沒有使用預訓練嵌入）"""
        # 如果詞嵌入不是預訓練的，則初始化為均勻分佈
        if self.embedding.weight.requires_grad:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            # Padding token 的嵌入設為 0
            self.embedding.weight.data[0].fill_(0)

    def forward(self, input_ids, aspect_mask):
        """
        前向傳播

        參數:
            input_ids (Tensor): 詞 ID 序列，形狀為 (batch_size, seq_len)
            aspect_mask (Tensor): 面向詞遮罩，形狀為 (batch_size, seq_len)
                                 True/1 表示面向詞位置，False/0 表示非面向詞位置

        返回:
            logits (Tensor): 情感類別機率分布，形狀為 (batch_size, num_classes)
            attention_weights (Tensor): 注意力權重，形狀為 (batch_size, seq_len)
        """
        # 1. 詞嵌入
        # input_ids: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # 2. BiLSTM 編碼
        # embedded: (batch_size, seq_len, embedding_dim)
        # lstm_output: (batch_size, seq_len, hidden_size * 2)
        lstm_output, _ = self.bilstm(embedded)

        # 確保 aspect_mask 是布林類型
        if aspect_mask.dtype != torch.bool:
            aspect_mask = aspect_mask.bool()

        # 3. 注意力機制
        # context_vector: (batch_size, hidden_size * 2)
        # attention_weights: (batch_size, seq_len)
        context_vector, attention_weights = self.attention(lstm_output, aspect_mask)

        # 4. 分類
        # logits: (batch_size, num_classes)
        logits = self.classifier(context_vector)

        return logits, attention_weights

    def get_num_params(self):
        """
        計算模型總參數量

        返回:
            total_params (int): 總參數量
            trainable_params (int): 可訓練參數量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# 測試程式碼
if __name__ == "__main__":
    """
    測試 Baseline 模型的功能和輸出形狀
    """
    print("="*50)
    print("測試 Baseline 模型")
    print("="*50)

    # 設定測試參數
    vocab_size = 10000
    embedding_dim = 300
    hidden_size = 128
    num_classes = 3
    dropout = 0.3
    batch_size = 4
    seq_len = 50

    # 1. 測試模型初始化
    print("\n1. 測試模型初始化")
    print("-" * 50)
    model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout
    )
    total_params, trainable_params = model.get_num_params()
    print(f"模型結構:")
    print(f"  - 詞彙表大小: {vocab_size}")
    print(f"  - 詞嵌入維度: {embedding_dim}")
    print(f"  - LSTM 隱藏層維度: {hidden_size}")
    print(f"  - 分類類別數: {num_classes}")
    print(f"  - Dropout 比率: {dropout}")
    print(f"\n參數統計:")
    print(f"  - 總參數量: {total_params:,}")
    print(f"  - 可訓練參數量: {trainable_params:,}")
    print("[PASS] 模型成功初始化")

    # 2. 測試使用預訓練嵌入
    print("\n2. 測試使用預訓練嵌入")
    print("-" * 50)
    pretrained_embeddings = torch.randn(vocab_size, embedding_dim)
    model_pretrained = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=False
    )
    total_params_pt, trainable_params_pt = model_pretrained.get_num_params()
    print(f"使用預訓練嵌入（可微調）:")
    print(f"  - 總參數量: {total_params_pt:,}")
    print(f"  - 可訓練參數量: {trainable_params_pt:,}")

    model_frozen = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=True
    )
    total_params_frozen, trainable_params_frozen = model_frozen.get_num_params()
    print(f"\n使用預訓練嵌入（凍結）:")
    print(f"  - 總參數量: {total_params_frozen:,}")
    print(f"  - 可訓練參數量: {trainable_params_frozen:,}")
    print(f"  - 凍結參數量: {total_params_frozen - trainable_params_frozen:,}")
    print("[PASS] 預訓練嵌入載入成功")

    # 3. 測試前向傳播
    print("\n3. 測試前向傳播")
    print("-" * 50)

    # 建立測試輸入
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_mask = torch.zeros(batch_size, seq_len)
    dummy_mask[:, 10:15] = 1  # 假設第 10-14 個詞是面向詞

    # 前向傳播
    model.eval()  # 設為評估模式
    with torch.no_grad():
        logits, attention_weights = model(dummy_input, dummy_mask)

    print(f"輸入形狀:")
    print(f"  - input_ids: {dummy_input.shape}")
    print(f"  - aspect_mask: {dummy_mask.shape}")
    print(f"\n輸出形狀:")
    print(f"  - logits: {logits.shape}")
    print(f"  - attention_weights: {attention_weights.shape}")

    # 驗證輸出形狀
    assert logits.shape == (batch_size, num_classes), f"預期 logits 形狀為 {(batch_size, num_classes)}，實際為 {logits.shape}"
    assert attention_weights.shape == (batch_size, seq_len), f"預期 attention_weights 形狀為 {(batch_size, seq_len)}，實際為 {attention_weights.shape}"
    print("[PASS] 前向傳播無錯誤")
    print("[PASS] 輸出形狀正確")

    # 4. 驗證輸出特性
    print("\n4. 驗證輸出特性")
    print("-" * 50)

    # 驗證機率分布
    print(f"機率分布總和: {logits.sum(dim=1)}")
    assert torch.allclose(logits.sum(dim=1), torch.ones(batch_size), atol=1e-6), "機率分布總和應為 1"
    print("[PASS] 機率分布總和為 1")

    # 驗證注意力權重
    print(f"注意力權重總和: {attention_weights.sum(dim=1)}")
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6), "注意力權重總和應為 1"
    print("[PASS] 注意力權重總和為 1")

    # 檢查注意力權重是否主要集中在面向詞位置
    aspect_attention = attention_weights[:, 10:15].sum(dim=1)
    print(f"\n面向詞位置的注意力權重佔比: {aspect_attention.mean().item():.4f}")
    print(f"預期範圍: 接近 1.0（因為只在面向詞位置計算注意力）")

    # 5. 測試梯度回傳
    print("\n5. 測試梯度回傳")
    print("-" * 50)
    model.train()  # 設為訓練模式
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 建立假標籤
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    # 前向傳播
    logits, _ = model(dummy_input, dummy_mask)

    # 計算損失
    loss = criterion(logits, dummy_labels)

    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"損失值: {loss.item():.4f}")
    print("[PASS] 梯度回傳成功")

    print("\n" + "="*50)
    print("所有測試通過！")
    print("="*50)

    # 顯示模型摘要
    print("\n模型摘要:")
    print("="*50)
    print(model)
