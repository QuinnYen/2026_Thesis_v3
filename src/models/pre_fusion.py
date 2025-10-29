"""
前融合架構（Pre-Fusion）模型 - 改進版

架構：Input → Embedding → Attention → Gate Fusion → BiLSTM(N層) → Classifier

改進重點（v2.0）：
- 使用 Gate Mechanism 智能融合原始 embedding 和 aspect context
- 不再將序列壓縮成單一向量，保留序列結構和多樣性
- BiLSTM 能學習到有意義的時序模式

特點：
- Attention 機制先作用在詞嵌入層輸出，提取 aspect 相關資訊
- Gate 機制動態決定每個位置保留多少原始資訊 vs. aspect 資訊
- 多層 BiLSTM 對融合後的序列進行深層編碼
- 取最後一層 BiLSTM 的最終隱藏狀態作為句子表示
- 適合先聚焦重要詞彙再進行深層編碼
"""

import torch
import torch.nn as nn
from .components import BiLSTMLayer, AttentionLayer, Classifier


class PreFusionModel(nn.Module):
    """
    前融合架構的 ABSA 模型

    架構流程：
    1. Embedding Layer: 將詞索引轉換為詞向量
    2. Attention Layer: 對詞嵌入計算注意力權重，獲得加權表示
    3. Multi-layer BiLSTM: 對 attention 輸出進行深層序列編碼
    4. Final State Pooling: 取最後一層 BiLSTM 的最終隱藏狀態
    5. Classifier: 全連接層 + Softmax 進行情感分類

    與後融合（Post-Fusion）的關鍵差異：
    - 後融合：BiLSTM → Attention（先編碼再聚焦）
    - 前融合：Attention → BiLSTM（先聚焦再編碼）

    參數：
        vocab_size (int): 詞彙表大小
        embedding_dim (int): 詞嵌入維度，默認 300
        hidden_size (int): LSTM 隱藏層大小，默認 128
        num_lstm_layers (int): LSTM 層數（2/3/4/5），默認 2
        num_classes (int): 分類類別數，默認 3（正面、負面、中性）
        dropout (float): Dropout 比例，默認 0.3
        pretrained_embeddings (torch.Tensor, optional): 預訓練詞嵌入矩陣
        freeze_embeddings (bool): 是否凍結詞嵌入層，默認 False
        bidirectional (bool): 是否使用雙向 LSTM，默認 True
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_lstm_layers=2,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        bidirectional=True
    ):
        super(PreFusionModel, self).__init__()

        # 驗證層數
        if num_lstm_layers < 2 or num_lstm_layers > 5:
            raise ValueError(f"num_lstm_layers 必須在 2-5 之間，但得到 {num_lstm_layers}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # 1. 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # 2. Attention 層（作用在詞嵌入層輸出）
        # 注意：這裡 AttentionLayer 會處理 embedding_dim，不是 hidden_size
        # 但 AttentionLayer 設計時假設輸入是 BiLSTM 的雙向輸出
        # 所以我們需要適配一下：這裡直接使用 embedding_dim 作為輸入維度
        # 為了與 AttentionLayer 兼容，我們傳入 embedding_dim // 2
        # 但這樣不太合理，所以我們創建一個自定義的 attention
        self.attention_W = nn.Linear(embedding_dim, embedding_dim // 2, bias=True)
        self.attention_v = nn.Linear(embedding_dim // 2, 1, bias=False)

        # 初始化 attention 參數
        nn.init.xavier_uniform_(self.attention_W.weight)
        nn.init.zeros_(self.attention_W.bias)
        nn.init.xavier_uniform_(self.attention_v.weight)

        # 2.5 Gate mechanism（用於融合原始 embedding 和 context）
        # 這是關鍵改進：不是直接用 context 替換，而是智能融合
        self.gate_linear = nn.Linear(embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)

        # 3. LSTM 層（輸入是 embedding_dim，支援單向/雙向）
        self.bilstm = BiLSTMLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # 4. 分類器（輸入是 LSTM 最終狀態：根據 bidirectional 決定）
        self.classifier = Classifier(
            input_size=self.lstm_output_size,
            num_classes=num_classes,
            dropout=dropout
        )

    def compute_attention(self, embeddings, aspect_mask):
        """
        計算 attention 權重並加權求和

        參數：
            embeddings (torch.Tensor): 詞嵌入 [batch_size, seq_len, embedding_dim]
            aspect_mask (torch.Tensor): Aspect 位置掩碼 [batch_size, seq_len]

        返回：
            context_vector (torch.Tensor): 加權上下文向量 [batch_size, embedding_dim]
            attention_weights (torch.Tensor): 注意力權重 [batch_size, seq_len]
        """
        # 計算注意力分數: e_i = v^T * tanh(W * x_i + b)
        attention_scores = self.attention_v(torch.tanh(self.attention_W(embeddings)))
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]

        # 在非 aspect 位置使用 -inf 掩碼
        attention_scores = attention_scores.masked_fill(~aspect_mask, float('-inf'))

        # Softmax 計算注意力權重（只在 aspect 位置）
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len]

        # 加權求和得到上下文向量
        # embeddings: [batch_size, seq_len, embedding_dim]
        # attention_weights: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            embeddings  # [batch_size, seq_len, embedding_dim]
        ).squeeze(1)  # [batch_size, embedding_dim]

        return context_vector, attention_weights

    def forward(self, input_ids, aspect_mask):
        """
        前向傳播

        參數：
            input_ids (torch.Tensor): 輸入詞索引 [batch_size, seq_len]
            aspect_mask (torch.Tensor): Aspect 位置掩碼 [batch_size, seq_len]
                                        True 表示 aspect 詞，False 表示非 aspect 詞

        返回：
            logits (torch.Tensor): 分類 logits [batch_size, num_classes]
            attention_weights (torch.Tensor): 注意力權重 [batch_size, seq_len]
        """
        batch_size = input_ids.size(0)

        # 確保 aspect_mask 是布林類型
        if aspect_mask.dtype != torch.bool:
            aspect_mask = aspect_mask.bool()

        # 1. 詞嵌入
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # 2. Attention 機制（作用在詞嵌入層）
        context_vector, attention_weights = self.compute_attention(embeddings, aspect_mask)
        # context_vector: [batch_size, embedding_dim]
        # attention_weights: [batch_size, seq_len]

        # 3. Aspect-Aware Enhancement（關鍵改進）
        # 不是直接替換，而是用 gate 機制融合原始 embedding 和 aspect context
        seq_len = input_ids.size(1)

        # 將 context_vector 擴展到序列長度
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        # context_expanded: [batch_size, seq_len, embedding_dim]

        # 計算 gate（決定保留多少原始資訊 vs. aspect 資訊）
        # gate 值在 [0, 1] 之間，由原始 embedding 決定
        gate = torch.sigmoid(self.gate_linear(embeddings))
        # gate: [batch_size, seq_len, embedding_dim]

        # 融合：保留原始序列結構 + 加入 aspect 資訊
        # gate=1: 完全保留原始 embedding
        # gate=0: 完全使用 aspect context
        # 實際上 gate 會學習在不同位置使用不同的混合比例
        enhanced_embeddings = gate * embeddings + (1 - gate) * context_expanded
        # enhanced_embeddings: [batch_size, seq_len, embedding_dim]
        # 優勢：保留了序列的多樣性，同時融入了 aspect 相關的全局資訊

        # 4. LSTM 編碼（現在輸入是有意義的序列）
        lstm_output, (h_n, c_n) = self.bilstm(enhanced_embeddings)
        # lstm_output: [batch_size, seq_len, lstm_output_size]
        # h_n: [num_layers * num_directions, batch_size, hidden_size]

        # 5. 取最後一層的最終隱藏狀態
        if self.bidirectional:
            # h_n 的形狀是 [num_layers * 2, batch_size, hidden_size]
            # 最後一層的前向和後向狀態在 h_n 的最後兩個位置
            forward_hidden = h_n[-2, :, :]  # [batch_size, hidden_size]
            backward_hidden = h_n[-1, :, :]  # [batch_size, hidden_size]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
            # final_hidden: [batch_size, hidden_size * 2]
        else:
            # 單向 LSTM：h_n 的形狀是 [num_layers, batch_size, hidden_size]
            # 取最後一層
            final_hidden = h_n[-1, :, :]  # [batch_size, hidden_size]

        # 6. 分類
        logits = self.classifier(final_hidden)  # [batch_size, num_classes]

        return logits, attention_weights

    def get_num_params(self):
        """
        計算模型參數量

        返回：
            total_params (int): 總參數量
            trainable_params (int): 可訓練參數量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_model_info(self):
        """
        獲取模型詳細信息

        返回：
            info (dict): 包含模型架構信息的字典
        """
        total_params, trainable_params = self.get_num_params()

        info = {
            'model_type': 'PreFusion',
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
            'bidirectional': self.bidirectional,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }

        return info

    def print_model_info(self):
        """
        打印模型信息
        """
        info = self.get_model_info()
        lstm_type = "雙向" if info['bidirectional'] else "單向"
        print("\n" + "="*70)
        print(f"模型類型: {info['model_type']} ({info['num_lstm_layers']} 層 {lstm_type}LSTM)")
        print("="*70)
        print(f"詞彙表大小:      {info['vocab_size']:,}")
        print(f"詞嵌入維度:      {info['embedding_dim']}")
        print(f"隱藏層大小:      {info['hidden_size']}")
        print(f"LSTM 層數:       {info['num_lstm_layers']}")
        print(f"LSTM 類型:       {lstm_type}")
        print(f"分類類別數:      {info['num_classes']}")
        print(f"Dropout:         {info['dropout']}")
        print("-"*70)
        print(f"總參數量:        {info['total_params']:,}")
        print(f"可訓練參數:      {info['trainable_params']:,}")
        print(f"凍結參數:        {info['frozen_params']:,}")
        print("="*70 + "\n")


# 便捷的工廠函數，用於創建不同層數的模型
def PreFusion_2Layer(**kwargs):
    """創建 2 層 BiLSTM 的前融合模型"""
    return PreFusionModel(num_lstm_layers=2, **kwargs)


def PreFusion_3Layer(**kwargs):
    """創建 3 層 BiLSTM 的前融合模型"""
    return PreFusionModel(num_lstm_layers=3, **kwargs)


def PreFusion_4Layer(**kwargs):
    """創建 4 層 BiLSTM 的前融合模型"""
    return PreFusionModel(num_lstm_layers=4, **kwargs)


def PreFusion_5Layer(**kwargs):
    """創建 5 層 BiLSTM 的前融合模型"""
    return PreFusionModel(num_lstm_layers=5, **kwargs)


# 測試代碼
if __name__ == '__main__':
    print("測試 PreFusionModel")
    print("="*70)

    # 測試參數
    vocab_size = 5000
    batch_size = 16
    seq_len = 50
    embedding_dim = 300
    hidden_size = 128
    num_classes = 4  # 負面、中性、正面、衝突

    # 測試不同層數的模型
    for num_layers in [2, 3, 4, 5]:
        print(f"\n測試 {num_layers} 層 BiLSTM 模型:")
        print("-"*70)

        # 創建模型
        model = PreFusionModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_lstm_layers=num_layers,
            num_classes=num_classes,
            dropout=0.3
        )

        # 打印模型信息
        model.print_model_info()

        # 創建測試數據
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        aspect_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        aspect_mask[:, 10:15] = True  # 假設 aspect 在位置 10-14

        # 前向傳播
        logits, attention_weights = model(input_ids, aspect_mask)

        # 驗證輸出形狀
        print(f"輸入形狀:")
        print(f"  input_ids:        {list(input_ids.shape)}")
        print(f"  aspect_mask:      {list(aspect_mask.shape)}")
        print(f"\n輸出形狀:")
        print(f"  logits:           {list(logits.shape)}")
        print(f"  attention_weights: {list(attention_weights.shape)}")

        # 驗證輸出形狀正確
        assert logits.shape == (batch_size, num_classes), \
            f"Logits 形狀錯誤: 期望 {(batch_size, num_classes)}, 實際 {logits.shape}"
        assert attention_weights.shape == (batch_size, seq_len), \
            f"Attention weights 形狀錯誤: 期望 {(batch_size, seq_len)}, 實際 {attention_weights.shape}"

        print(f"\n[OK] {num_layers} 層模型測試通過！")

    print("\n" + "="*70)
    print("測試便捷工廠函數:")
    print("-"*70)

    # 測試工廠函數
    models = [
        ('PreFusion_2Layer', PreFusion_2Layer),
        ('PreFusion_3Layer', PreFusion_3Layer),
        ('PreFusion_4Layer', PreFusion_4Layer),
        ('PreFusion_5Layer', PreFusion_5Layer)
    ]

    for name, factory_func in models:
        model = factory_func(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_classes=num_classes
        )
        total_params, _ = model.get_num_params()
        print(f"{name:25s} - 參數量: {total_params:,}")

    print("\n" + "="*70)
    print("所有測試通過！[OK]")
    print("="*70)
