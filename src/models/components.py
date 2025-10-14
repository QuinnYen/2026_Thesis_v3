"""
模型基礎組件模組

此模組包含可重用的深度學習組件：
- BiLSTMLayer: 雙向 LSTM 層
- AttentionLayer: 加性注意力機制
- Classifier: 分類器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMLayer(nn.Module):
    """
    雙向 LSTM 層

    此層實現雙向 LSTM，可以捕捉序列的前向和後向上下文信息。

    參數:
        input_size (int): 輸入特徵的維度
        hidden_size (int): LSTM 隱藏狀態的維度
        num_layers (int): LSTM 層數
        dropout (float): Dropout 比率，應用於除最後一層外的所有層

    輸入形狀:
        - input: (batch_size, seq_len, input_size)

    輸出形狀:
        - sequence_output: (batch_size, seq_len, hidden_size * 2)
        - h_n: (num_layers * 2, batch_size, hidden_size)
        - c_n: (num_layers * 2, batch_size, hidden_size)
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(BiLSTMLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # 定義雙向 LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 參數初始化
        self._init_weights()

    def _init_weights(self):
        """初始化 LSTM 權重"""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                # 輸入到隱藏層的權重使用 Xavier 初始化
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 隱藏層到隱藏層的權重使用正交初始化
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 偏置初始化為 0，但遺忘門偏置初始化為 1
                param.data.fill_(0)
                # LSTM 的偏置分為四部分: input, forget, cell, output
                # 將遺忘門的偏置設為 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, input_seq):
        """
        前向傳播

        參數:
            input_seq (Tensor): 輸入序列，形狀為 (batch_size, seq_len, input_size)

        返回:
            sequence_output (Tensor): 所有時間步的輸出，形狀為 (batch_size, seq_len, hidden_size * 2)
            h_n (Tensor): 最後時間步的隱藏狀態，形狀為 (num_layers * 2, batch_size, hidden_size)
            c_n (Tensor): 最後時間步的細胞狀態，形狀為 (num_layers * 2, batch_size, hidden_size)
        """
        # LSTM 前向傳播
        sequence_output, (h_n, c_n) = self.bilstm(input_seq)

        return sequence_output, (h_n, c_n)


class AttentionLayer(nn.Module):
    """
    加性注意力機制層（Additive Attention）

    此層實現面向詞感知的注意力機制，只在面向詞位置計算注意力權重。

    注意力計算公式:
        e_i = v^T * tanh(W * h_i + b)
        α_i = softmax(e_i)  # 只在面向詞位置計算
        context = Σ(α_i * h_i)

    參數:
        hidden_size (int): LSTM 隱藏狀態的維度（雙向 LSTM 輸出維度的一半）
        attention_size (int): 注意力層的維度，預設與 hidden_size 相同

    輸入形狀:
        - lstm_outputs: (batch_size, seq_len, hidden_size * 2)
        - aspect_mask: (batch_size, seq_len) 布林遮罩，True 表示面向詞位置

    輸出形狀:
        - attention_output: (batch_size, hidden_size * 2)
        - attention_weights: (batch_size, seq_len)
    """

    def __init__(self, hidden_size, attention_size=None):
        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size * 2  # 雙向 LSTM 輸出維度
        self.attention_size = attention_size if attention_size is not None else hidden_size

        # 注意力權重矩陣 W
        self.W = nn.Linear(self.hidden_size, self.attention_size, bias=True)

        # 注意力向量 v
        self.v = nn.Linear(self.attention_size, 1, bias=False)

        # 參數初始化
        self._init_weights()

    def _init_weights(self):
        """初始化注意力層權重"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, lstm_outputs, aspect_mask):
        """
        前向傳播

        參數:
            lstm_outputs (Tensor): LSTM 輸出，形狀為 (batch_size, seq_len, hidden_size * 2)
            aspect_mask (Tensor): 面向詞遮罩，形狀為 (batch_size, seq_len)，
                                 True 表示面向詞位置，False 表示非面向詞位置

        返回:
            attention_output (Tensor): 注意力加權的上下文向量，形狀為 (batch_size, hidden_size * 2)
            attention_weights (Tensor): 注意力權重，形狀為 (batch_size, seq_len)
        """
        # 計算注意力分數: e_i = v^T * tanh(W * h_i + b)
        # lstm_outputs: (batch_size, seq_len, hidden_size * 2)
        # W(lstm_outputs): (batch_size, seq_len, attention_size)
        attention_scores = self.v(torch.tanh(self.W(lstm_outputs)))  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)

        # 應用遮罩：將非面向詞位置的分數設為負無窮大
        # 這樣 softmax 後這些位置的權重會接近 0
        attention_scores = attention_scores.masked_fill(~aspect_mask, float('-inf'))

        # 計算注意力權重: α_i = softmax(e_i)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # 計算上下文向量: context = Σ(α_i * h_i)
        # attention_weights: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        # lstm_outputs: (batch_size, seq_len, hidden_size * 2)
        attention_output = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            lstm_outputs  # (batch_size, seq_len, hidden_size * 2)
        ).squeeze(1)  # (batch_size, hidden_size * 2)

        return attention_output, attention_weights


class Classifier(nn.Module):
    """
    分類器

    此層將上下文向量映射到情感類別機率分布。

    架構:
        輸入 -> Dropout -> 全連接層 -> Softmax -> 輸出

    參數:
        input_size (int): 輸入特徵的維度
        num_classes (int): 分類類別數量，預設為 3（負面、中性、正面）
        dropout (float): Dropout 比率

    輸入形狀:
        - context_vector: (batch_size, input_size)

    輸出形狀:
        - output: (batch_size, num_classes) 情感類別機率分布
    """

    def __init__(self, input_size, num_classes=3, dropout=0.5):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Dropout 層
        self.dropout = nn.Dropout(dropout)

        # 全連接層
        self.fc = nn.Linear(input_size, num_classes)

        # 參數初始化
        self._init_weights()

    def _init_weights(self):
        """初始化分類器權重"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, context_vector):
        """
        前向傳播

        參數:
            context_vector (Tensor): 上下文向量，形狀為 (batch_size, input_size)

        返回:
            output (Tensor): 情感類別機率分布，形狀為 (batch_size, num_classes)
        """
        # 應用 Dropout
        x = self.dropout(context_vector)

        # 全連接層
        logits = self.fc(x)

        # Softmax（注意：如果使用 CrossEntropyLoss，不需要手動應用 softmax）
        # 因為 CrossEntropyLoss 內部會計算 softmax
        # 這裡返回 logits，在推理時再應用 softmax
        output = F.softmax(logits, dim=1)

        return output


# 組件測試函數
if __name__ == "__main__":
    """
    測試各個組件的功能和輸出形狀
    """
    print("="*50)
    print("測試模型組件")
    print("="*50)

    # 設定測試參數
    batch_size = 4
    seq_len = 10
    input_size = 300  # 詞嵌入維度
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    num_classes = 3

    # 建立測試數據
    input_seq = torch.randn(batch_size, seq_len, input_size)
    aspect_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    aspect_mask[:, 3:6] = True  # 假設第 3-5 個詞是面向詞

    # 測試 BiLSTMLayer
    print("\n1. 測試 BiLSTMLayer")
    print("-" * 50)
    bilstm = BiLSTMLayer(input_size, hidden_size, num_layers, dropout)
    sequence_output, (h_n, c_n) = bilstm(input_seq)
    print(f"輸入形狀: {input_seq.shape}")
    print(f"序列輸出形狀: {sequence_output.shape}")
    print(f"隱藏狀態形狀: {h_n.shape}")
    print(f"細胞狀態形狀: {c_n.shape}")
    assert sequence_output.shape == (batch_size, seq_len, hidden_size * 2)
    assert h_n.shape == (num_layers * 2, batch_size, hidden_size)
    assert c_n.shape == (num_layers * 2, batch_size, hidden_size)
    print("[PASS] BiLSTMLayer 測試通過")

    # 測試 AttentionLayer
    print("\n2. 測試 AttentionLayer")
    print("-" * 50)
    attention = AttentionLayer(hidden_size)
    attention_output, attention_weights = attention(sequence_output, aspect_mask)
    print(f"LSTM 輸出形狀: {sequence_output.shape}")
    print(f"面向詞遮罩形狀: {aspect_mask.shape}")
    print(f"注意力輸出形狀: {attention_output.shape}")
    print(f"注意力權重形狀: {attention_weights.shape}")
    print(f"注意力權重總和: {attention_weights.sum(dim=1)}")
    assert attention_output.shape == (batch_size, hidden_size * 2)
    assert attention_weights.shape == (batch_size, seq_len)
    # 驗證注意力權重和為 1
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    print("[PASS] AttentionLayer 測試通過")
    print("[PASS] 注意力權重和為 1")

    # 測試 Classifier
    print("\n3. 測試 Classifier")
    print("-" * 50)
    classifier = Classifier(hidden_size * 2, num_classes, dropout)
    output = classifier(attention_output)
    print(f"上下文向量形狀: {attention_output.shape}")
    print(f"分類輸出形狀: {output.shape}")
    print(f"機率分布總和: {output.sum(dim=1)}")
    assert output.shape == (batch_size, num_classes)
    # 驗證機率總和為 1
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    print("[PASS] Classifier 測試通過")
    print("[PASS] 機率分布總和為 1")

    print("\n" + "="*50)
    print("所有組件測試通過！")
    print("="*50)
