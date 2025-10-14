"""
後融合架構（Post-Fusion）模型

架構：Input → Embedding → BiLSTM(N層) → Attention → Classifier

特點：
- 多層 BiLSTM 堆疊（2-5層）
- 層間 Dropout 防止過擬合
- Attention 只作用在最後一層的 BiLSTM 輸出
- 適合學習深層的序列特徵
"""

import torch
import torch.nn as nn
from .components import BiLSTMLayer, AttentionLayer, Classifier


class PostFusionModel(nn.Module):
    """
    後融合架構的 ABSA 模型

    架構流程：
    1. Embedding Layer: 將詞索引轉換為詞向量
    2. Multi-layer BiLSTM: 堆疊多層雙向 LSTM，提取深層序列特徵
    3. Attention Layer: 對最後一層 BiLSTM 的輸出計算注意力權重
    4. Classifier: 全連接層 + Softmax 進行情感分類

    參數：
        vocab_size (int): 詞彙表大小
        embedding_dim (int): 詞嵌入維度，默認 300
        hidden_size (int): LSTM 隱藏層大小，默認 128
        num_lstm_layers (int): BiLSTM 層數（2/3/4/5），默認 2
        num_classes (int): 分類類別數，默認 3（正面、負面、中性）
        dropout (float): Dropout 比例，默認 0.3
        pretrained_embeddings (torch.Tensor, optional): 預訓練詞嵌入矩陣
        freeze_embeddings (bool): 是否凍結詞嵌入層，默認 False
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
        freeze_embeddings=False
    ):
        super(PostFusionModel, self).__init__()

        # 驗證層數
        if num_lstm_layers < 2 or num_lstm_layers > 5:
            raise ValueError(f"num_lstm_layers 必須在 2-5 之間，但得到 {num_lstm_layers}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # 1. 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # 2. 多層 BiLSTM
        # 使用現有的 BiLSTMLayer，它已經支持多層
        self.bilstm = BiLSTMLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0  # 單層不需要 dropout
        )

        # 3. Attention 層（只作用在最後一層的輸出）
        # 注意：AttentionLayer 會自動處理 BiLSTM 的雙向輸出（hidden_size * 2）
        # 所以這裡傳入的是 hidden_size，不是 hidden_size * 2
        self.attention = AttentionLayer(hidden_size)

        # 4. 分類器
        self.classifier = Classifier(
            input_size=hidden_size * 2,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, input_ids, aspect_mask):
        """
        前向傳播

        參數：
            input_ids (torch.Tensor): 輸入詞索引 [batch_size, seq_len]
            aspect_mask (torch.Tensor): Aspect 位置掩碼 [batch_size, seq_len]
                                        1 表示 aspect 詞，0 表示非 aspect 詞

        返回：
            logits (torch.Tensor): 分類 logits [batch_size, num_classes]
            attention_weights (torch.Tensor): 注意力權重 [batch_size, seq_len]
        """
        # 確保 aspect_mask 是布林類型
        if aspect_mask.dtype != torch.bool:
            aspect_mask = aspect_mask.bool()

        # 1. 詞嵌入
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # 2. 多層 BiLSTM 編碼
        # BiLSTMLayer 返回 (sequence_output, (h_n, c_n))，我們只需要 sequence_output
        lstm_output, _ = self.bilstm(embeddings)  # [batch_size, seq_len, hidden_size * 2]

        # 3. Attention 機制（對最後一層輸出）
        context_vector, attention_weights = self.attention(lstm_output, aspect_mask)
        # context_vector: [batch_size, hidden_size * 2]
        # attention_weights: [batch_size, seq_len]

        # 4. 分類
        logits = self.classifier(context_vector)  # [batch_size, num_classes]

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
            'model_type': 'PostFusion',
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
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
        print("\n" + "="*70)
        print(f"模型類型: {info['model_type']} ({info['num_lstm_layers']} 層 BiLSTM)")
        print("="*70)
        print(f"詞彙表大小:      {info['vocab_size']:,}")
        print(f"詞嵌入維度:      {info['embedding_dim']}")
        print(f"隱藏層大小:      {info['hidden_size']}")
        print(f"BiLSTM 層數:     {info['num_lstm_layers']}")
        print(f"分類類別數:      {info['num_classes']}")
        print(f"Dropout:         {info['dropout']}")
        print("-"*70)
        print(f"總參數量:        {info['total_params']:,}")
        print(f"可訓練參數:      {info['trainable_params']:,}")
        print(f"凍結參數:        {info['frozen_params']:,}")
        print("="*70 + "\n")


# 便捷的工廠函數，用於創建不同層數的模型
def PostFusion_2Layer(**kwargs):
    """創建 2 層 BiLSTM 的後融合模型"""
    return PostFusionModel(num_lstm_layers=2, **kwargs)


def PostFusion_3Layer(**kwargs):
    """創建 3 層 BiLSTM 的後融合模型"""
    return PostFusionModel(num_lstm_layers=3, **kwargs)


def PostFusion_4Layer(**kwargs):
    """創建 4 層 BiLSTM 的後融合模型"""
    return PostFusionModel(num_lstm_layers=4, **kwargs)


def PostFusion_5Layer(**kwargs):
    """創建 5 層 BiLSTM 的後融合模型"""
    return PostFusionModel(num_lstm_layers=5, **kwargs)


# 測試代碼
if __name__ == '__main__':
    print("測試 PostFusionModel")
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
        model = PostFusionModel(
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
        aspect_mask = torch.zeros(batch_size, seq_len)
        aspect_mask[:, 10:15] = 1  # 假設 aspect 在位置 10-14

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

        print(f"\n✓ {num_layers} 層模型測試通過！")

    print("\n" + "="*70)
    print("測試便捷工廠函數:")
    print("-"*70)

    # 測試工廠函數
    models = [
        ('PostFusion_2Layer', PostFusion_2Layer),
        ('PostFusion_3Layer', PostFusion_3Layer),
        ('PostFusion_4Layer', PostFusion_4Layer),
        ('PostFusion_5Layer', PostFusion_5Layer)
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
    print("所有測試通過！✓")
    print("="*70)
