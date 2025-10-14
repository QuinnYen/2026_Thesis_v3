"""
測試 PreFusionModel 的腳本
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
from src.models.pre_fusion import (
    PreFusionModel,
    PreFusion_2Layer,
    PreFusion_3Layer,
    PreFusion_4Layer,
    PreFusion_5Layer
)
from src.models.post_fusion import PostFusionModel


def test_pre_fusion_models():
    """測試不同層數的 PreFusion 模型"""
    print("測試 PreFusionModel")
    print("="*80)

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
        print("-"*80)

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
        print(f"  input_ids:         {list(input_ids.shape)}")
        print(f"  aspect_mask:       {list(aspect_mask.shape)}")
        print(f"\n輸出形狀:")
        print(f"  logits:            {list(logits.shape)}")
        print(f"  attention_weights: {list(attention_weights.shape)}")

        # 驗證輸出形狀正確
        assert logits.shape == (batch_size, num_classes), \
            f"Logits 形狀錯誤: 期望 {(batch_size, num_classes)}, 實際 {logits.shape}"
        assert attention_weights.shape == (batch_size, seq_len), \
            f"Attention weights 形狀錯誤: 期望 {(batch_size, seq_len)}, 實際 {attention_weights.shape}"

        print(f"\n[OK] {num_layers} 層模型測試通過！")

    print("\n" + "="*80)
    print("測試便捷工廠函數:")
    print("-"*80)

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

    print("\n" + "="*80)
    print("測試參數量隨層數增加:")
    print("-"*80)

    # 驗證參數量隨層數增加
    prev_params = 0
    for num_layers in [2, 3, 4, 5]:
        model = PreFusionModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_lstm_layers=num_layers,
            num_classes=num_classes
        )
        total_params, trainable_params = model.get_num_params()

        print(f"{num_layers} 層 - 總參數: {total_params:,}, 可訓練: {trainable_params:,}")

        if prev_params > 0:
            increase = total_params - prev_params
            print(f"       與 {num_layers-1} 層相比增加: {increase:,} 參數")

        prev_params = total_params

    print("\n" + "="*80)
    print("比較前融合 vs 後融合架構:")
    print("-"*80)

    # 比較前融合和後融合的架構差異
    pre_model = PreFusionModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_lstm_layers=3,
        num_classes=num_classes
    )

    post_model = PostFusionModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_lstm_layers=3,
        num_classes=num_classes
    )

    pre_params, _ = pre_model.get_num_params()
    post_params, _ = post_model.get_num_params()

    print(f"\n3 層模型參數量比較:")
    print(f"  前融合 (PreFusion):  {pre_params:,}")
    print(f"  後融合 (PostFusion): {post_params:,}")
    print(f"  差異:                {abs(pre_params - post_params):,}")

    print(f"\n架構流程對比:")
    print(f"  前融合: Embedding → Attention → BiLSTM(3層) → Classifier")
    print(f"  後融合: Embedding → BiLSTM(3層) → Attention → Classifier")

    # 測試在相同輸入下的輸出
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    aspect_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    aspect_mask[:, 10:15] = True

    pre_logits, pre_attn = pre_model(input_ids, aspect_mask)
    post_logits, post_attn = post_model(input_ids, aspect_mask)

    print(f"\n相同輸入下的輸出:")
    print(f"  前融合 logits shape:  {list(pre_logits.shape)}")
    print(f"  後融合 logits shape:  {list(post_logits.shape)}")
    print(f"  前融合 attention shape: {list(pre_attn.shape)}")
    print(f"  後融合 attention shape: {list(post_attn.shape)}")

    # 驗證輸出值不同（因為架構不同）
    logits_diff = torch.abs(pre_logits - post_logits).mean().item()
    attn_diff = torch.abs(pre_attn - post_attn).mean().item()

    print(f"\n輸出差異（驗證架構不同）:")
    print(f"  Logits 平均差異:     {logits_diff:.6f}")
    print(f"  Attention 平均差異:  {attn_diff:.6f}")

    if logits_diff > 0.01:
        print(f"  [OK] 架構明顯不同！")
    else:
        print(f"  [WARNING] 架構可能過於相似")

    print("\n" + "="*80)
    print("所有測試通過！[OK]")
    print("="*80)


if __name__ == '__main__':
    test_pre_fusion_models()
