"""
Aspect Mask 機制分析與優化方案

目的：深入理解當前aspect mask如何運作，以及如何優化以提升模型性能
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加專案路徑
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "models"))

from models.components import AttentionLayer


def print_section(title):
    """打印分隔線"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)


# ============================================================================
# 第一部分：當前 Aspect Mask 機制分析
# ============================================================================

print_section("1. 當前 Aspect Mask 機制運作原理")

print("""
當前機制：Hard Attention Masking（硬注意力遮罩）

【工作流程】
┌──────────────────────────────────────────────────────────────┐
│ 步驟1: 數據準備（cleaned_data_loader.py:94-121）             │
│  - 輸入: 句子 + aspect term + 字符位置 [from_pos, to_pos)    │
│  - 輸出: aspect_mask (布林向量)                              │
│                                                              │
│  例如：                                                       │
│    句子: "The food is great but service is bad"             │
│    Aspect: "food"                                            │
│    Tokens:  [The, food, is, great, but, service, is, bad]   │
│    Mask:    [  ·,   █,  ·,    ·,   ·,      ·,  ·,   ·]     │
│             (只有aspect詞為True，其他為False)                 │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 步驟2: BiLSTM編碼（baseline.py:126）                         │
│  - 輸入: 詞嵌入 (batch_size, seq_len, embed_dim)            │
│  - 輸出: LSTM隱藏狀態 (batch_size, seq_len, hidden_size*2)  │
│  - 作用: 捕捉雙向上下文信息                                   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 步驟3: Attention計算（components.py:137-171）                │
│                                                              │
│  3.1 計算原始attention scores:                               │
│      score_i = v^T * tanh(W * h_i + b)                      │
│      → 所有token都有分數                                      │
│                                                              │
│  3.2 應用aspect mask（硬遮罩）:                              │
│      score_i = -1e9  if mask_i == False                     │
│      score_i = score_i  if mask_i == True                   │
│                                                              │
│  3.3 Softmax歸一化:                                          │
│      attention_weights = softmax(scores)                     │
│      → 非aspect token的權重 ≈ 0                              │
│      → aspect token的權重總和 = 1                            │
│                                                              │
│  3.4 加權求和:                                               │
│      context = Σ(attention_weights_i * h_i)                 │
│      → 幾乎只使用aspect token的表示                          │
└──────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# 第二部分：當前機制的優缺點
# ============================================================================

print_section("2. 當前機制的優缺點分析")

print("""
✅ 優點：

1. 【明確的目標聚焦】
   - 強制模型只關注aspect詞本身
   - 避免無關詞干擾
   - 符合aspect-based sentiment analysis的直覺

2. 【計算高效】
   - 硬遮罩實現簡單
   - 不需要額外的可學習參數
   - 前向傳播快速

3. 【可解釋性強】
   - Attention權重清晰：aspect詞權重高，其他為0
   - 便於視覺化和debug

4. 【數值穩定】（已修復）
   - 使用-1e9替代-inf避免NaN
   - 有fallback機制處理異常情況

❌ 缺點和局限性：

1. 【忽略上下文信息】★★★ 最嚴重
   問題：只看aspect詞，丟失周圍的opinion詞

   例子：
   句子: "The food is great but service is terrible"
   Aspect: "food"

   當前: 只關注 "food"
   問題: "great" 這個opinion詞沒被利用！

   理想: 應該同時關注 "food" + "great" + 上下文

2. 【Multi-word aspect處理不理想】
   問題：多詞aspect (如 "battery life") 被同等對待

   當前: ["battery", "life"] 兩個token權重相同
   改進: "life" 可能應該有更高權重（核心詞）

3. 【無法處理隱式aspect】
   問題：aspect詞不在句子中時無法工作

   例子: "This place is overpriced"
   真實aspect: price (隱式)
   當前: 無法處理，因為沒有"price"這個詞

4. 【硬約束太強】
   問題：完全阻斷非aspect詞的信息流

   改進方向: 軟遮罩 (soft mask) 允許少量上下文信息

5. 【長距離依賴困難】
   例子: "The food, despite being cold, was still tasty"
   Aspect: "food"

   當前: 只看"food"，"cold"和"tasty"都沒用到
   問題: 情感修飾詞離aspect較遠時信息丟失
""")


# ============================================================================
# 第三部分：優化方案
# ============================================================================

print_section("3. 優化方案：從硬遮罩到軟遮罩 + 上下文感知")

print("""
┌──────────────────────────────────────────────────────────────┐
│ 方案A：Soft Masking（軟遮罩）                                 │
├──────────────────────────────────────────────────────────────┤
│ 核心思想：不是完全遮蔽，而是降低非aspect詞的權重              │
│                                                              │
│ 實現：                                                        │
│   原始: score_i = -1e9 if not aspect                        │
│   改進: score_i = score_i - penalty if not aspect           │
│         penalty = lambda * distance_to_aspect                │
│                                                              │
│ 效果：                                                        │
│   - aspect詞依然有最高權重（90%+）                           │
│   - 近鄰詞獲得少量權重（5-10%）                              │
│   - 遠處詞權重極低（<1%）                                    │
│                                                              │
│ 優點：                                                        │
│   ✓ 保留aspect為中心                                         │
│   ✓ 允許opinion詞貢獻                                        │
│   ✓ 距離衰減符合語言學直覺                                   │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 方案B：Contextual Window（上下文窗口）                        │
├──────────────────────────────────────────────────────────────┤
│ 核心思想：擴展mask範圍到aspect的鄰近詞                        │
│                                                              │
│ 實現：                                                        │
│   window_size = 3  # 左右各3個詞                            │
│   extended_mask[i] = True if |i - aspect_pos| <= window     │
│                                                              │
│ 例子：                                                        │
│   句子: [The, food, is, great, but, service, is, bad]       │
│   原始mask:  [·, █, ·, ·, ·, ·, ·, ·]                       │
│   擴展mask:  [█, █, █, █, █, ·, ·, ·]  (window=3)          │
│                                                              │
│ 優點：                                                        │
│   ✓ 簡單直接                                                 │
│   ✓ 捕捉局部opinion詞                                        │
│   ✓ 可調節window大小                                         │
│                                                              │
│ 缺點：                                                        │
│   ✗ 固定窗口無法適應不同句子結構                              │
│   ✗ 可能引入噪聲詞                                           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 方案C：Dependency-Aware Masking（依賴句法感知）★★★ 最佳      │
├──────────────────────────────────────────────────────────────┤
│ 核心思想：使用句法依賴關係確定相關詞                          │
│                                                              │
│ 實現：                                                        │
│   1. 使用依賴解析器（如spaCy）                               │
│   2. 找出與aspect有直接/間接依賴關係的詞                     │
│   3. 根據依賴距離設置mask權重                                │
│                                                              │
│ 例子：                                                        │
│   句子: "The food is great"                                  │
│   依賴: food --nsubj--> is --acomp--> great                 │
│                                                              │
│   Mask權重:                                                  │
│     food:  1.0  (aspect本身)                                │
│     is:    0.5  (直接依賴)                                  │
│     great: 0.8  (opinion詞，高權重)                         │
│     The:   0.1  (間接依賴，低權重)                          │
│                                                              │
│ 優點：                                                        │
│   ✓✓✓ 語言學合理性最強                                       │
│   ✓✓✓ 自動識別opinion詞                                      │
│   ✓✓ 適應不同句子結構                                        │
│   ✓ 可處理長距離依賴                                         │
│                                                              │
│ 缺點：                                                        │
│   ✗ 需要額外的依賴解析（計算成本）                           │
│   ✗ 依賴解析器可能有錯誤                                     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 方案D：Learnable Mask（可學習遮罩）                           │
├──────────────────────────────────────────────────────────────┤
│ 核心思想：讓模型自己學習如何利用context                       │
│                                                              │
│ 實現：                                                        │
│   mask_logits = MaskNetwork(h_i, aspect_representation)      │
│   soft_mask = sigmoid(mask_logits)                          │
│   adjusted_score = original_score * soft_mask               │
│                                                              │
│ 架構：                                                        │
│   MaskNetwork = 2-layer MLP                                  │
│   輸入: [h_i; aspect_repr; distance_feature]                │
│   輸出: mask權重 ∈ [0, 1]                                    │
│                                                              │
│ 優點：                                                        │
│   ✓✓ 完全數據驅動                                            │
│   ✓✓ 適應性最強                                              │
│   ✓ 可學習複雜模式                                           │
│                                                              │
│ 缺點：                                                        │
│   ✗ 增加參數量                                               │
│   ✗ 需要更多訓練數據                                         │
│   ✗ 可解釋性降低                                             │
└──────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# 第四部分：實驗代碼示範
# ============================================================================

print_section("4. 代碼示範：當前 vs 軟遮罩")

print("\n【模擬實驗設置】")
batch_size = 2
seq_len = 8
hidden_size = 128

# 模擬LSTM輸出
lstm_outputs = torch.randn(batch_size, seq_len, hidden_size * 2)

# 模擬aspect mask
# 樣本1: aspect在位置2 ("food")
# 樣本2: aspect在位置5-6 ("battery life")
aspect_mask = torch.tensor([
    [False, False, True, False, False, False, False, False],  # "food"
    [False, False, False, False, False, True, True, False],   # "battery life"
], dtype=torch.bool)

print(f"\nLSTM輸出形狀: {lstm_outputs.shape}")
print(f"Aspect Mask形狀: {aspect_mask.shape}")
print(f"\nAspect Mask:")
for i, mask in enumerate(aspect_mask):
    mask_str = ''.join(['█' if m else '·' for m in mask])
    print(f"  樣本{i+1}: {mask_str}")


# 當前硬遮罩方法
print("\n" + "-"*80)
print("【方法1：當前硬遮罩 (Hard Masking)】")
print("-"*80)

attention_layer = AttentionLayer(hidden_size=hidden_size)
context_hard, weights_hard = attention_layer(lstm_outputs, aspect_mask)

print("\nAttention權重分佈:")
for i, weights in enumerate(weights_hard):
    print(f"  樣本{i+1}: {[f'{w:.4f}' for w in weights.tolist()]}")
    aspect_sum = weights[aspect_mask[i]].sum().item()
    print(f"          Aspect詞權重總和: {aspect_sum:.4f}")


# 軟遮罩方法（模擬）
print("\n" + "-"*80)
print("【方法2：軟遮罩 (Soft Masking) - 距離衰減】")
print("-"*80)

def soft_masking_attention(lstm_outputs, aspect_mask, penalty_weight=5.0):
    """
    軟遮罩：使用距離衰減而非硬遮蔽

    Args:
        penalty_weight: 懲罰強度（越大，非aspect詞權重越低）
    """
    batch_size, seq_len, hidden_dim = lstm_outputs.shape

    # 計算原始attention scores（簡化版）
    scores = torch.randn(batch_size, seq_len)  # 模擬attention scores

    # 計算每個token到最近aspect token的距離
    distances = torch.zeros(batch_size, seq_len)
    for b in range(batch_size):
        aspect_positions = torch.where(aspect_mask[b])[0]
        for t in range(seq_len):
            if aspect_mask[b, t]:
                distances[b, t] = 0  # aspect詞本身距離為0
            else:
                # 到最近aspect詞的距離
                dist = torch.abs(torch.tensor(t) - aspect_positions.float()).min()
                distances[b, t] = dist

    # 應用距離衰減懲罰
    penalties = penalty_weight * distances
    adjusted_scores = scores - penalties

    # Softmax
    attention_weights = F.softmax(adjusted_scores, dim=1)

    return attention_weights, distances

weights_soft, distances = soft_masking_attention(lstm_outputs, aspect_mask, penalty_weight=5.0)

print("\n距離矩陣（到最近aspect詞）:")
for i, dist in enumerate(distances):
    print(f"  樣本{i+1}: {[f'{d:.0f}' for d in dist.tolist()]}")

print("\nSoft Masking Attention權重分佈:")
for i, weights in enumerate(weights_soft):
    print(f"  樣本{i+1}: {[f'{w:.4f}' for w in weights.tolist()]}")
    aspect_sum = weights[aspect_mask[i]].sum().item()
    non_aspect_sum = weights[~aspect_mask[i]].sum().item()
    print(f"          Aspect詞權重: {aspect_sum:.4f}")
    print(f"          非Aspect詞權重: {non_aspect_sum:.4f}")
    print(f"          鄰近詞(-1,+1)獲得權重: {weights_soft[i][torch.abs(distances[i]) == 1].sum():.4f}")


# 上下文窗口方法
print("\n" + "-"*80)
print("【方法3：上下文窗口 (Context Window)】")
print("-"*80)

def context_window_masking(aspect_mask, window_size=2):
    """
    擴展mask到鄰近窗口

    Args:
        window_size: 窗口大小（aspect左右各延伸多少個token）
    """
    batch_size, seq_len = aspect_mask.shape
    extended_mask = aspect_mask.clone()

    for b in range(batch_size):
        aspect_positions = torch.where(aspect_mask[b])[0]
        for pos in aspect_positions:
            # 擴展到左右鄰近token
            start = max(0, pos - window_size)
            end = min(seq_len, pos + window_size + 1)
            extended_mask[b, start:end] = True

    return extended_mask

extended_mask = context_window_masking(aspect_mask, window_size=2)

print("\n擴展後的Mask (window_size=2):")
for i, (orig, ext) in enumerate(zip(aspect_mask, extended_mask)):
    orig_str = ''.join(['█' if m else '·' for m in orig])
    ext_str = ''.join(['█' if m else '·' for m in ext])
    print(f"  樣本{i+1} 原始: {orig_str}")
    print(f"         擴展: {ext_str}")


# ============================================================================
# 第五部分：推薦方案和實施建議
# ============================================================================

print_section("5. 推薦方案和實施建議")

print("""
┌──────────────────────────────────────────────────────────────┐
│ 推薦實施優先級                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 🥇 優先級1：軟遮罩 (Soft Masking) + 距離衰減                 │
│    理由：                                                     │
│      ✓ 實現簡單，改動最小                                     │
│      ✓ 不需要額外數據或模型                                   │
│      ✓ 效果明顯（預期+3-5% F1）                              │
│      ✓ 計算開銷極小                                           │
│                                                              │
│    實施步驟：                                                 │
│      1. 修改 components.py 的 AttentionLayer.forward()       │
│      2. 將 masked_fill(-1e9) 改為距離衰減懲罰                │
│      3. 添加超參數 penalty_weight (建議範圍: 3-7)            │
│      4. 在驗證集上調優 penalty_weight                         │
│                                                              │
│    代碼修改量：約30行                                         │
│    預估開發時間：2-3小時                                      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 🥈 優先級2：上下文窗口 (Context Window)                       │
│    理由：                                                     │
│      ✓ 實現也很簡單                                           │
│      ✓ 可解釋性強                                             │
│      ✓ 適合短句子                                             │
│                                                              │
│    實施步驟：                                                 │
│      1. 在數據加載時擴展aspect_mask                           │
│      2. 添加 window_size 參數到配置                          │
│      3. 保持attention機制不變                                │
│                                                              │
│    代碼修改量：約20行                                         │
│    預估開發時間：1-2小時                                      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 🥉 優先級3：依賴句法感知 (Dependency-Aware)                  │
│    理由：                                                     │
│      ✓✓ 理論上效果最好                                        │
│      ✓ 語言學合理性強                                         │
│      ✗ 需要整合spaCy                                          │
│      ✗ 計算開銷較大                                           │
│                                                              │
│    實施步驟：                                                 │
│      1. 安裝spaCy + 英文模型                                 │
│      2. 預處理階段計算依賴關係                                │
│      3. 將依賴距離存入CSV                                     │
│      4. 訓練時使用依賴權重                                    │
│                                                              │
│    代碼修改量：約100行                                        │
│    預估開發時間：1-2天                                        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 🔬 研究方向：可學習遮罩 (Learnable Mask)                      │
│    理由：                                                     │
│      ✓✓ 潛力最大                                              │
│      ✗✗ 實現複雜                                              │
│      ✗ 需要消融實驗                                           │
│                                                              │
│    建議：作為後續改進方向                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 組合策略建議                                                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 最佳組合：軟遮罩 + 上下文窗口                                │
│                                                              │
│   實施方式：                                                  │
│     1. 先擴展mask到window (如window=2)                       │
│     2. 在window內使用軟遮罩（距離衰減）                       │
│     3. window外硬遮蔽                                         │
│                                                              │
│   優點：                                                      │
│     ✓ 兼顧效率和效果                                          │
│     ✓ 控制信息量（避免過多噪聲）                              │
│     ✓ 兩個超參數可調                                          │
│                                                              │
│   預期效果提升：                                              │
│     - Restaurant: F1 67% → 72% (+5%)                        │
│     - Laptop: F1 41% → 47% (+6%)                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
""")


print_section("6. 快速實施代碼示例")

print("""
修改 src/models/components.py 的 AttentionLayer:

```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, attention_size=None, use_soft_mask=True,
                 penalty_weight=5.0, window_size=2):
        super().__init__()
        self.hidden_size = hidden_size * 2
        self.use_soft_mask = use_soft_mask
        self.penalty_weight = penalty_weight
        self.window_size = window_size

        self.W = nn.Linear(self.hidden_size, attention_size or hidden_size, bias=True)
        self.v = nn.Linear(attention_size or hidden_size, 1, bias=False)
        self._init_weights()

    def forward(self, lstm_outputs, aspect_mask):
        # 計算attention scores
        attention_scores = self.v(torch.tanh(self.W(lstm_outputs)))
        attention_scores = attention_scores.squeeze(-1)

        if self.use_soft_mask:
            # 軟遮罩：計算距離懲罰
            batch_size, seq_len = aspect_mask.shape
            distances = self._compute_distances(aspect_mask, seq_len)
            penalties = self.penalty_weight * distances
            attention_scores = attention_scores - penalties
        else:
            # 硬遮罩：原始方法
            attention_scores = attention_scores.masked_fill(~aspect_mask, -1e9)

        attention_weights = F.softmax(attention_scores, dim=1)

        # 數值穩定性檢查
        if torch.isnan(attention_weights).any():
            attention_weights = aspect_mask.float() / aspect_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # 加權求和
        attention_output = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_outputs
        ).squeeze(1)

        return attention_output, attention_weights

    def _compute_distances(self, aspect_mask, seq_len):
        \"\"\"計算每個token到最近aspect token的距離\"\"\"
        batch_size = aspect_mask.size(0)
        distances = torch.zeros_like(aspect_mask, dtype=torch.float)

        for b in range(batch_size):
            aspect_positions = torch.where(aspect_mask[b])[0]
            if len(aspect_positions) == 0:
                continue

            for t in range(seq_len):
                if aspect_mask[b, t]:
                    distances[b, t] = 0
                else:
                    # 擴展窗口內使用漸變距離，外部使用大懲罰
                    dist = torch.abs(torch.tensor(t, dtype=torch.float) - aspect_positions.float()).min()
                    if dist <= self.window_size:
                        distances[b, t] = dist  # 窗口內：線性距離
                    else:
                        distances[b, t] = dist * 2  # 窗口外：雙倍懲罰

        return distances
```

在 experiment_config.py 中添加配置:

```python
BASELINE_CONFIG = {
    ...
    "use_soft_mask": True,       # 使用軟遮罩
    "penalty_weight": 5.0,       # 懲罰強度
    "context_window": 2,         # 上下文窗口大小
}
```
""")

print("\n" + "="*80)
print("分析完成！建議先實施【優先級1：軟遮罩】，預期可提升5-6% F1分數")
print("="*80 + "\n")
