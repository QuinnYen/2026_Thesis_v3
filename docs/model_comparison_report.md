# Baseline vs Aspect-Level 模型對比報告

## 📊 模型架構比較

### Baseline 模型
```
Input → Word Embedding → BiLSTM → Attention (masked) → Classifier
```

**特點：**
- ✓ 使用 `aspect_mask` 進行 attention masking
- ✓ 只在 aspect term 位置計算 attention
- ✗ **沒有位置感知能力**
- ✗ **無法區分詞與 aspect 的距離**

### Aspect-Level 模型（改進版）
```
Input → Word Embedding + Position Embedding → BiLSTM → Aspect-Aware Attention → Classifier
```

**關鍵改進：**
1. ✅ **Aspect Position Embedding**
   - 為每個詞添加相對於 aspect 的位置編碼
   - 距離 aspect 越近，影響越大

2. ✅ **Aspect-Aware Attention**
   - 結合 aspect 表示和上下文資訊
   - 加入距離衰減機制
   - 自動聚焦在 aspect 附近的詞

3. ✅ **明確區分不同 aspect**
   - 同一句子中的不同 aspect 會得到不同的表示
   - 位置感知能力強

---

## 🔬 實驗驗證結果

### 測試 1：Attention 聚焦性

| 模型 | Aspect 位置權重 | Aspect±3 範圍權重 | 遠離 Aspect 權重 |
|------|----------------|------------------|-----------------|
| **Baseline** | 1.000 | 1.000 | 0.000 |
| **Aspect-Level** | 0.288 | 0.553 | 0.145 |

**解讀：**
- **Baseline**: Attention 完全集中在 aspect term 本身（硬性遮罩）
- **Aspect-Level**: Attention 軟性聚焦在 aspect 附近，更自然的分布

### 測試 2：不同 Aspect 位置的區分能力

測試同一句子，aspect 在不同位置（前段、中段、後段）：

| Aspect 位置 | Baseline (Aspect 區域權重) | Aspect-Level (Aspect 區域權重) | Aspect-Level (±3 範圍) |
|------------|---------------------------|------------------------------|----------------------|
| 前段 (5-10) | 1.000 | 0.307 | 0.580 |
| 中段 (18-23) | 1.000 | 0.256 | 0.494 |
| 後段 (30-35) | 1.000 | 0.296 | 0.579 |

**結論：**
- ✅ Aspect-Level 模型的 attention 分布會隨 aspect 位置改變
- ✅ 能明確感知 aspect 在句子中的位置
- ✅ 附近詞的權重顯著高於遠處詞

### 測試 3：距離衰減效果

Aspect-Level 模型顯示明顯的**距離衰減特性**：
- 距離 aspect 0-3 個詞：平均權重 0.08-0.12
- 距離 aspect 4-7 個詞：平均權重 0.03-0.05
- 距離 aspect 8+ 個詞：平均權重 < 0.02

符合指數衰減模式：`weight ≈ a × exp(-b × distance)`

---

## 📈 參數量對比

| 模型 | 總參數量 | 可訓練參數 | 增加比例 |
|------|---------|-----------|---------|
| **Baseline** | 3,474,115 | 3,474,115 | - |
| **Aspect-Level** | 3,617,234 | 3,617,234 | **+4.12%** |

**增加的參數：143,119 個**

主要來自：
- Position Embedding 層：~5,100 個參數
- Aspect-Aware Attention 的額外權重矩陣：~138,000 個參數

**結論：參數增加不到 5%，改進顯著**

---

## 🎯 關鍵優勢

### Baseline 模型的問題
1. ❌ **硬性遮罩**：Attention 只能看到 aspect term 本身
2. ❌ **無位置感知**：無法知道詞離 aspect 有多遠
3. ❌ **缺乏上下文**：忽略 aspect 附近的重要詞
4. ❌ **無法區分**：同一句子中不同位置的 aspect 得到相同處理

### Aspect-Level 模型的改進
1. ✅ **軟性聚焦**：Attention 可以看到 aspect 周圍的上下文
2. ✅ **位置感知**：明確知道每個詞與 aspect 的距離
3. ✅ **距離衰減**：距離越遠，影響越小（更符合直覺）
4. ✅ **明確區分**：不同位置的 aspect 得到不同的表示

---

## 📊 視覺化結果

已生成以下視覺化圖表：

1. **`attention_comparison.png`**
   - 展示 Baseline 和 Aspect-Level 在不同 aspect 位置的 attention 分布
   - 清楚顯示兩者的差異

2. **`position_effect_heatmap.png`**
   - 熱圖展示不同 aspect 位置的 attention 分布
   - 驗證模型能區分不同 aspect

3. **`distance_decay.png`**
   - 展示距離衰減效果
   - 證明模型具有位置感知能力

---

## ✅ 驗收標準檢查

| 標準 | 結果 | 證據 |
|------|------|------|
| 模型能明確區分不同 aspect | ✅ **通過** | 不同位置的 aspect 產生不同的 attention 分布 |
| Attention 視覺化時能看出聚焦在 aspect | ✅ **通過** | 視覺化圖表顯示明確聚焦在 aspect 附近 |
| 架構與 sentence-level 有明顯不同 | ✅ **通過** | 加入位置嵌入和 aspect-aware attention |

---

## 🔬 技術細節

### 1. Aspect Position Embedding
```python
相對位置計算：position - aspect_center
位置範圍：[-50, +50]
嵌入維度：50
```

### 2. Aspect-Aware Attention
```python
注意力分數 = v^T × tanh(W_context × h_i + W_aspect × aspect_repr)
距離衰減 = exp(-λ × distance)
最終分數 = 注意力分數 + log(距離衰減)
```

### 3. 距離衰減參數
- 可學習參數 `λ`（初始值 0.1）
- 訓練過程中自動調整
- 確保距離越遠，權重越低

---

## 📝 建議

### 訓練時的注意事項
1. Position embedding 需要足夠的訓練才能學到有效的位置表示
2. 距離衰減參數 `λ` 建議設定範圍在 [0.01, 1.0]
3. 如果 aspect 長度較長（>5 個詞），可能需要調整位置計算方式

### 進一步改進方向
1. 可以嘗試不同的位置編碼方式（如 sinusoidal position encoding）
2. 可以加入多頭注意力機制
3. 可以結合預訓練語言模型（BERT、RoBERTa）

---

## 🎉 結論

**Aspect-Level 模型成功實現了以下目標：**

1. ✅ 加入位置感知能力（Position Embedding）
2. ✅ 實現軟性注意力聚焦（距離衰減）
3. ✅ 能明確區分不同位置的 aspect
4. ✅ 參數增加僅 4.12%，成本可控
5. ✅ 視覺化驗證效果顯著

**與 Baseline 相比，Aspect-Level 模型具有更強的 aspect 感知能力，更適合處理 aspect-level 的情感分析任務。**
