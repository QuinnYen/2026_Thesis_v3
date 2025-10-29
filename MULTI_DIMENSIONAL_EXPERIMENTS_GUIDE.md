

# 多維度實驗系統使用指南

## 📋 概述

本系統專為論文的五個維度實驗設計，支援：
- **維度1**：基礎架構對比（RQ1核心）
- **維度2**：詞向量提升
- **維度3**：上下文編碼器
- **維度4**：句法信息
- **維度5**：多種子穩定性

## 🎯 系統架構

```
configs/
  └── experiment_dimensions.py       # 五維度實驗配置

scripts/
  ├── run_multi_dimensional_experiments.py   # 多維度實驗調度器
  └── analyze_multi_dimensional_results.py   # 結果分析腳本

outputs/
  ├── experiment_checkpoints/        # 實驗檢查點（斷點續傳）
  ├── experiment_logs/               # 實驗日誌
  └── experiment_analysis/           # 分析結果
```

## 🚀 快速開始

### 測試實驗配置

查看實驗計劃而不執行：

```bash
# 測試配置文件
python configs/experiment_dimensions.py
```

### 執行基礎實驗（維度1）

```bash
# 模擬模式（快速測試）
python scripts/run_multi_dimensional_experiments.py \
    --dimensions 1 \
    --dry-run

# 實際執行
python scripts/run_multi_dimensional_experiments.py \
    --dimensions 1
```

## 📊 五個維度詳細說明

### 維度1：基礎架構對比 (RQ1核心)

**目標**：比較不同融合策略和模型深度的影響

**實驗組：**
- **A1**: Baseline（無Attention）- 1/2/3/4/5層
- **A2**: Pre-Fusion（Attention → BiLSTM）- 2/3/4/5層
- **A3**: Post-Fusion（BiLSTM → Attention）- 2/3/4/5層
- **A4**: Pre-Fusion（Attention → LSTM）- 2/3/4/5層
- **A5**: Post-Fusion（LSTM → Attention）- 2/3/4/5層

**執行方式：**
```bash
python scripts/run_multi_dimensional_experiments.py --dimensions 1
```

**研究問題：**
- RQ1：融合策略效能差異
- RQ2：模型深度影響

**論文章節**：第5.1節「基礎架構比較」

---

### 維度2：詞向量提升

**目標**：測試核心論點在更好特徵下是否成立

**實驗組：**
- **B1**: 隨機初始化詞向量
- **B2**: GloVe-100d 預訓練詞向量
- **B3**: GloVe-300d 預訓練詞向量
- **B4**: Word2Vec 預訓練詞向量

每組包含：Baseline、Pre-Fusion、Post-Fusion（使用最佳層數）

**執行方式：**
```bash
# 先執行維度1找出最佳層數
python scripts/run_multi_dimensional_experiments.py --dimensions 1

# 然後執行維度2
python scripts/run_multi_dimensional_experiments.py --dimensions 2

# 或一次執行兩個維度
python scripts/run_multi_dimensional_experiments.py --dimensions 1 2
```

**驗證**：不同詞向量下，融合策略的優劣是否一致

**論文章節**：第5.2節「詞向量對融合策略的影響」

---

### 維度3：上下文編碼器

**目標**：測試在SOTA編碼器下的表現

**實驗組：**
- **C1**: BiLSTM編碼器
  - Pre-Fusion
  - Post-Fusion
- **C2**: BERT編碼器
  - Pre-Fusion（BERT → Attention → BiLSTM）
  - Post-Fusion（BERT → BiLSTM → Attention）

**執行方式：**
```bash
python scripts/run_multi_dimensional_experiments.py --dimensions 3
```

**驗證**：使用SOTA編碼器後，融合策略是否仍有差異

**論文章節**：第5.3節「預訓練模型對融合策略的影響」

---

### 維度4：句法信息

**目標**：測試在結構化信息下的表現

**實驗組：**
- **D1**: 無句法信息
  - Pre-Fusion
  - Post-Fusion
- **D2**: 加入依存樹
  - Pre-Fusion + Dependency GNN
  - Post-Fusion + Dependency GNN

**執行方式：**
```bash
python scripts/run_multi_dimensional_experiments.py --dimensions 4
```

**驗證**：加入語法結構後，融合策略的差異

**論文章節**：第5.4節「句法信息對融合策略的影響」

---

### 維度5：多種子穩定性測試

**目標**：證明結果可靠性

**實驗組：**
- **E1**: 單一種子（seed=42）
- **E2**: 多種子測試（seeds=[42, 43, 44, 45, 46]）
  - 對所有最佳模型重複5次
  - 報告：均值 ± 標準差
  - 執行顯著性檢驗

**執行方式：**
```bash
# 針對所有最佳模型執行穩定性測試
python scripts/run_multi_dimensional_experiments.py --dimensions 5
```

**證明**：結果的穩定性和統計顯著性

**論文章節**：第5.5節「穩定性分析」

---

## 🔧 使用預定義組合

系統提供預定義的維度組合：

```bash
# 基礎實驗（僅維度1）
python scripts/run_multi_dimensional_experiments.py --combination basic

# 詞向量實驗（維度1+2）
python scripts/run_multi_dimensional_experiments.py --combination embeddings

# 編碼器實驗（維度1+3）
python scripts/run_multi_dimensional_experiments.py --combination encoder

# 句法實驗（維度1+4）
python scripts/run_multi_dimensional_experiments.py --combination syntax

# 穩定性測試（維度5）
python scripts/run_multi_dimensional_experiments.py --combination stability

# 完整實驗（所有維度）
python scripts/run_multi_dimensional_experiments.py --combination full
```

## 💾 斷點續傳

系統自動保存實驗進度，可隨時恢復：

```bash
# 實驗會自動儲存檢查點，會話ID格式：20231029_123456

# 從檢查點恢復
python scripts/run_multi_dimensional_experiments.py \
    --resume 20231029_123456
```

**檢查點位置：**
- 檢查點：`outputs/experiment_checkpoints/session_{session_id}.json`
- 日誌：`outputs/experiment_logs/experiment_log_{session_id}.txt`

## 📈 結果分析

完成實驗後，使用分析腳本：

```bash
# 分析指定會話
python scripts/analyze_multi_dimensional_results.py \
    --session 20231029_123456

# 生成 LaTeX 表格
python scripts/analyze_multi_dimensional_results.py \
    --session 20231029_123456 \
    --generate-latex

# 分析所有會話
python scripts/analyze_multi_dimensional_results.py --all
```

**分析輸出：**
- 統計摘要：`outputs/experiment_analysis/{session_id}/summary.txt`
- 視覺化圖表：`outputs/experiment_analysis/{session_id}/*.png`
- LaTeX 表格：`outputs/experiment_analysis/{session_id}/*.tex`

## 📊 生成的圖表

### 1. 維度比較圖
- 各維度內模型的 F1 和 Accuracy 箱型圖
- 檔案：`dimension_{n}_comparison.png`

### 2. 層數影響圖
- 展示不同層數對性能的影響
- 檔案：`{model_type}_layer_effect.png`

### 3. 整體熱力圖
- 所有模型在所有數據集上的性能熱力圖
- 檔案：`overall_heatmap.png`

## 🎯 推薦實驗流程

### 階段1：基礎實驗（1-2週）
```bash
# 1. 執行基礎架構實驗（維度1）
python scripts/run_multi_dimensional_experiments.py --combination basic

# 2. 分析結果，找出最佳層數和融合策略
python scripts/analyze_multi_dimensional_results.py --session {session_id}

# 3. 根據結果決定後續實驗重點
```

### 階段2：特徵提升實驗（1週）
```bash
# 執行詞向量實驗（維度2）
python scripts/run_multi_dimensional_experiments.py --combination embeddings

# 分析詞向量對融合策略的影響
python scripts/analyze_multi_dimensional_results.py --session {session_id}
```

### 階段3：編碼器和句法實驗（2週）
```bash
# 執行編碼器實驗（維度3）
python scripts/run_multi_dimensional_experiments.py --combination encoder

# 執行句法實驗（維度4）
python scripts/run_multi_dimensional_experiments.py --combination syntax

# 綜合分析
python scripts/analyze_multi_dimensional_results.py --all --generate-latex
```

### 階段4：穩定性驗證（3-5天）
```bash
# 對所有最佳模型執行多種子測試
python scripts/run_multi_dimensional_experiments.py --combination stability

# 生成統計顯著性分析
python scripts/analyze_multi_dimensional_results.py --session {session_id} --generate-latex
```

## ⚙️ 配置修改

所有實驗配置集中在 `configs/experiment_dimensions.py`：

```python
# 範例：修改維度1的實驗組
DIMENSION_1_ARCHITECTURE = {
    "experiment_groups": {
        "A1_baseline_no_attention": {
            "model_type": "baseline_no_attention",
            "use_attention": False,
            "layers": [1, 2, 3, 4, 5],  # 修改層數列表
        },
        # ... 其他配置
    }
}
```

## 🔍 實驗進度監控

實驗執行中會顯示：
- 當前實驗進度（X/總數）
- 完成百分比
- 預估剩餘時間
- 失敗/跳過的實驗數

**即時日誌：**
```bash
# 監控實驗日誌
tail -f outputs/experiment_logs/experiment_log_{session_id}.txt
```

## 💡 常見問題

### Q1: 實驗中斷了怎麼辦？
A: 使用 `--resume {session_id}` 從檢查點恢復。系統會自動跳過已完成的實驗。

### Q2: 如何只執行特定數據集？
A: 使用 `--datasets` 參數：
```bash
python scripts/run_multi_dimensional_experiments.py \
    --dimensions 1 \
    --datasets SemEval2014_Restaurant
```

### Q3: 如何預估總執行時間？
A: 執行前會顯示預估時間。單個模型約15-30分鐘，可根據實驗數量估算。

### Q4: 實驗失敗了怎麼辦？
A: 系統會記錄失敗原因在日誌中，繼續執行其他實驗。可以稍後手動重試失敗的實驗。

### Q5: 如何添加新的實驗維度？
A: 在 `experiment_dimensions.py` 中添加新的維度配置，參考現有維度的格式。

## 📝 待實現功能

**注意**：目前以下功能尚未實現，需要逐步開發：

### 維度1
- ✅ 配置文件定義
- ❌ Baseline（無Attention）模型
- ❌ LSTM（單向）版本的 Pre/Post-Fusion

### 維度2
- ✅ 配置文件定義
- ❌ GloVe-100d 載入邏輯
- ❌ Word2Vec 載入邏輯

### 維度3
- ✅ 配置文件定義
- ❌ BERT編碼器整合
- ❌ BERT + Pre/Post-Fusion 模型

### 維度4
- ✅ 配置文件定義
- ❌ 依存句法解析
- ❌ GNN 模組
- ❌ 句法信息整合到 Pre/Post-Fusion

### 維度5
- ✅ 配置文件定義
- ❌ 多種子訓練邏輯
- ❌ 統計顯著性檢驗

### 系統功能
- ✅ 實驗調度器框架
- ✅ 檢查點機制
- ✅ 結果分析腳本
- ❌ 實驗執行器與訓練腳本整合
- ❌ 平行執行支援
- ❌ GPU 資源管理

## 🎓 下一步

1. **先測試基礎功能**：確保維度1可以正常運行
2. **逐步實現模型**：按照維度1→2→3→4→5的順序實現
3. **整合訓練邏輯**：將 `train_unified.py` 整合到調度器
4. **完善分析功能**：添加更多統計分析和視覺化

## 📧 系統特色

- ✅ **統一配置**：所有實驗集中配置，易於管理
- ✅ **斷點續傳**：實驗中斷可隨時恢復
- ✅ **自動追蹤**：實驗進度和結果自動記錄
- ✅ **靈活組合**：支援任意維度組合
- ✅ **可擴展性**：易於添加新維度和實驗
- ✅ **結果分析**：自動生成統計和視覺化
- ✅ **論文就緒**：直接生成 LaTeX 表格

---

**重要提醒**：這是一個腳本框架，核心訓練邏輯需要逐步實現。建議先從維度1開始，確保基礎功能正常後再逐步擴展。
