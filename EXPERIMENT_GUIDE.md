# 系統性實驗指南

## 快速開始

### 1. 測試單個實驗流程

在運行完整的批次實驗前，建議先測試單個實驗：

```bash
python test_single_experiment.py
```

這會訓練一個 PostFusion_2Layer 模型（3 epochs）來驗證整個流程。

### 2. 運行完整實驗

#### 所有模型 + 所有數據集

```bash
python scripts/run_all_experiments.py
```

這會訓練：
- 2 個數據集（Restaurant + Laptop）
- 9 個模型（1個Baseline + 4個Post-Fusion + 4個Pre-Fusion）
- **總共 18 個實驗**

預計時間：約 2-4 小時（取決於硬體）

#### 只訓練特定數據集

```bash
# 只訓練 Restaurant 數據集
python scripts/run_all_experiments.py --datasets restaurant

# 只訓練 Laptop 數據集
python scripts/run_all_experiments.py --datasets laptop
```

#### 只訓練特定模型類型

```bash
# 只訓練 Baseline
python scripts/run_all_experiments.py --models baseline

# 只訓練 Post-Fusion 模型
python scripts/run_all_experiments.py --models post_fusion

# 只訓練 Pre-Fusion 模型
python scripts/run_all_experiments.py --models pre_fusion

# 訓練 Baseline + Post-Fusion
python scripts/run_all_experiments.py --models baseline post_fusion
```

#### 自定義訓練參數

```bash
# 訓練 50 epochs，batch size 64
python scripts/run_all_experiments.py --num_epochs 50 --batch_size 64

# Restaurant 數據集 + Post-Fusion 模型 + 30 epochs
python scripts/run_all_experiments.py \
    --datasets restaurant \
    --models post_fusion \
    --num_epochs 30
```

## 輸出結構

```
outputs/experiments/
├── SemEval2014_Restaurant/
│   ├── baseline/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   └── last_model.pt
│   │   ├── results/
│   │   │   ├── experiment_result.json
│   │   │   ├── metrics.json
│   │   │   ├── confusion_matrix.png
│   │   │   └── evaluation_report.md
│   │   └── visualizations/
│   │       └── training_curves.png
│   ├── post_fusion_2/
│   ├── post_fusion_3/
│   ├── post_fusion_4/
│   ├── post_fusion_5/
│   ├── pre_fusion_2/
│   ├── pre_fusion_3/
│   ├── pre_fusion_4/
│   └── pre_fusion_5/
├── SemEval2014_Laptop/
│   └── (same structure)
└── summary/
    ├── experiments_summary.json
    ├── experiments_report.md
    └── performance_comparison.csv
```

## 結果分析

### 1. 查看總結報告

實驗完成後，查看 Markdown 報告：

```bash
cat outputs/experiments/summary/experiments_report.md
```

或打開：`outputs/experiments/summary/experiments_report.md`

### 2. 查看性能比較表

CSV 格式，可用 Excel 或 Python 分析：

```bash
cat outputs/experiments/summary/performance_comparison.csv
```

### 3. 查看 JSON 總結

包含所有實驗的詳細信息：

```bash
cat outputs/experiments/summary/experiments_summary.json
```

## 實驗配置

默認配置（可在 `scripts/run_all_experiments.py` 中修改）：

```python
'training': {
    'num_epochs': 20,           # 訓練 epochs
    'batch_size': 32,           # 批次大小
    'learning_rate': 0.001,     # 學習率
    'hidden_size': 128,         # LSTM 隱藏層大小
    'embedding_dim': 300,       # 詞嵌入維度
    'dropout': 0.3,             # Dropout 比例
    'patience': 10,             # Early stopping patience
    'val_split': 0.2,           # 驗證集比例
}
```

## 模型列表

| 模型名稱 | 架構 | 層數 |
|---------|------|------|
| baseline | BiLSTM → Attention | 1 |
| post_fusion_2 | BiLSTM(2層) → Attention | 2 |
| post_fusion_3 | BiLSTM(3層) → Attention | 3 |
| post_fusion_4 | BiLSTM(4層) → Attention | 4 |
| post_fusion_5 | BiLSTM(5層) → Attention | 5 |
| pre_fusion_2 | Attention → BiLSTM(2層) | 2 |
| pre_fusion_3 | Attention → BiLSTM(3層) | 3 |
| pre_fusion_4 | Attention → BiLSTM(4層) | 4 |
| pre_fusion_5 | Attention → BiLSTM(5層) | 5 |

## 預期結果

基於 Baseline 模型的經驗：

| 模型類型 | 預期準確度 | 預期 Macro-F1 |
|---------|-----------|--------------|
| Baseline (1層) | 70-72% | 0.60-0.62 |
| Post-Fusion (2層) | 72-74% | 0.62-0.64 |
| Post-Fusion (3層) | 74-76% | 0.64-0.66 |
| Post-Fusion (4-5層) | 74-77% | 0.64-0.68 |
| Pre-Fusion (2-5層) | 72-76% | 0.62-0.66 |

**目標：** 準確度 ≥ 75%，Macro-F1 ≥ 0.70

## 故障排除

### 記憶體不足

如果遇到 OOM (Out of Memory) 錯誤：

```bash
# 降低 batch size
python scripts/run_all_experiments.py --batch_size 16

# 或只訓練部分模型
python scripts/run_all_experiments.py --models baseline post_fusion
```

### 訓練中斷

實驗腳本會自動保存每個模型的結果，中斷後可以：

1. 查看 `outputs/experiments/summary/experiments_summary.json` 了解已完成的實驗
2. 手動運行剩餘的實驗，或
3. 重新運行整個腳本（會覆蓋已有結果）

### 查看進度

實驗運行時會顯示：
- 當前數據集
- 當前模型
- 實驗進度 (X/18)
- 每個 epoch 的訓練和驗證指標

## 高級用法

### 自定義實驗配置

編輯 `scripts/run_all_experiments.py` 中的 `EXPERIMENTS_CONFIG`：

```python
EXPERIMENTS_CONFIG = {
    'datasets': [
        # 添加或移除數據集
    ],
    'models': [
        # 添加或移除模型
    ],
    'training': {
        # 修改訓練超參數
    }
}
```

### 添加新數據集

在 `EXPERIMENTS_CONFIG['datasets']` 中添加：

```python
{
    'name': 'SemEval2016_Restaurant',
    'train_path': 'data/raw/SemEval-2016/...',
    'test_path': 'data/raw/SemEval-2016/...',
}
```

### 並行訓練

如果有多個 GPU，可以同時運行多個實驗：

```bash
# Terminal 1: 訓練 Restaurant 數據集
python scripts/run_all_experiments.py --datasets restaurant &

# Terminal 2: 訓練 Laptop 數據集
python scripts/run_all_experiments.py --datasets laptop &
```

## 結果可視化

每個實驗的結果包含：

1. **訓練曲線** (`visualizations/training_curves.png`)
   - Loss, Accuracy, F1 隨 epoch 變化

2. **混淆矩陣** (`results/confusion_matrix.png`)
   - 分類結果的混淆矩陣

3. **評估報告** (`results/evaluation_report.md`)
   - 詳細的性能指標和分析

## 論文用圖表

總結報告中包含：

1. **性能比較表** (CSV)
   - 可直接導入 LaTeX 或 Excel

2. **Markdown 報告**
   - 包含所有實驗的性能表格

3. **JSON 數據**
   - 可用於進一步分析和繪圖

---

## 快速檢查清單

訓練前：
- [ ] 確認數據集已下載並解壓到 `data/raw/`
- [ ] 確認有足夠的磁盤空間（至少 5GB）
- [ ] 確認有足夠的記憶體（建議 8GB+）

訓練中：
- [ ] 監控訓練進度
- [ ] 檢查是否有錯誤信息
- [ ] 確認模型正在收斂

訓練後：
- [ ] 查看總結報告
- [ ] 比較不同模型的性能
- [ ] 分析最佳模型的混淆矩陣
- [ ] 檢查是否達到目標性能
