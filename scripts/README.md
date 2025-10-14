# Scripts 目錄說明

本目錄包含論文實驗的所有生產腳本。所有測試腳本已清理。

## 📁 腳本清單

### 1. 資料處理

#### `convert_to_sentence_level.py`
**功能**: 將 Aspect-Level 資料轉換為 Sentence-Level 資料

**用途**:
- 解析 SemEval XML 格式
- 使用多數決合併同一句子的多個 aspects
- 生成 sentence-level CSV 資料

**執行**:
```bash
python scripts/convert_to_sentence_level.py
```

**輸出**:
- `data/processed/sentence_level/SemEval2014_Restaurant_train_sentence_level.csv`
- `data/processed/sentence_level/SemEval2014_Restaurant_test_sentence_level.csv`
- `data/processed/sentence_level/SemEval2014_Laptop_train_sentence_level.csv`
- `data/processed/sentence_level/SemEval2014_Laptop_test_sentence_level.csv`

---

### 2. 模型訓練

#### `train_sentence_level_baseline.py`
**功能**: 訓練 Sentence-Level Baseline 模型

**支援模型**:
- `base`: BiLSTM（無 Attention）
- `pre_fusion`: Attention → BiLSTM
- `post_fusion`: BiLSTM → Attention

**參數**:
```bash
python scripts/train_sentence_level_baseline.py \
    --model_type base \
    --hidden_size 128 \
    --num_epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dataset SemEval2014_Restaurant \
    --patience 5
```

**主要參數**:
- `--model_type`: 模型類型 (base/pre_fusion/post_fusion)
- `--hidden_size`: LSTM 隱藏層大小 (128/256)
- `--num_lstm_layers`: LSTM 層數 (預設 2)
- `--dropout`: Dropout 比例 (預設 0.3)
- `--patience`: Early stopping 耐心值

**輸出**:
- `outputs/experiments/sentence_level/{dataset}/{model_type}/checkpoints/best_model.pt`
- `outputs/experiments/sentence_level/{dataset}/{model_type}/results/experiment_result.json`

---

#### `train_baseline.py`
**功能**: 訓練 Aspect-Level Baseline 模型

**用途**: 訓練原始的 aspect-level 情感分析模型

**執行**:
```bash
python scripts/train_baseline.py \
    --num_epochs 30 \
    --batch_size 32 \
    --use_official_split
```

**輸出**:
- `outputs/experiments/SemEval2014_Restaurant/baseline/checkpoints/best_model.pt`

---

### 3. 模型評估

#### `evaluate_sentence_level_model.py`
**功能**: 評估 Sentence-Level 模型並生成詳細報告

**功能**:
- 計算準確度、F1、Precision、Recall
- 生成混淆矩陣
- 生成分類報告
- 錯誤分析（列出前N個錯誤樣本）

**執行**:
```bash
python scripts/evaluate_sentence_level_model.py \
    --model_type base \
    --checkpoint "outputs/experiments/.../best_model.pt" \
    --dataset SemEval2014_Restaurant
```

**輸出**:
- `outputs/evaluations/{dataset}/{model_type}/confusion_matrix.png`
- `outputs/evaluations/{dataset}/{model_type}/classification_report.txt`
- `outputs/evaluations/{dataset}/{model_type}/error_analysis.txt`

---

### 4. 結果比較與分析

#### `compare_sentence_level_models.py`
**功能**: 比較不同 Sentence-Level 模型的效能

**比較項目**:
- F1-Macro、準確度
- 訓練曲線（Loss、Accuracy、F1）
- 最終效能條形圖

**執行**:
```bash
python scripts/compare_sentence_level_models.py
```

**輸出**:
- `outputs/comparisons/sentence_level/f1_macro_comparison.png`
- `outputs/comparisons/sentence_level/accuracy_comparison.png`
- `outputs/comparisons/sentence_level/loss_comparison.png`
- `outputs/comparisons/sentence_level/final_performance_comparison.png`
- `outputs/comparisons/sentence_level/baseline_sentence_attention_comparison.csv`

---

#### `generate_baseline_summary.py`
**功能**: 生成 Baseline 實驗總結報告

**功能**:
- 整合所有實驗結果
- 生成效能比較表
- 繪製綜合比較圖（多模型、多指標）
- 分析 Hidden Size 影響
- 生成 Markdown 報告

**執行**:
```bash
python scripts/generate_baseline_summary.py
```

**輸出**:
- `outputs/baseline_summary/baseline_sentence_results.csv`
- `outputs/baseline_summary/baseline_performance_comparison.png`
- `outputs/baseline_summary/baseline_training_curves.png`
- `outputs/baseline_summary/baseline_hidden_size_impact.png`
- `outputs/baseline_summary/baseline_summary_report.md`

---

### 5. 批次實驗

#### `run_all_experiments.py`
**功能**: 批次執行所有實驗

**執行**:
```bash
python scripts/run_all_experiments.py
```

**包含實驗**:
- 不同資料集（Restaurant, Laptop）
- 不同模型（Baseline, Pre-Fusion, Post-Fusion）
- 不同層數（2, 3, 4, 5 層）

---

## 📊 典型工作流程

### 階段 1: 資料準備
```bash
# 1. 轉換資料
python scripts/convert_to_sentence_level.py
```

### 階段 2: 訓練模型
```bash
# 2a. 訓練 Base 模型
python scripts/train_sentence_level_baseline.py --model_type base --hidden_size 128

# 2b. 訓練 Pre-Fusion 模型
python scripts/train_sentence_level_baseline.py --model_type pre_fusion --hidden_size 128

# 2c. 訓練 Post-Fusion 模型
python scripts/train_sentence_level_baseline.py --model_type post_fusion --hidden_size 128
```

### 階段 3: 評估模型
```bash
# 3. 評估每個模型
python scripts/evaluate_sentence_level_model.py \
    --model_type base \
    --checkpoint "outputs/experiments/.../base/checkpoints/best_model.pt"
```

### 階段 4: 比較與分析
```bash
# 4a. 比較模型
python scripts/compare_sentence_level_models.py

# 4b. 生成總結報告
python scripts/generate_baseline_summary.py
```

---

## 🔧 常用命令範例

### 快速訓練（5 epochs 測試）
```bash
python scripts/train_sentence_level_baseline.py \
    --model_type pre_fusion \
    --num_epochs 5 \
    --batch_size 32
```

### 訓練不同 Hidden Size
```bash
# Hidden Size = 128
python scripts/train_sentence_level_baseline.py --model_type base --hidden_size 128

# Hidden Size = 256
python scripts/train_sentence_level_baseline.py --model_type base --hidden_size 256
```

### 訓練不同 LSTM 層數
```bash
# 2 層
python scripts/train_sentence_level_baseline.py --num_lstm_layers 2

# 3 層
python scripts/train_sentence_level_baseline.py --num_lstm_layers 3
```

---

## 📝 注意事項

1. **隨機種子**: 所有訓練腳本使用固定種子 `--seed 42` 確保可重現性

2. **Early Stopping**: 使用 `--patience 5` 避免過擬合

3. **類別權重**: 自動計算並應用類別權重處理不平衡問題

4. **GPU 支援**: 腳本會自動檢測並使用 GPU（如果可用）

5. **輸出結構**: 所有輸出遵循統一的目錄結構，便於後續分析

---

## 📂 輸出目錄結構

```
outputs/
├── experiments/              # 實驗結果
│   └── sentence_level/
│       └── {dataset}/
│           └── {model_type}/
│               ├── checkpoints/
│               │   └── best_model.pt
│               └── results/
│                   └── experiment_result.json
│
├── evaluations/              # 評估報告
│   └── {dataset}/
│       └── {model_type}/
│           ├── confusion_matrix.png
│           ├── classification_report.txt
│           └── error_analysis.txt
│
├── comparisons/              # 模型比較
│   └── sentence_level/
│       ├── *.png
│       └── *.csv
│
└── baseline_summary/         # 總結報告
    ├── *.png
    ├── *.csv
    └── baseline_summary_report.md
```

---

## 🚀 快速開始

```bash
# 1. 準備資料
python scripts/convert_to_sentence_level.py

# 2. 訓練模型
python scripts/train_sentence_level_baseline.py --model_type pre_fusion

# 3. 評估模型
python scripts/evaluate_sentence_level_model.py \
    --model_type pre_fusion \
    --checkpoint "outputs/experiments/.../pre_fusion/checkpoints/best_model.pt"

# 4. 生成報告
python scripts/generate_baseline_summary.py
```

---

**最後更新**: 2025-10-14
**維護者**: Thesis Project
