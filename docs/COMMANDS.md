# 命令速查表

## ⚙️ 配置實驗（新增功能）

所有訓練腳本現在從 `configs/experiment_config.py` 讀取配置。

### 配置要訓練的數據集

編輯 `configs/experiment_config.py`，取消註解需要的數據集：

```python
DATASETS = [
    "SemEval2014_Restaurant",  # ✓ 已啟用
    "SemEval2014_Laptop",      # ✓ 已啟用
    # "SemEval2016_Restaurant",  # 取消註解以啟用
]
```

### 在多個數據集上訓練

```bash
# 在所有配置的數據集上訓練所有模型
python scripts/train_all_datasets.py --model all

# 只訓練特定模型類型
python scripts/train_all_datasets.py --model baseline
python scripts/train_all_datasets.py --model prefusion
python scripts/train_all_datasets.py --model postfusion
```

### 手動指定數據集

```bash
# 在特定數據集上訓練單個模型
python scripts/train_prefusion_cleaned.py --dataset SemEval2014_Laptop
python scripts/train_postfusion_cleaned.py --dataset SemEval2014_Restaurant
```

**詳細說明：** 參見 [docs/CONFIG_USAGE.md](CONFIG_USAGE.md)

---

## 🚀 一鍵執行完整實驗流程

**✨ 現在自動使用 `configs/experiment_config.py` 中配置的數據集！**

```bash
# 執行所有步驟（資料清理 + 訓練所有模型 + 生成表格圖表 + 產生報告）
python scripts/run_full_experiment.py

# 跳過資料清理（如果已清理過） ⭐ 推薦
python scripts/run_full_experiment.py --skip-cleaning

# 只訓練特定模型類型
python scripts/run_full_experiment.py --models baseline
python scripts/run_full_experiment.py --models postfusion

# 只生成分析報告（跳過訓練）
python scripts/run_full_experiment.py --skip-training --skip-cleaning
```

**完整流程包含：**
1. 資料清理與驗證（移除 conflict 標籤）
2. **在所有配置的數據集上**訓練 Baseline、Pre-Fusion、Post-Fusion 模型
3. 生成比較表格（Table 1-3）
4. 生成高品質圖表（Figure 1-5，300 DPI）
5. 產生完整實驗報告（包含所有數據集）

**預估耗時：**
- 單個數據集：1-1.5 小時/數據集
- 4 個數據集（目前配置）：4-6 小時

**使用的數據集：** 根據 `configs/experiment_config.py` 中的 `DATASETS` 設定

---

## 📋 資料處理

### 資料清理

```bash
# 清理 Aspect-Level 資料（移除標註衝突）
python scripts/clean_aspect_data.py

# 驗證清理結果
python scripts/verify_cleaning.py
```

### 資料轉換

```bash
# 轉換 Aspect-Level → Sentence-Level
python scripts/convert_to_sentence_level.py
```

## 🎓 模型訓練（Aspect-Level，已清理資料）

### Baseline 模型

```bash
# 訓練 Baseline（Hard Attention Masking）
python scripts/train_baseline_cleaned.py --num_epochs 30 --batch_size 32
```

### Pre-Fusion 模型（Attention → BiLSTM）

```bash
# 訓練單層（--auto 自動選擇最佳層數）
python scripts/train_prefusion_cleaned.py --layers 2 --auto

# 訓練多層配置
python scripts/train_prefusion_cleaned.py --layers 2 3 4 5
```

### Post-Fusion 模型（BiLSTM → Attention）

```bash
# 訓練多層配置（2-5層）
python scripts/train_postfusion_cleaned.py --layers 2 3 4 5

# 訓練單一配置
python scripts/train_postfusion_cleaned.py --layers 3
```

## 📊 結果分析與視覺化

### 生成論文材料

```bash
# 生成比較表格（Table 1-3）
python scripts/generate_comparison_tables.py

# 生成高品質圖表（Figure 1-5，300 DPI）
python scripts/generate_figures.py
```

### 注意力視覺化

```bash
# 視覺化注意力權重
python scripts/visualize_attention.py \
    --checkpoint "outputs/checkpoints/postfusion_3layer_cleaned/best_model.pt" \
    --model_type post_fusion \
    --layers 3
```

## ⚙️ 常用參數

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--num_epochs` | 30 | 訓練輪數 |
| `--batch_size` | 32 | 批次大小 |
| `--learning_rate` | 0.001 | 學習率 |
| `--patience` | 5 | Early stopping 耐心值 |
| `--layers` | - | 模型層數（Pre/Post-Fusion） |
| `--auto` | False | 自動選擇最佳層數（Pre-Fusion） |

### 視覺化參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--checkpoint` | - | 模型檢查點路徑 |
| `--model_type` | - | baseline / pre_fusion / post_fusion |
| `--layers` | - | 模型層數 |

## 📂 輸出位置

```
outputs/
├── checkpoints/                    # 訓練檢查點
│   ├── baseline_cleaned/
│   ├── prefusion_{N}layer_cleaned/
│   └── postfusion_{N}layer_cleaned/
├── experiments/                    # 實驗結果 JSON
│   └── aspect_level_cleaned/
├── paper_materials/                # 論文材料
│   ├── tables/                    # 比較表格（CSV + Markdown）
│   │   ├── table1_data_cleaning_comparison.csv
│   │   ├── table2_fusion_strategy_comparison.csv
│   │   ├── table3_prefusion_vs_postfusion.csv
│   │   └── all_results_summary.csv
│   └── figures/                   # 高品質圖表（PNG + PDF，300 DPI）
│       ├── fig1_cleaning_improvement.{png,pdf}
│       ├── fig2_layers_impact.{png,pdf}
│       ├── fig3_fusion_comparison.{png,pdf}
│       ├── fig4_roc_curves.{png,pdf}
│       └── fig5_confusion_matrices.{png,pdf}
└── data/
    └── processed/
        └── SemEval2014_Restaurant_cleaned/  # 清理後資料
```

## 🏆 最佳實踐結果

根據實驗結果，推薦配置：

```bash
# 最佳模型：Post-Fusion 3-layer
python scripts/train_postfusion_cleaned.py --layers 3 --num_epochs 30 --batch_size 32

# 預期性能：
# - Accuracy: 74.54%
# - Macro-F1: 67.63%
# - AUC (Macro): 0.8495
```

## 📝 實驗重現

完整重現所有實驗結果：

```bash
# Step 1: 清理資料
python scripts/clean_aspect_data.py
python scripts/verify_cleaning.py

# Step 2: 訓練所有模型
python scripts/train_baseline_cleaned.py
python scripts/train_prefusion_cleaned.py --layers 2
python scripts/train_postfusion_cleaned.py --layers 2 3 4 5

# Step 3: 生成論文材料
python scripts/generate_comparison_tables.py
python scripts/generate_figures.py
```
