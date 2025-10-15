# 快速開始指南

## 🎯 目標

在所有數據集上訓練和評估 ABSA 模型（Baseline、Pre-Fusion、Post-Fusion）。

## ✅ 前置準備檢查

### 1. 檢查數據集是否已清理

```bash
ls data/processed/aspect_level_cleaned/*.csv
```

應該看到 8 個文件（4個數據集 × 2個splits）：
- ✅ SemEval2014_Restaurant (train + test)
- ✅ SemEval2014_Laptop (train + test)
- ✅ SemEval2016_Restaurant (train + test)
- ✅ SemEval2016_Laptop (train + test)

### 2. 檢查配置

```bash
python configs/experiment_config.py
```

應該顯示 4 個數據集已配置。

## 🚀 執行方式

### ⚡ 使用 GPU 訓練（強烈推薦）

你有 **RTX 3090**，使用 GPU 可以加速 **10-20 倍**！

**最簡單 - 雙擊批次文件：**
- `train_gpu.bat` - 訓練所有模型
- `run_with_gpu.bat` - 完整實驗流程

**或使用命令行：**
```bash
# 使用虛擬環境的 Python（啟用 GPU）
env\Scripts\python.exe scripts\train_all_datasets.py --model all
```

⚠️ **重要**: 不要直接用 `python`，會使用系統 Python（CPU版本）！

詳見：**[GPU 設置指南](GPU_SETUP.md)**

### 方式一：完整自動化流程

```bash
# 使用 GPU（推薦）
env\Scripts\python.exe scripts\run_full_experiment.py --skip-cleaning

# 或雙擊
run_with_gpu.bat
```

這將：
1. ✅ 在所有 4 個數據集上訓練所有 3 個模型
2. ✅ 生成比較表格和圖表
3. ✅ 產生完整報告

**預估時間：**
- CPU: 4-6 小時
- **GPU (RTX 3090): 30-60 分鐘** 🚀

### 方式二：分步執行

如果需要更多控制：

#### Step 1: 訓練所有模型
```bash
python scripts/train_all_datasets.py --model all
```

#### Step 2: 生成分析報告
```bash
python scripts/generate_comparison_tables.py
python scripts/generate_figures.py
```

### 方式三：訓練單個模型

```bash
# 只訓練 Baseline
python scripts/train_all_datasets.py --model baseline

# 或在特定數據集上訓練
python scripts/train_baseline_cleaned.py --dataset SemEval2016_Restaurant
```

## 📊 查看結果

### 訓練結果位置

```
outputs/experiments/aspect_level_v2/
├── SemEval2014_Restaurant/
│   ├── baseline_cleaned/
│   ├── pre_fusion_2layer_cleaned/
│   └── post_fusion_3layer_cleaned/
├── SemEval2014_Laptop/
├── SemEval2016_Restaurant/
└── SemEval2016_Laptop/
```

### 查看實驗報告

```bash
cat outputs/paper_materials/experiment_report.txt
```

### 查看單個模型結果

```bash
cat outputs/experiments/aspect_level_v2/SemEval2014_Restaurant/baseline_cleaned/results/experiment_result.json
```

## ⚙️ 自定義配置

### 修改要訓練的數據集

編輯 `configs/experiment_config.py`:

```python
DATASETS = [
    "SemEval2014_Restaurant",  # 註解掉不需要的
    # "SemEval2014_Laptop",
    # "SemEval2016_Restaurant",
    # "SemEval2016_Laptop",
]
```

### 修改訓練參數

編輯 `configs/experiment_config.py`:

```python
BASELINE_CONFIG = {
    "num_epochs": 50,          # 增加訓練輪數
    "batch_size": 64,          # 增大批次大小
    "learning_rate": 2e-5,     # 調整學習率
    ...
}
```

## 🔧 常見問題

### Q: 如果只想在 Restaurant 數據集上訓練？

A: 編輯 `configs/experiment_config.py`，只保留：
```python
DATASETS = ["SemEval2014_Restaurant"]
```

### Q: 如何重新清理數據？

A: 執行：
```bash
python scripts/clean_aspect_data.py
python scripts/process_test_data.py
```

### Q: 訓練中斷了怎麼辦？

A: 使用 `train_all_datasets.py` 會自動跳過已完成的訓練。或者手動訓練缺失的部分：
```bash
python scripts/train_baseline_cleaned.py --dataset SemEval2016_Laptop
```

### Q: 如何查看當前配置？

A: 執行：
```bash
python configs/experiment_config.py
```

## 📚 詳細文檔

- [命令速查表](docs/COMMANDS.md) - 所有可用命令
- [配置使用說明](docs/CONFIG_USAGE.md) - 配置系統詳解
- [數據摘要](DATA_SUMMARY.md) - 數據集詳細信息
- [配置更新日誌](configs/CHANGELOG.md) - 配置系統更新記錄

## 💡 推薦工作流程

1. **首次運行**:
   ```bash
   # 檢查配置
   python configs/experiment_config.py

   # 執行完整實驗（數據已清理）
   python scripts/run_full_experiment.py --skip-cleaning
   ```

2. **添加新數據集**:
   - 將原始數據放入 `data/raw/`
   - 運行清理腳本
   - 在配置文件中啟用
   - 重新執行訓練

3. **調整參數**:
   - 修改 `configs/experiment_config.py`
   - 運行 `train_all_datasets.py`
   - 比較結果

## 🎉 完成！

現在你可以開始訓練了。祝實驗順利！
