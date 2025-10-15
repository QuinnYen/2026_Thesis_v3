# 配置系統使用說明

## 問題

之前訓練腳本只會在 `SemEval2014_Restaurant` 數據集上訓練，無法執行其他數據集（如 Laptop 和 2016）。

## 解決方案

新增了集中式配置系統，所有訓練腳本現在從 `configs/experiment_config.py` 讀取配置。

## 快速開始

### 1. 配置要訓練的數據集

編輯 `configs/experiment_config.py`:

```python
# 取消註解你想要訓練的數據集
DATASETS = [
    "SemEval2014_Restaurant",  # ✓ 已啟用
    "SemEval2014_Laptop",      # ✓ 已啟用
    # "SemEval2016_Restaurant",  # ✗ 未啟用（如需使用請取消註解）
]
```

### 2. 執行訓練

#### 方法一：使用包裝腳本（推薦）

在所有配置的數據集上訓練:

```bash
# 訓練所有模型（Baseline + Pre-Fusion + Post-Fusion）
python scripts/train_all_datasets.py --model all

# 只訓練特定模型
python scripts/train_all_datasets.py --model prefusion
```

#### 方法二：手動指定數據集

訓練單個模型在特定數據集上:

```bash
# 在 Laptop 數據集上訓練 Pre-Fusion
python scripts/train_prefusion_cleaned.py --dataset SemEval2014_Laptop

# 在 Restaurant 數據集上訓練 Post-Fusion
python scripts/train_postfusion_cleaned.py --dataset SemEval2014_Restaurant
```

### 3. 修改訓練參數

在 `configs/experiment_config.py` 中修改:

```python
PREFUSION_CONFIG = {
    "layers": [2, 3],       # 訓練 2 層和 3 層
    "num_epochs": 50,       # 增加到 50 個 epoch
    "batch_size": 64,       # 增大 batch size
    "learning_rate": 2e-5,  # 調整學習率
    ...
}
```

## 輸出位置

訓練結果會自動保存到對應的數據集目錄:

```
outputs/
└── experiments/
    └── aspect_level_v2/
        ├── SemEval2014_Restaurant/
        │   ├── baseline_cleaned/
        │   ├── pre_fusion_2layer_cleaned/
        │   └── post_fusion_3layer_cleaned/
        └── SemEval2014_Laptop/
            ├── baseline_cleaned/
            ├── pre_fusion_2layer_cleaned/
            └── post_fusion_3layer_cleaned/
```

## 添加新數據集

如果你有新的數據集（例如 SemEval2016）:

1. 確保數據文件存在:
   ```
   data/processed/aspect_level_cleaned/
   ├── SemEval2016_Restaurant_train_cleaned.csv
   └── SemEval2016_Restaurant_test_cleaned.csv
   ```

2. 在 `experiment_config.py` 中添加路徑映射:
   ```python
   DATASET_PATHS = {
       "SemEval2016_Restaurant": {
           "train": "data/processed/aspect_level_cleaned/SemEval2016_Restaurant_train_cleaned.csv",
           "test": "data/processed/aspect_level_cleaned/SemEval2016_Restaurant_test_cleaned.csv",
       },
   }
   ```

3. 在 `DATASETS` 列表中啟用:
   ```python
   DATASETS = [
       "SemEval2014_Restaurant",
       "SemEval2014_Laptop",
       "SemEval2016_Restaurant",  # 新增
   ]
   ```

## 檢查配置

查看當前配置:

```bash
python configs/experiment_config.py
```

輸出示例:
```
================================================================================
當前實驗配置
================================================================================

要執行的數據集 (2 個):
  - SemEval2014_Restaurant
  - SemEval2014_Laptop

Baseline 配置:
  num_epochs: 30
  batch_size: 32
  ...
```

## 注意事項

1. **數據文件必須存在**: 確保在 `DATASET_PATHS` 中指定的文件路徑有效
2. **記憶體使用**: 如果同時訓練多個數據集，需要足夠的磁碟空間和記憶體
3. **訓練時間**: 每個數據集每個模型約需要 30-60 分鐘（依硬體而定）

## 範例工作流程

完整的實驗流程:

```bash
# 1. 配置數據集（編輯 configs/experiment_config.py）
# 2. 檢查配置
python configs/experiment_config.py

# 3. 執行完整實驗
python scripts/train_all_datasets.py --model all

# 4. 查看結果
# 結果保存在 outputs/experiments/aspect_level_v2/ 下的各個數據集目錄中
```
