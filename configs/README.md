# 實驗配置說明

## 快速開始

### 1. 修改要執行的數據集

編輯 `configs/experiment_config.py`，取消註解你想要執行的數據集：

```python
DATASETS = [
    "SemEval2014_Restaurant",  # 已啟用
    "SemEval2014_Laptop",      # 取消註解以啟用
    # "SemEval2016_Restaurant",  # 取消註解以啟用
]
```

### 2. 執行訓練

使用包裝腳本在所有配置的數據集上訓練:

```bash
# 訓練所有模型在所有數據集上
python scripts/train_all_datasets.py --model all

# 只訓練 Baseline
python scripts/train_all_datasets.py --model baseline

# 只訓練 Pre-Fusion
python scripts/train_all_datasets.py --model prefusion

# 只訓練 Post-Fusion
python scripts/train_all_datasets.py --model postfusion
```

或者手動指定數據集訓練單個模型:

```bash
python scripts/train_prefusion_cleaned.py --dataset SemEval2014_Laptop
python scripts/train_postfusion_cleaned.py --dataset SemEval2016_Restaurant
```

### 3. 修改模型參數

在 `experiment_config.py` 中修改模型訓練參數:

```python
BASELINE_CONFIG = {
    "num_epochs": 30,      # 訓練輪數
    "batch_size": 32,      # 批次大小
    "learning_rate": 1e-5, # 學習率
    ...
}
```

## 配置文件結構

```
configs/
├── experiment_config.py  # 主要配置文件
└── README.md            # 本文件
```

## 添加新數據集

1. 在 `DATASET_PATHS` 中添加數據集路徑映射
2. 在 `DATASETS` 列表中添加數據集名稱
3. 確保數據文件存在於指定路徑

示例:

```python
DATASET_PATHS = {
    "MyNewDataset": {
        "train": "data/processed/aspect_level_cleaned/MyNewDataset_train_cleaned.csv",
        "test": "data/processed/aspect_level_cleaned/MyNewDataset_test_cleaned.csv",
    },
}

DATASETS = [
    "MyNewDataset",
]
```
