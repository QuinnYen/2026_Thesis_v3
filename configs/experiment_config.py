"""
實驗配置文件

在此配置所有要執行的數據集和模型參數
所有訓練腳本會從這裡讀取配置
"""

from pathlib import Path

# ============================================================================
# 數據集配置
# ============================================================================

# 要執行的數據集列表（取消註解即可執行）
DATASETS = [
    "SemEval2014_Restaurant",
    "SemEval2014_Laptop",
    "SemEval2016_Restaurant",  # ✅ 已清理完成
    "SemEval2016_Laptop",      # ✅ 已清理完成
]

# 數據集路徑映射
DATASET_PATHS = {
    "SemEval2014_Restaurant": {
        "train": "data/processed/aspect_level_cleaned/SemEval2014_Restaurant_train_cleaned.csv",
        "test": "data/processed/aspect_level_cleaned/SemEval2014_Restaurant_test_cleaned.csv",
    },
    "SemEval2014_Laptop": {
        "train": "data/processed/aspect_level_cleaned/SemEval2014_Laptop_train_cleaned.csv",
        "test": "data/processed/aspect_level_cleaned/SemEval2014_Laptop_test_cleaned.csv",
    },
    "SemEval2016_Restaurant": {
        "train": "data/processed/aspect_level_cleaned/SemEval2016_Restaurant_train_cleaned.csv",
        "test": "data/processed/aspect_level_cleaned/SemEval2016_Restaurant_test_cleaned.csv",
    },
    "SemEval2016_Laptop": {
        "train": "data/processed/aspect_level_cleaned/SemEval2016_Laptop_train_cleaned.csv",
        "test": "data/processed/aspect_level_cleaned/SemEval2016_Laptop_test_cleaned.csv",
    },
}

# ============================================================================
# 模型訓練配置
# ============================================================================

# Baseline 配置
BASELINE_CONFIG = {
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,       # 調高學習率：1e-5 -> 1e-3
    "warmup_steps": 100,
    "max_length": 128,
    "dropout": 0.3,              # 調高 dropout：0.1 -> 0.3
    "weight_decay": 0.01,
    # 詞向量配置
    "use_pretrained_embeddings": True,  # 啟用 GloVe 預訓練詞向量
    "freeze_embeddings": False,         # 是否凍結詞向量（False=微調）
    "oov_strategy": "random",           # OOV 處理策略: 'random', 'zeros', 'mean'
    # 軟遮罩配置（Soft Masking + Context Window）
    "use_soft_mask": True,       # 啟用軟遮罩（False=硬遮罩）
    "penalty_weight": 5.0,       # 距離懲罰強度（建議範圍: 3-10）
    "context_window": 2,         # 上下文窗口大小（建議範圍: 1-4）
    "normalize_penalty": True,   # 啟用懲罰歸一化（根據attention scores動態調整）
}

# 實驗用：不同的軟遮罩配置組合（供超參數搜索使用）
SOFT_MASK_EXPERIMENTS = {
    # 配置 1：硬遮罩（基線）
    "hard_mask": {
        "use_soft_mask": False,
        "penalty_weight": 0.0,
        "context_window": 0,
        "normalize_penalty": False,
    },
    # 配置 2：軟遮罩 + 固定懲罰
    "soft_mask_fixed": {
        "use_soft_mask": True,
        "penalty_weight": 5.0,
        "context_window": 2,
        "normalize_penalty": False,
    },
    # 配置 3：軟遮罩 + 歸一化懲罰（推薦）
    "soft_mask_normalized": {
        "use_soft_mask": True,
        "penalty_weight": 5.0,
        "context_window": 2,
        "normalize_penalty": True,
    },
    # 配置 4：強懲罰 + 大窗口
    "soft_mask_strong": {
        "use_soft_mask": True,
        "penalty_weight": 10.0,
        "context_window": 3,
        "normalize_penalty": True,
    },
    # 配置 5：弱懲罰 + 小窗口
    "soft_mask_weak": {
        "use_soft_mask": True,
        "penalty_weight": 3.0,
        "context_window": 1,
        "normalize_penalty": True,
    },
}

# Pre-Fusion 配置
PREFUSION_CONFIG = {
    "layers": [2, 3, 4, 5],  # 訓練 2-5 層（與 Post-Fusion 一致）
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,  # 調高學習率：1e-5 -> 1e-3
    "warmup_steps": 100,
    "max_length": 128,
    "dropout": 0.3,  # 調高 dropout：0.1 -> 0.3
    "weight_decay": 0.01,
    "auto_select_best": False,  # 關閉自動模式，訓練所有層數
    # 詞向量配置
    "use_pretrained_embeddings": True,  # 啟用 GloVe
    "freeze_embeddings": False,
    "oov_strategy": "random",
}

# Post-Fusion 配置
POSTFUSION_CONFIG = {
    "layers": [2, 3, 4, 5],  # 要訓練的層數列表
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,  # 調高學習率：1e-5 -> 1e-3
    "warmup_steps": 100,
    "max_length": 128,
    "dropout": 0.3,  # 調高 dropout：0.1 -> 0.3
    "weight_decay": 0.01,
    # 詞向量配置
    "use_pretrained_embeddings": True,  # 啟用 GloVe
    "freeze_embeddings": False,
    "oov_strategy": "random",
}

# ============================================================================
# 輸出路徑配置
# ============================================================================

def get_output_dir(dataset_name: str, model_type: str, **kwargs) -> Path:
    """
    生成輸出目錄路徑

    Args:
        dataset_name: 數據集名稱 (例如 "SemEval2014_Restaurant")
        model_type: 模型類型 ("baseline", "pre_fusion", "post_fusion")
        **kwargs: 額外參數 (例如 layers=2)

    Returns:
        Path: 輸出目錄路徑
    """
    base_dir = Path(__file__).resolve().parent.parent
    exp_dir = base_dir / "outputs" / "experiments" / "aspect_level_v2" / dataset_name

    if model_type == "baseline":
        return exp_dir / "baseline_cleaned"
    elif model_type == "pre_fusion":
        layers = kwargs.get("layers", 2)
        return exp_dir / f"pre_fusion_{layers}layer_cleaned"
    elif model_type == "post_fusion":
        layers = kwargs.get("layers", 2)
        return exp_dir / f"post_fusion_{layers}layer_cleaned"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_dataset_path(dataset_name: str, split: str = "train") -> Path:
    """
    獲取數據集路徑

    Args:
        dataset_name: 數據集名稱
        split: "train" 或 "test"

    Returns:
        Path: 數據集檔案路徑
    """
    base_dir = Path(__file__).resolve().parent.parent

    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    rel_path = DATASET_PATHS[dataset_name][split]
    return base_dir / rel_path


# ============================================================================
# 實用函數
# ============================================================================

def print_config():
    """印出當前配置"""
    print("=" * 80)
    print("當前實驗配置")
    print("=" * 80)
    print(f"\n要執行的數據集 ({len(DATASETS)} 個):")
    for dataset in DATASETS:
        print(f"  - {dataset}")

    print(f"\nBaseline 配置:")
    for key, value in BASELINE_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nPre-Fusion 配置:")
    for key, value in PREFUSION_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nPost-Fusion 配置:")
    for key, value in POSTFUSION_CONFIG.items():
        print(f"  {key}: {value}")

    print("=" * 80)


if __name__ == "__main__":
    print_config()
