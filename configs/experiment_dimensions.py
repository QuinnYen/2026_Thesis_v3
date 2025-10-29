"""
多維度實驗配置文件

定義論文中五個維度的實驗組合：
1. 基礎架構 (RQ1核心)
2. 詞向量提升
3. 上下文編碼器
4. 句法信息
5. 多種子穩定性
"""

from typing import Dict, List, Any
from pathlib import Path

# ============================================================================
# 維度 1: 基礎架構 (RQ1核心)
# ============================================================================

DIMENSION_1_ARCHITECTURE = {
    "name": "基礎架構對比",
    "description": "比較不同融合策略和模型深度的影響",
    "research_questions": ["RQ1: 融合策略效能差異", "RQ2: 模型深度影響"],
    "paper_section": "第5.1節「基礎架構比較」",

    "experiment_groups": {
        # A1: Baseline (無 Attention)
        "A1_baseline_no_attention": {
            "model_type": "baseline_no_attention",
            "use_attention": False,
            "layers": [1, 2, 3, 4, 5],
            "description": "Baseline（無Attention）多層實驗"
        },

        # A2: Pre-Fusion (Attention → BiLSTM)
        "A2_prefusion_bilstm": {
            "model_type": "prefusion",
            "encoder_type": "bilstm",
            "layers": [2, 3, 4, 5],
            "description": "Pre-Fusion（Attention → BiLSTM）"
        },

        # A3: Post-Fusion (BiLSTM → Attention)
        "A3_postfusion_bilstm": {
            "model_type": "postfusion",
            "encoder_type": "bilstm",
            "layers": [2, 3, 4, 5],
            "description": "Post-Fusion（BiLSTM → Attention）"
        },

        # A4: Pre-Fusion (Attention → LSTM)
        "A4_prefusion_lstm": {
            "model_type": "prefusion",
            "encoder_type": "lstm",
            "bidirectional": False,
            "layers": [2, 3, 4, 5],
            "description": "Pre-Fusion（Attention → LSTM）"
        },

        # A5: Post-Fusion (LSTM → Attention)
        "A5_postfusion_lstm": {
            "model_type": "postfusion",
            "encoder_type": "lstm",
            "bidirectional": False,
            "layers": [2, 3, 4, 5],
            "description": "Post-Fusion（LSTM → Attention）"
        }
    }
}

# ============================================================================
# 維度 2: 詞向量提升
# ============================================================================

DIMENSION_2_EMBEDDINGS = {
    "name": "詞向量提升",
    "description": "測試核心論點在更好特徵下是否成立",
    "research_questions": ["驗證不同詞向量下，融合策略的優劣是否一致"],
    "paper_section": "第5.2節「詞向量對融合策略的影響」",

    "experiment_groups": {
        # B1: 隨機初始化詞向量
        "B1_random_init": {
            "embedding_type": "random",
            "embedding_dim": 300,
            "freeze_embeddings": False,
            "models": ["baseline", "prefusion", "postfusion"],
            "layers": 2,  # 使用最佳層數
            "description": "隨機初始化詞向量"
        },

        # B2: GloVe-100d
        "B2_glove_100d": {
            "embedding_type": "glove",
            "embedding_dim": 100,
            "freeze_embeddings": False,
            "models": ["baseline", "prefusion", "postfusion"],
            "layers": 2,
            "description": "GloVe-100d 預訓練詞向量"
        },

        # B3: GloVe-300d
        "B3_glove_300d": {
            "embedding_type": "glove",
            "embedding_dim": 300,
            "freeze_embeddings": False,
            "models": ["baseline", "prefusion", "postfusion"],
            "layers": 2,
            "description": "GloVe-300d 預訓練詞向量"
        },

        # B4: Word2Vec
        "B4_word2vec": {
            "embedding_type": "word2vec",
            "embedding_dim": 300,
            "freeze_embeddings": False,
            "models": ["baseline", "prefusion", "postfusion"],
            "layers": 2,
            "description": "Word2Vec 預訓練詞向量"
        }
    }
}

# ============================================================================
# 維度 3: 上下文編碼器
# ============================================================================

DIMENSION_3_ENCODER = {
    "name": "上下文編碼器",
    "description": "測試在SOTA編碼器下的表現",
    "research_questions": ["使用SOTA編碼器後，融合策略是否仍有差異"],
    "paper_section": "第5.3節「預訓練模型對融合策略的影響」",

    "experiment_groups": {
        # C1: BiLSTM編碼器
        "C1_bilstm_encoder": {
            "encoder_type": "bilstm",
            "models": ["prefusion", "postfusion"],
            "layers": 2,
            "description": "BiLSTM編碼器"
        },

        # C2: BERT編碼器
        "C2_bert_encoder": {
            "encoder_type": "bert",
            "bert_model": "bert-base-uncased",
            "models": ["prefusion", "postfusion"],
            "freeze_bert": False,
            "description": "BERT編碼器"
        }
    }
}

# ============================================================================
# 維度 4: 句法信息
# ============================================================================

DIMENSION_4_SYNTAX = {
    "name": "句法信息",
    "description": "測試在結構化信息下的表現",
    "research_questions": ["加入語法結構後，融合策略的差異"],
    "paper_section": "第5.4節「句法信息對融合策略的影響」",

    "experiment_groups": {
        # D1: 無句法信息
        "D1_no_syntax": {
            "use_syntax": False,
            "models": ["prefusion", "postfusion"],
            "layers": 2,
            "description": "無句法信息"
        },

        # D2: 加入依存樹
        "D2_dependency_tree": {
            "use_syntax": True,
            "syntax_type": "dependency",
            "gnn_type": "gcn",
            "gnn_layers": 2,
            "models": ["prefusion", "postfusion"],
            "layers": 2,
            "description": "加入依存句法樹 + GNN"
        }
    }
}

# ============================================================================
# 維度 5: 多種子穩定性
# ============================================================================

DIMENSION_5_STABILITY = {
    "name": "多種子穩定性測試",
    "description": "證明結果可靠性",
    "research_questions": ["結果的穩定性和統計顯著性"],
    "paper_section": "第5.5節「穩定性分析」",

    "experiment_groups": {
        # E1: 單一種子
        "E1_single_seed": {
            "seeds": [42],
            "description": "單一種子（seed=42）"
        },

        # E2: 多種子測試
        "E2_multiple_seeds": {
            "seeds": [42, 43, 44, 45, 46],
            "report_stats": True,  # 報告均值±標準差
            "run_significance_test": True,  # 執行顯著性檢驗
            "description": "多種子測試（5個隨機種子）"
        }
    }
}

# ============================================================================
# 維度組合策略
# ============================================================================

# 定義哪些維度可以組合
DIMENSION_COMBINATIONS = {
    # 基礎實驗：維度1（必須先執行）
    "basic": {
        "dimensions": [1],
        "description": "基礎架構實驗（RQ1核心）",
        "priority": 1
    },

    # 詞向量實驗：維度1 + 維度2
    "embeddings": {
        "dimensions": [1, 2],
        "description": "詞向量影響實驗",
        "priority": 2,
        "depends_on": ["basic"]
    },

    # 編碼器實驗：維度1 + 維度3
    "encoder": {
        "dimensions": [1, 3],
        "description": "上下文編碼器實驗",
        "priority": 3,
        "depends_on": ["basic"]
    },

    # 句法實驗：維度1 + 維度4
    "syntax": {
        "dimensions": [1, 4],
        "description": "句法信息實驗",
        "priority": 4,
        "depends_on": ["basic"]
    },

    # 穩定性測試：對所有最佳模型執行維度5
    "stability": {
        "dimensions": [5],
        "description": "穩定性測試（應用於所有最佳模型）",
        "priority": 5,
        "depends_on": ["basic", "embeddings", "encoder", "syntax"]
    },

    # 完整實驗：所有維度
    "full": {
        "dimensions": [1, 2, 3, 4, 5],
        "description": "完整五維度實驗",
        "priority": 99
    }
}

# ============================================================================
# 實驗執行配置
# ============================================================================

EXECUTION_CONFIG = {
    # 平行執行設定
    "parallel_execution": {
        "enabled": False,  # 是否啟用平行執行
        "max_workers": 2,  # 最大平行任務數（根據GPU數量調整）
        "gpu_allocation": "auto"  # "auto" 或指定 GPU ID 列表
    },

    # 斷點續傳
    "checkpoint": {
        "enabled": True,
        "save_frequency": "per_experiment",  # "per_epoch" 或 "per_experiment"
        "checkpoint_dir": "outputs/experiment_checkpoints"
    },

    # 實驗追蹤
    "tracking": {
        "enabled": True,
        "log_dir": "outputs/experiment_logs",
        "save_intermediate_results": True
    },

    # 資源管理
    "resource_management": {
        "max_experiment_time": 7200,  # 單個實驗最大時間（秒）
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.001
        }
    }
}

# ============================================================================
# 數據集配置（繼承自 experiment_config.py）
# ============================================================================

# 從原有配置導入
try:
    from experiment_config import DATASETS as BASE_DATASETS
    DATASETS = BASE_DATASETS
except ImportError:
    # 預設數據集
    DATASETS = [
        "SemEval2014_Restaurant",
        "SemEval2014_Laptop",
        "SemEval2016_Restaurant",
        "SemEval2016_Laptop"
    ]

# ============================================================================
# 輔助函數
# ============================================================================

def get_dimension_config(dimension_id: int) -> Dict[str, Any]:
    """
    獲取指定維度的配置

    Args:
        dimension_id: 維度ID (1-5)

    Returns:
        維度配置字典
    """
    dimensions = {
        1: DIMENSION_1_ARCHITECTURE,
        2: DIMENSION_2_EMBEDDINGS,
        3: DIMENSION_3_ENCODER,
        4: DIMENSION_4_SYNTAX,
        5: DIMENSION_5_STABILITY
    }

    if dimension_id not in dimensions:
        raise ValueError(f"無效的維度ID: {dimension_id}。必須在 1-5 之間。")

    return dimensions[dimension_id]


def get_combination_config(combination_name: str) -> Dict[str, Any]:
    """
    獲取維度組合配置

    Args:
        combination_name: 組合名稱 ("basic", "embeddings", "encoder", "syntax", "stability", "full")

    Returns:
        組合配置字典
    """
    if combination_name not in DIMENSION_COMBINATIONS:
        raise ValueError(f"無效的組合名稱: {combination_name}")

    return DIMENSION_COMBINATIONS[combination_name]


def generate_experiment_plan(dimensions: List[int], datasets: List[str] = None) -> Dict[str, Any]:
    """
    生成實驗計劃

    Args:
        dimensions: 要執行的維度列表
        datasets: 要使用的數據集列表（None=使用所有數據集）

    Returns:
        實驗計劃字典
    """
    if datasets is None:
        datasets = DATASETS

    plan = {
        "dimensions": dimensions,
        "datasets": datasets,
        "experiments": []
    }

    # 為每個維度生成實驗
    for dim_id in dimensions:
        dim_config = get_dimension_config(dim_id)

        for group_name, group_config in dim_config["experiment_groups"].items():
            for dataset in datasets:
                experiment = {
                    "dimension": dim_id,
                    "dimension_name": dim_config["name"],
                    "group": group_name,
                    "dataset": dataset,
                    "config": group_config.copy()
                }
                plan["experiments"].append(experiment)

    return plan


def estimate_experiment_time(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    估算實驗執行時間

    Args:
        plan: 實驗計劃

    Returns:
        時間估算字典
    """
    # 基準時間（分鐘）
    base_time_per_model = 15  # 單個模型訓練時間

    total_experiments = len(plan["experiments"])
    total_time_minutes = total_experiments * base_time_per_model

    return {
        "total_experiments": total_experiments,
        "estimated_time_minutes": total_time_minutes,
        "estimated_time_hours": total_time_minutes / 60,
        "estimated_time_days": total_time_minutes / 60 / 24
    }


def print_experiment_summary(plan: Dict[str, Any]):
    """印出實驗摘要"""
    print("=" * 80)
    print("實驗計劃摘要")
    print("=" * 80)

    print(f"\n維度數量: {len(plan['dimensions'])}")
    print(f"維度: {plan['dimensions']}")

    print(f"\n數據集數量: {len(plan['datasets'])}")
    print(f"數據集: {plan['datasets']}")

    print(f"\n總實驗數: {len(plan['experiments'])}")

    # 按維度分組統計
    dim_counts = {}
    for exp in plan["experiments"]:
        dim_id = exp["dimension"]
        dim_counts[dim_id] = dim_counts.get(dim_id, 0) + 1

    print("\n各維度實驗數量:")
    for dim_id, count in sorted(dim_counts.items()):
        dim_config = get_dimension_config(dim_id)
        print(f"  維度 {dim_id} ({dim_config['name']}): {count} 個實驗")

    # 時間估算
    time_est = estimate_experiment_time(plan)
    print(f"\n預估執行時間:")
    print(f"  總實驗數: {time_est['total_experiments']}")
    print(f"  預估時間: {time_est['estimated_time_hours']:.1f} 小時 ({time_est['estimated_time_days']:.1f} 天)")

    print("=" * 80)


# ============================================================================
# 測試與範例
# ============================================================================

if __name__ == "__main__":
    print("多維度實驗配置測試\n")

    # 測試 1: 基礎實驗計劃
    print("=" * 80)
    print("測試 1: 基礎實驗計劃（維度1）")
    print("=" * 80)
    plan1 = generate_experiment_plan([1], ["SemEval2014_Restaurant"])
    print_experiment_summary(plan1)

    # 測試 2: 詞向量實驗計劃
    print("\n" + "=" * 80)
    print("測試 2: 詞向量實驗計劃（維度1+2）")
    print("=" * 80)
    plan2 = generate_experiment_plan([1, 2], ["SemEval2014_Restaurant"])
    print_experiment_summary(plan2)

    # 測試 3: 完整實驗計劃
    print("\n" + "=" * 80)
    print("測試 3: 完整實驗計劃（所有維度+所有數據集）")
    print("=" * 80)
    plan3 = generate_experiment_plan([1, 2, 3, 4, 5])
    print_experiment_summary(plan3)
