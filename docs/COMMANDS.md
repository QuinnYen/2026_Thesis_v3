# 命令速查表

## 📋 資料處理

```bash
# 轉換 Aspect-Level → Sentence-Level
python scripts/convert_to_sentence_level.py
```

## 🎓 模型訓練

### Sentence-Level

```bash
# Base (無 Attention)
python scripts/train_sentence_level_baseline.py --model_type base --hidden_size 128

# Pre-Fusion (Attention → BiLSTM)
python scripts/train_sentence_level_baseline.py --model_type pre_fusion --hidden_size 128

# Post-Fusion (BiLSTM → Attention)
python scripts/train_sentence_level_baseline.py --model_type post_fusion --hidden_size 128

# Hidden Size = 256
python scripts/train_sentence_level_baseline.py --model_type base --hidden_size 256
```

### Aspect-Level

```bash
# Baseline
python scripts/train_baseline.py --num_epochs 30 --batch_size 32
```

## 📊 評估與分析

```bash
# 評估單一模型
python scripts/evaluate_sentence_level_model.py \
    --model_type base \
    --checkpoint "outputs/experiments/.../best_model.pt"

# 比較模型
python scripts/compare_sentence_level_models.py

# 生成總結報告
python scripts/generate_baseline_summary.py
```

## 🚀 批次執行

```bash
# 執行所有實驗
python scripts/run_all_experiments.py
```

## ⚙️ 常用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model_type` | - | base / pre_fusion / post_fusion |
| `--hidden_size` | 128 | 128 / 256 |
| `--num_epochs` | 30 | 訓練輪數 |
| `--batch_size` | 32 | 批次大小 |
| `--learning_rate` | 0.001 | 學習率 |
| `--patience` | 5 | Early stopping 耐心值 |
| `--dataset` | SemEval2014_Restaurant | 資料集名稱 |

## 📂 輸出位置

```
outputs/
├── experiments/              # 訓練結果
├── evaluations/              # 評估報告
├── comparisons/              # 模型比較
└── baseline_summary/         # 總結報告
```
