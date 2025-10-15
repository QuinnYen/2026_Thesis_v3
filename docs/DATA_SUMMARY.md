# 數據集摘要

## ✅ 已清理完成的數據集

所有數據集都已經清理並準備好進行訓練。

### SemEval 2014

| 數據集 | 訓練集 | 測試集 |
|--------|--------|--------|
| **Restaurant** | 1,861 句 (3,259 aspects) | 606 句 (1,134 aspects) |
| **Laptop** | 1,413 句 (2,182 aspects) | 422 句 (654 aspects) |

### SemEval 2016

| 數據集 | 訓練集 | 測試集 |
|--------|--------|--------|
| **Restaurant** | 1,157 句 (1,691 aspects) | - 句 (650 aspects) |
| **Laptop** | 1,979 句 (2,771 aspects) | - 句 (801 aspects) |

## 📁 數據文件位置

```
data/
├── raw/
│   ├── SemEval-2014/
│   │   ├── Restaurants_Train_v2.xml
│   │   ├── Restaurants_Test_Data_phaseB.xml
│   │   ├── Laptop_Train_v2.xml
│   │   └── Laptops_Test_Data_phaseB.xml
│   └── SemEval-2016/
│       ├── restaurants_train_sb1.xml
│       ├── restaurants_test_sb1.xml
│       ├── Laptops_Train_sb1.xml
│       └── laptops_test_sb1.xml
│
└── processed/
    └── aspect_level_cleaned/
        ├── SemEval2014_Restaurant_train_cleaned.csv
        ├── SemEval2014_Restaurant_test_cleaned.csv
        ├── SemEval2014_Laptop_train_cleaned.csv
        ├── SemEval2014_Laptop_test_cleaned.csv
        ├── SemEval2016_Restaurant_train_cleaned.csv
        ├── SemEval2016_Restaurant_test_cleaned.csv
        ├── SemEval2016_Laptop_train_cleaned.csv
        └── SemEval2016_Laptop_test_cleaned.csv
```

## 🔧 清理過程

### 訓練集清理
使用規則 A：移除同時包含 positive 和 negative aspects 的句子

- **執行腳本**: `python scripts/clean_aspect_data.py`
- **清理規則**: 移除明顯衝突的句子（positive + negative）
- **保留**: 情感一致的句子和包含 neutral 的句子

### 測試集處理
直接轉換為 CSV 格式，不進行衝突移除

- **執行腳本**: `python scripts/process_test_data.py`
- **處理內容**: XML → CSV 格式轉換
- **保留**: 所有測試樣本（用於評估）

## 📊 數據統計

### 總體統計
- **總訓練句子**: 6,410 句
- **總訓練 aspects**: 9,901 個
- **清理移除率**: 5.49% 句子，8.14% aspects

### 類別分佈 (訓練集)

#### SemEval2014 Restaurant
- Positive: 1,997 (61.28%)
- Negative: 654 (20.07%)
- Neutral: 608 (18.66%)
- 類別不平衡比: 3.28:1

#### SemEval2014 Laptop
- Positive: 934 (42.80%)
- Negative: 799 (36.62%)
- Neutral: 449 (20.58%)
- 類別不平衡比: 2.08:1

## 🚀 如何使用

### 1. 驗證數據品質
```bash
python scripts/verify_cleaning.py
```

### 2. 在所有數據集上訓練
```bash
# 檢查配置
python configs/experiment_config.py

# 開始訓練
python scripts/train_all_datasets.py --model all
```

### 3. 在特定數據集上訓練
```bash
python scripts/train_prefusion_cleaned.py --dataset SemEval2016_Restaurant
python scripts/train_postfusion_cleaned.py --dataset SemEval2016_Laptop
```

## 📝 格式說明

### CSV 格式
所有清理後的數據都使用統一的 CSV 格式：

```csv
aspect_id,sentence_id,sentence_text,aspect_term,aspect_polarity,from_position,to_position
1,3121,"But the staff was so horrible to us.",staff,negative,8,13
```

**欄位說明**:
- `aspect_id`: 全局唯一的 aspect ID
- `sentence_id`: 句子 ID（來自原始數據）
- `sentence_text`: 完整句子文本
- `aspect_term`: aspect 詞彙
- `aspect_polarity`: 情感極性 (positive/negative/neutral)
- `from_position`: aspect 在句子中的起始位置
- `to_position`: aspect 在句子中的結束位置

## ⚠️ 注意事項

1. **2014 vs 2016 格式差異**
   - 2014: 使用 `<aspectTerm>` 標籤
   - 2016: 使用 `<Opinion>` 標籤，包含 category 屬性

2. **測試集標籤**
   - 2014 測試集: 原始文件無 polarity，但已補充
   - 2016 測試集: 包含完整標籤

3. **NULL targets**
   - 2016 數據集中的 NULL targets 已被過濾

## 🔄 重新清理數據

如果需要重新清理數據：

```bash
# 清理訓練集（移除衝突）
python scripts/clean_aspect_data.py

# 處理測試集（格式轉換）
python scripts/process_test_data.py

# 驗證清理結果
python scripts/verify_cleaning.py
```

## 📚 相關文檔

- [配置系統使用說明](docs/CONFIG_USAGE.md)
- [命令速查表](docs/COMMANDS.md)
- [配置更新日誌](configs/CHANGELOG.md)
