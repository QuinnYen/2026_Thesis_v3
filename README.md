# 🎓 2026 論文專案 - Aspect-Based Sentiment Analysis

本專案為基於深度學習的面向基情感分析（Aspect-Based Sentiment Analysis, ABSA）研究。

---

## 📋 專案概覽

### 研究目標
- 實作並比較不同的多模態融合策略（前融合 vs 後融合）
- 在 SemEval 資料集上進行實驗和評估
- 分析不同融合方法對 ABSA 任務的影響

### 資料集
- **SemEval-2014 Task 4**: Laptop 和 Restaurant 領域
- **SemEval-2016 Task 5**: Laptop 和 Restaurant 領域

---

## 📁 專案結構

```
2026_Thesis_v3/
├── data/                      # 資料目錄
│   ├── raw/                   # 原始資料
│   │   ├── SemEval-2014/     # SemEval-2014 資料集
│   │   └── SemEval-2016/     # SemEval-2016 資料集
│   ├── processed/            # 處理後的資料
│   └── embeddings/           # 詞嵌入檔案
│
├── models/                    # 模型目錄
│   ├── baseline/             # Baseline 模型
│   ├── post_fusion/          # 後融合模型
│   └── pre_fusion/           # 前融合模型
│
├── outputs/                   # 輸出目錄
│   ├── checkpoints/          # 模型檢查點
│   ├── results/              # 實驗結果
│   └── visualizations/       # 視覺化圖表
│
├── src/                       # 原始碼
│   ├── data_processing/      # 資料處理模組
│   │   ├── xml_parser.py    # XML 解析器
│   │   └── dataset_statistics.py  # 統計分析
│   ├── models/               # 模型定義
│   ├── training/             # 訓練腳本
│   └── evaluation/           # 評估腳本
│
├── scripts/                   # 工具腳本
│   └── verify_datasets.py    # 資料集驗證腳本
│
├── notebooks/                 # Jupyter notebooks
├── configs/                   # 配置檔案
│   └── dataset_info.md       # 資料集詳細說明
│
├── requirements.txt           # Python 套件需求
├── README.md                  # 專案說明（本檔案）
└── DATA_PREPARATION_GUIDE.md # 資料準備指南
```

---

## 🚀 快速開始

### 1. 環境設定

```bash
# 克隆或進入專案目錄
cd d:\Quinn_SmallHouse\2026_Thesis_v3

# 安裝必要套件
pip install -r requirements.txt
```

### 2. 準備資料集

詳細步驟請參考 [DATA_PREPARATION_GUIDE.md](DATA_PREPARATION_GUIDE.md)

**簡要步驟**:
1. 下載 SemEval-2014 和 SemEval-2016 資料集
2. 將 XML 檔案放置到 `data/raw/SemEval-2014/` 和 `data/raw/SemEval-2016/`
3. 執行驗證腳本

```bash
python scripts/verify_datasets.py
```

### 3. 查看資料集統計

```bash
# 生成完整統計報告和視覺化
python src/data_processing/dataset_statistics.py
```

輸出檔案：
- `outputs/results/dataset_summary.csv` - 統計摘要
- `outputs/visualizations/sentiment_distribution.png` - 情感分佈圖
- `outputs/visualizations/dataset_comparison.png` - 資料集比較圖

---

## 📚 使用說明

### 資料處理

```python
from src.data_processing import SemEvalDatasetLoader

# 建立載入器
loader = SemEvalDatasetLoader(base_path="data/raw")

# 載入資料集
loader.load_semeval_2014(domain='both')  # 'laptop', 'restaurant', or 'both'
loader.load_semeval_2016(domain='both')

# 顯示統計資訊
loader.print_all_statistics()

# 獲取特定資料集
dataset = loader.get_dataset('semeval2014_laptop_train')

# 訪問資料
for sentence in dataset.sentences:
    print(f"Text: {sentence['text']}")
    for aspect in sentence['aspects']:
        print(f"  Aspect: {aspect['term']}, Polarity: {aspect['polarity']}")
```

### 統計分析

```python
from src.data_processing import DatasetStatistics

# 建立統計分析器
stats = DatasetStatistics(loader)

# 收集統計資訊
stats.collect_all_statistics()

# 生成完整報告
stats.generate_full_report()
```

---

## 🔧 開發指南

### 安裝的主要套件

- **深度學習**: PyTorch, TorchVision
- **資料處理**: NumPy, Pandas
- **機器學習**: scikit-learn
- **視覺化**: Matplotlib, Seaborn
- **詞嵌入**: Gensim
- **自然語言處理**: NLTK

### 程式碼規範

- 使用 Python 3.8+
- 遵循 PEP 8 編碼規範
- 為函數和類別撰寫 docstring
- 使用 type hints

---

## 📊 資料集資訊

### SemEval-2014 Task 4

| 資料集 | 訓練集 | 測試集 |
|--------|--------|--------|
| Laptop | Laptop_Train_v2.xml | Laptops_Test_Data_PhaseA/B.xml |
| Restaurant | Restaurants_Train_v2.xml | Restaurants_Test_Data_PhaseA/B.xml |

### SemEval-2016 Task 5

| 資料集 | 訓練集 | 測試集 |
|--------|--------|--------|
| Laptop | Laptops_Train_sb1.xml | laptops_test_sb1.xml |
| Restaurant | restaurants_train_sb1.xml | restaurants_test_sb1.xml |

詳細資料集格式說明請參考 [configs/dataset_info.md](configs/dataset_info.md)

---

## 📈 實驗計劃

### Phase 1: 資料準備 ✅
- [x] 建立專案結構
- [x] 準備資料集
- [x] 實作資料解析器
- [x] 統計分析工具

### Phase 2: 資料預處理（進行中）
- [ ] 文本清理和標準化
- [ ] 分詞和詞彙表建立
- [ ] 詞嵌入準備（Word2Vec, GloVe）
- [ ] 資料增強

### Phase 3: 模型實作
- [ ] Baseline 模型
- [ ] 前融合模型
- [ ] 後融合模型

### Phase 4: 訓練與評估
- [ ] 訓練流程
- [ ] 評估指標
- [ ] 結果分析

### Phase 5: 論文撰寫
- [ ] 實驗結果整理
- [ ] 視覺化分析
- [ ] 論文撰寫

---

## 🛠️ 工具腳本

### 資料集驗證
```bash
python scripts/verify_datasets.py
```

### 資料集統計
```bash
python src/data_processing/dataset_statistics.py
```

---

## 📖 參考文獻

### SemEval-2014 Task 4
```bibtex
@inproceedings{pontiki2014semeval,
  title={SemEval-2014 Task 4: Aspect Based Sentiment Analysis},
  author={Pontiki, Maria and Galanis, Dimitris and Pavlopoulos, John and Papageorgiou, Harris and Androutsopoulos, Ion and Manandhar, Suresh},
  booktitle={Proceedings of SemEval},
  year={2014}
}
```

### SemEval-2016 Task 5
```bibtex
@inproceedings{pontiki2016semeval,
  title={SemEval-2016 Task 5: Aspect Based Sentiment Analysis},
  author={Pontiki, Maria and Galanis, Dimitris and Papageorgiou, Haris and Androutsopoulos, Ion and Manandhar, Suresh and Mohammad, AL-Smadi and Mahmoud, Al-Ayyoub and others},
  booktitle={Proceedings of SemEval},
  year={2016}
}
```

---

## ⚠️ 常見問題

### Q: 找不到資料檔案
**A**: 請確認已按照 `DATA_PREPARATION_GUIDE.md` 下載並放置資料集檔案。

### Q: 模組導入錯誤
**A**: 確保在專案根目錄執行腳本，或設定 PYTHONPATH。

### Q: 視覺化中文顯示問題
**A**: 已在程式碼中設定中文字體，如仍有問題請安裝對應字體。

更多問題請參考 [DATA_PREPARATION_GUIDE.md](DATA_PREPARATION_GUIDE.md)

---

## 📞 聯絡資訊

如有任何問題或建議，歡迎聯繫。

---

## 📝 授權

本專案僅供學術研究使用。

---

**最後更新**: 2025-10-13  
**版本**: 1.0.0

