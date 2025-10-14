"""
測試 ABSA 預處理器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import SemEvalDataLoader
from src.data_processing.preprocessor import ABSAPreprocessor
import numpy as np


def test_basic_preprocessing():
    """測試基本預處理功能"""
    print("=" * 70)
    print(" 測試 1: 基本預處理功能")
    print("=" * 70)
    print()

    # 建立預處理器
    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)

    # 測試資料
    test_data = [
        {
            'text': 'The food was great but the service was terrible.',
            'aspect': 'food',
            'polarity': 'positive',
            'aspect_position': [4, 8]
        },
        {
            'text': 'The food was great but the service was terrible.',
            'aspect': 'service',
            'polarity': 'negative',
            'aspect_position': [27, 34]
        },
        {
            'text': 'I love the battery life of this laptop.',
            'aspect': 'battery life',
            'polarity': 'positive',
            'aspect_position': [11, 23]
        }
    ]

    # 1. 測試文本清理
    print("[1.1] 文本清理測試")
    print("-" * 70)
    test_texts = [
        "This   is   a    test",
        "Don't worry about it",
        "It's REALLY great!",
    ]
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"原始: {text}")
        print(f"清理: {cleaned}")
        print()

    # 2. 測試分詞
    print("[1.2] 分詞測試")
    print("-" * 70)
    for item in test_data[:2]:
        tokens = preprocessor.tokenize(item['text'])
        print(f"文本: {item['text']}")
        print(f"Tokens: {tokens}")
        print()

    # 3. 建立詞彙表
    print("[1.3] 建立詞彙表")
    print("-" * 70)
    preprocessor.build_vocab(test_data)
    print()

    # 4. 測試數值化
    print("[1.4] 數值化測試")
    print("-" * 70)
    tokens = preprocessor.tokenize(test_data[0]['text'])
    indices = preprocessor.tokens_to_indices(tokens)
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    print()

    # 5. 測試面向詞遮罩
    print("[1.5] 面向詞遮罩測試")
    print("-" * 70)
    for item in test_data:
        tokens = preprocessor.tokenize(item['text'])
        mask = preprocessor.get_aspect_mask(
            tokens,
            item['aspect'],
            item['aspect_position'],
            item['text']
        )
        print(f"文本: {item['text']}")
        print(f"面向詞: {item['aspect']}")
        print(f"Tokens: {tokens}")
        print(f"Mask: {mask}")
        # 顯示遮罩效果
        masked_tokens = [t if m == 1 else '_' for t, m in zip(tokens, mask)]
        print(f"標記: {' '.join(masked_tokens)}")
        print()

    # 6. 完整預處理
    print("[1.6] 完整預處理測試")
    print("-" * 70)
    processed = preprocessor.preprocess_batch(test_data)
    for item in processed:
        print(f"文本: {item['text']}")
        print(f"面向詞: {item['aspect']}")
        print(f"Input IDs: {item['input_ids']}")
        print(f"Aspect Mask: {item['aspect_mask']}")
        print(f"Label: {item['label']}")
        print(f"Seq Length: {item['seq_length']}")
        print()

    return preprocessor, processed


def test_with_real_data():
    """測試真實資料集"""
    print("\n" + "=" * 70)
    print(" 測試 2: 真實資料集預處理")
    print("=" * 70)
    print()

    # 載入真實資料
    print("[2.1] 載入 SemEval2014_Laptop 訓練集")
    print("-" * 70)
    loader = SemEvalDataLoader()
    train_data = loader.load('SemEval2014_Laptop', 'train')
    print(f"載入 {len(train_data)} 個樣本\n")

    # 建立預處理器
    print("[2.2] 建立預處理器並建立詞彙表")
    print("-" * 70)
    preprocessor = ABSAPreprocessor(min_freq=3, max_vocab_size=10000, lower=True)
    preprocessor.build_vocab(train_data)
    print()

    # 預處理資料
    print("[2.3] 預處理資料")
    print("-" * 70)
    print("處理中...")
    processed_data = preprocessor.preprocess_batch(train_data)
    print(f"完成！處理了 {len(processed_data)} 個樣本\n")

    # 統計資訊
    print("[2.4] 統計資訊")
    print("-" * 70)
    preprocessor.print_statistics(processed_data)

    # 顯示範例
    print("\n[2.5] 預處理範例（前 3 筆）")
    print("-" * 70)
    for i, item in enumerate(processed_data[:3], 1):
        print(f"\n[範例 {i}]")
        print(f"原始文本: {item['text']}")
        print(f"面向詞: {item['aspect']}")
        print(f"Tokens: {' '.join(item['tokens'])}")
        print(f"Tokens 數量: {len(item['tokens'])}")
        print(f"Input IDs: {item['input_ids']}")
        print(f"Aspect Mask: {item['aspect_mask']}")
        # 顯示哪些 token 被標記為面向詞
        aspect_tokens = [t for t, m in zip(item['tokens'], item['aspect_mask']) if m == 1]
        print(f"標記的面向詞 tokens: {' '.join(aspect_tokens)}")
        print(f"Label: {item['label']} (polarity)")

    return preprocessor, processed_data


def test_padding():
    """測試 Padding 功能"""
    print("\n" + "=" * 70)
    print(" 測試 3: Padding 功能")
    print("=" * 70)
    print()

    # 載入資料
    loader = SemEvalDataLoader()
    train_data = loader.load('SemEval2014_Laptop', 'train')[:100]  # 只用前 100 筆

    # 建立並訓練預處理器
    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)
    preprocessor.build_vocab(train_data)
    processed_data = preprocessor.preprocess_batch(train_data)

    # 測試不同批次大小的 Padding
    print("[3.1] 測試批次 Padding")
    print("-" * 70)

    batch_sizes = [8, 16, 32]
    for batch_size in batch_sizes:
        batch = processed_data[:batch_size]
        padded = preprocessor.pad_sequences(batch)

        print(f"\nBatch Size: {batch_size}")
        print(f"  Input IDs shape: {padded['input_ids'].shape}")
        print(f"  Aspect Mask shape: {padded['aspect_mask'].shape}")
        print(f"  Labels shape: {padded['label'].shape}")
        print(f"  Seq Lengths: {padded['seq_length']}")
        print(f"  Max length in batch: {padded['seq_length'].max()}")

    # 測試固定長度 Padding
    print("\n[3.2] 測試固定長度 Padding")
    print("-" * 70)

    batch = processed_data[:10]
    for max_len in [30, 50, 80]:
        padded = preprocessor.pad_sequences(batch, max_length=max_len)
        print(f"\nMax Length: {max_len}")
        print(f"  Shape: {padded['input_ids'].shape}")
        print(f"  實際序列長度: {padded['seq_length']}")

    # 驗證 Padding 正確性
    print("\n[3.3] 驗證 Padding 正確性")
    print("-" * 70)

    batch = processed_data[:5]
    padded = preprocessor.pad_sequences(batch)

    pad_idx = preprocessor.word2idx[preprocessor.PAD_TOKEN]
    print(f"PAD token index: {pad_idx}")

    for i in range(len(batch)):
        seq_len = padded['seq_length'][i]
        input_ids = padded['input_ids'][i]

        # 檢查有效部分
        valid_part = input_ids[:seq_len]
        # 檢查 padding 部分
        padding_part = input_ids[seq_len:]

        has_pad_in_valid = (valid_part == pad_idx).any()
        all_pad_in_padding = (padding_part == pad_idx).all() if len(padding_part) > 0 else True

        status_valid = "[O]" if not has_pad_in_valid else "[X]"
        status_pad = "[O]" if all_pad_in_padding else "[X]"

        print(f"樣本 {i+1}:")
        print(f"  {status_valid} 有效部分無 PAD token")
        print(f"  {status_pad} Padding 部分全是 PAD token")


def test_label_encoding():
    """測試標籤編碼"""
    print("\n" + "=" * 70)
    print(" 測試 4: 標籤編碼")
    print("=" * 70)
    print()

    preprocessor = ABSAPreprocessor()

    print("[4.1] 情感極性映射")
    print("-" * 70)
    for polarity, label in preprocessor.POLARITY_MAP.items():
        encoded = preprocessor.encode_polarity(polarity)
        print(f"{polarity:12s} -> {encoded}")

    # 測試大小寫
    print("\n[4.2] 大小寫測試")
    print("-" * 70)
    test_polarities = ['POSITIVE', 'Negative', 'NeUtRaL']
    for pol in test_polarities:
        encoded = preprocessor.encode_polarity(pol)
        print(f"{pol:12s} -> {encoded}")


def test_save_load():
    """測試儲存和載入"""
    print("\n" + "=" * 70)
    print(" 測試 5: 儲存和載入預處理器")
    print("=" * 70)
    print()

    # 建立預處理器
    loader = SemEvalDataLoader()
    train_data = loader.load('SemEval2014_Laptop', 'train')[:500]

    preprocessor1 = ABSAPreprocessor(min_freq=3, lower=True)
    preprocessor1.build_vocab(train_data)

    # 儲存
    save_path = "data/processed/preprocessor.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    preprocessor1.save(save_path)

    # 載入
    preprocessor2 = ABSAPreprocessor()
    preprocessor2.load(save_path)

    # 驗證
    print("\n[5.1] 驗證載入的預處理器")
    print("-" * 70)
    checks = [
        ("詞彙表大小相同", preprocessor1.vocab_size == preprocessor2.vocab_size),
        ("word2idx 相同", preprocessor1.word2idx == preprocessor2.word2idx),
        ("min_freq 相同", preprocessor1.min_freq == preprocessor2.min_freq),
        ("is_fitted 相同", preprocessor1.is_fitted == preprocessor2.is_fitted),
    ]

    for check_name, passed in checks:
        status = "[O]" if passed else "[X]"
        print(f"{status} {check_name}")

    # 測試功能一致性
    print("\n[5.2] 測試預處理結果一致性")
    print("-" * 70)
    test_item = train_data[0]

    result1 = preprocessor1.preprocess_item(test_item)
    result2 = preprocessor2.preprocess_item(test_item)

    results_match = (
        result1['input_ids'] == result2['input_ids'] and
        result1['aspect_mask'] == result2['aspect_mask'] and
        result1['label'] == result2['label']
    )

    status = "[O]" if results_match else "[X]"
    print(f"{status} 預處理結果一致")


def main():
    """主測試函數"""
    print("\n" + "=" * 70)
    print(" ABSA 預處理器完整測試")
    print("=" * 70)
    print()

    try:
        # 執行所有測試
        test_basic_preprocessing()
        preprocessor, processed_data = test_with_real_data()
        test_padding()
        test_label_encoding()
        test_save_load()

        # 最終驗證
        print("\n" + "=" * 70)
        print(" 驗證標準檢查")
        print("=" * 70)
        print()

        stats = preprocessor.get_statistics(processed_data)

        checks = [
            ("詞彙表大小合理 (5000-15000)", 5000 <= stats['vocab_size'] <= 15000, f"{stats['vocab_size']:,}"),
            ("平均序列長度合理 (<100)", stats['avg_seq_length'] < 100, f"{stats['avg_seq_length']:.2f}"),
            ("標籤分佈存在", len(stats['label_distribution']) > 0, f"{len(stats['label_distribution'])} 類"),
            ("所有標籤都有樣本", all(count > 0 for count in stats['label_distribution'].values()), "是"),
        ]

        all_passed = True
        for check_name, passed, detail in checks:
            status = "[O]" if passed else "[X]"
            print(f"{status} {check_name}: {detail}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print("=" * 70)
            print(" 所有測試通過！[OK]")
            print("=" * 70)
        else:
            print("=" * 70)
            print(" 部分測試失敗 [WARNING]")
            print("=" * 70)

    except Exception as e:
        print(f"\n[錯誤] 測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
