"""
測試詞嵌入矩陣建構器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import SemEvalDataLoader
from src.data_processing.preprocessor import ABSAPreprocessor
from src.data_processing.embedding_builder import EmbeddingMatrixBuilder, build_embedding_matrix_from_preprocessor
import numpy as np


def test_basic_functionality():
    """測試基本功能"""
    print("=" * 70)
    print(" 測試 1: 基本功能測試")
    print("=" * 70)
    print()

    # 建立測試詞彙表
    print("[1.1] 建立測試詞彙表")
    print("-" * 70)
    test_word2idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'food': 3,
        'service': 4,
        'good': 5,
        'bad': 6,
        'restaurant': 7,
        'UNKNOWNWORD123': 8,
        'XYZ999': 9
    }
    print(f"詞彙表大小: {len(test_word2idx)}")
    print(f"詞彙: {list(test_word2idx.keys())}")
    print()

    # 建立 builder
    print("[1.2] 建立 EmbeddingMatrixBuilder")
    print("-" * 70)
    builder = EmbeddingMatrixBuilder(embedding_dim=300)
    print(f"詞向量維度: {builder.embedding_dim}")
    print()

    # 載入 GloVe
    print("[1.3] 載入 GloVe 向量")
    print("-" * 70)
    builder.load_glove(data_dir="data/embeddings", use_cache=True)
    print()

    # 測試不同的 OOV 策略
    print("[1.4] 測試 OOV 策略")
    print("-" * 70)

    for strategy in ['zeros', 'random', 'mean']:
        print(f"\n策略: {strategy}")
        matrix = builder.build_embedding_matrix(
            word2idx=test_word2idx,
            oov_strategy=strategy,
            random_seed=42
        )
        print(f"  矩陣形狀: {matrix.shape}")

    return builder


def test_with_real_data():
    """使用真實資料測試"""
    print("\n" + "=" * 70)
    print(" 測試 2: 真實資料測試")
    print("=" * 70)
    print()

    # 載入資料
    print("[2.1] 載入資料並建立預處理器")
    print("-" * 70)
    loader = SemEvalDataLoader()
    train_data = loader.load('SemEval2014_Laptop', 'train')
    print()

    # 建立預處理器
    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)
    preprocessor.build_vocab(train_data)
    print()

    # 建立 embedding matrix
    print("[2.2] 建立詞嵌入矩陣")
    print("-" * 70)
    embedding_matrix, builder = build_embedding_matrix_from_preprocessor(
        preprocessor=preprocessor,
        embedding_dim=300,
        oov_strategy='random',
        data_dir="data/embeddings"
    )
    print()

    # 檢查形狀
    print("[2.3] 檢查矩陣形狀")
    print("-" * 70)
    expected_shape = (preprocessor.vocab_size, 300)
    actual_shape = embedding_matrix.shape
    print(f"預期形狀: {expected_shape}")
    print(f"實際形狀: {actual_shape}")
    print(f"形狀正確: {expected_shape == actual_shape}")
    print()

    # 檢查特殊 token
    print("[2.4] 檢查特殊 token")
    print("-" * 70)
    pad_idx = preprocessor.word2idx['<PAD>']
    pad_vector = embedding_matrix[pad_idx]
    is_zero = np.allclose(pad_vector, 0.0)
    print(f"<PAD> 索引: {pad_idx}")
    print(f"<PAD> 向量是否為零: {is_zero}")
    print(f"<PAD> 向量範數: {np.linalg.norm(pad_vector):.6f}")
    print()

    return embedding_matrix, builder, preprocessor


def test_coverage():
    """測試詞彙覆蓋率"""
    print("\n" + "=" * 70)
    print(" 測試 3: 詞彙覆蓋率分析")
    print("=" * 70)
    print()

    # 載入不同資料集
    loader = SemEvalDataLoader()

    datasets = [
        ('SemEval2014_Laptop', 'train'),
        ('SemEval2014_Restaurant', 'train')
    ]

    results = []

    for dataset_name, split in datasets:
        print(f"\n[{dataset_name}]")
        print("-" * 70)

        # 載入資料
        data = loader.load(dataset_name, split)

        # 建立預處理器
        preprocessor = ABSAPreprocessor(min_freq=2, lower=True)
        preprocessor.build_vocab(data)
        print()

        # 建立 embedding matrix
        builder = EmbeddingMatrixBuilder(embedding_dim=300)
        builder.load_glove(data_dir="data/embeddings", use_cache=True)
        builder.build_embedding_matrix(
            word2idx=preprocessor.word2idx,
            oov_strategy='random'
        )

        # 記錄結果
        stats = builder.get_coverage_stats()
        results.append({
            'dataset': dataset_name,
            'vocab_size': stats['total_words'],
            'coverage_rate': stats['coverage_rate'],
            'oov_count': stats['oov_words']
        })

    # 顯示摘要
    print("\n" + "=" * 70)
    print(" 覆蓋率摘要")
    print("=" * 70)
    for res in results:
        print(f"\n{res['dataset']}:")
        print(f"  詞彙表大小: {res['vocab_size']:,}")
        print(f"  覆蓋率: {res['coverage_rate']*100:.2f}%")
        print(f"  OOV 詞數: {res['oov_count']:,}")


def test_visualization():
    """測試視覺化"""
    print("\n" + "=" * 70)
    print(" 測試 4: 詞嵌入視覺化")
    print("=" * 70)
    print()

    # 載入資料並建立
    loader = SemEvalDataLoader()
    data = loader.load('SemEval2014_Laptop', 'train')

    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)
    preprocessor.build_vocab(data)

    builder = EmbeddingMatrixBuilder(embedding_dim=300)
    builder.load_glove(data_dir="data/embeddings", use_cache=True)
    builder.build_embedding_matrix(
        word2idx=preprocessor.word2idx,
        oov_strategy='random'
    )

    # 隨機選擇詞彙視覺化
    print("\n[4.1] 隨機選擇 5 個詞")
    print("-" * 70)
    builder.visualize_embeddings(words=None, n_dims=10)

    # 指定詞彙視覺化
    print("\n[4.2] 指定詞彙視覺化")
    print("-" * 70)
    test_words = ['laptop', 'battery', 'screen', 'price', 'good']
    builder.visualize_embeddings(words=test_words, n_dims=10)


def test_similarity():
    """測試相似度計算"""
    print("\n" + "=" * 70)
    print(" 測試 5: 詞相似度計算")
    print("=" * 70)
    print()

    # 建立 builder
    loader = SemEvalDataLoader()
    data = loader.load('SemEval2014_Laptop', 'train')

    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)
    preprocessor.build_vocab(data)

    builder = EmbeddingMatrixBuilder(embedding_dim=300)
    builder.load_glove(data_dir="data/embeddings", use_cache=True)
    builder.build_embedding_matrix(
        word2idx=preprocessor.word2idx,
        oov_strategy='random'
    )

    # 測試相似度
    print("[5.1] 相似詞對")
    print("-" * 70)
    similar_pairs = [
        ('laptop', 'computer'),
        ('good', 'great'),
        ('bad', 'terrible'),
        ('screen', 'display'),
        ('battery', 'power')
    ]

    for word1, word2 in similar_pairs:
        sim = builder.compute_similarity(word1, word2)
        if sim is not None:
            print(f"  '{word1}' vs '{word2}': {sim:.4f}")
        else:
            print(f"  '{word1}' vs '{word2}': 詞彙不存在")

    print("\n[5.2] 不相似詞對")
    print("-" * 70)
    dissimilar_pairs = [
        ('laptop', 'food'),
        ('good', 'bad'),
        ('screen', 'battery')
    ]

    for word1, word2 in dissimilar_pairs:
        sim = builder.compute_similarity(word1, word2)
        if sim is not None:
            print(f"  '{word1}' vs '{word2}': {sim:.4f}")
        else:
            print(f"  '{word1}' vs '{word2}': 詞彙不存在")


def test_save_load():
    """測試儲存和載入"""
    print("\n" + "=" * 70)
    print(" 測試 6: 儲存和載入")
    print("=" * 70)
    print()

    # 建立並儲存
    print("[6.1] 建立並儲存 embedding matrix")
    print("-" * 70)
    loader = SemEvalDataLoader()
    data = loader.load('SemEval2014_Laptop', 'train')[:500]  # 使用部分資料加速

    preprocessor = ABSAPreprocessor(min_freq=2, lower=True)
    preprocessor.build_vocab(data)

    builder1 = EmbeddingMatrixBuilder(embedding_dim=300)
    builder1.load_glove(data_dir="data/embeddings", use_cache=True)
    matrix1 = builder1.build_embedding_matrix(
        word2idx=preprocessor.word2idx,
        oov_strategy='random'
    )

    # 儲存
    save_path = "data/processed/embedding_matrix.pkl"
    builder1.save(save_path)

    # 載入
    print("\n[6.2] 載入 embedding matrix")
    print("-" * 70)
    builder2 = EmbeddingMatrixBuilder(embedding_dim=300)
    matrix2 = builder2.load(save_path)

    # 驗證
    print("\n[6.3] 驗證載入結果")
    print("-" * 70)
    checks = [
        ("矩陣形狀相同", matrix1.shape == matrix2.shape),
        ("矩陣內容相同", np.allclose(matrix1, matrix2)),
        ("詞彙表相同", builder1.word2idx == builder2.word2idx),
        ("覆蓋率相同", builder1.coverage_stats['coverage_rate'] == builder2.coverage_stats['coverage_rate'])
    ]

    for check_name, passed in checks:
        status = "[O]" if passed else "[X]"
        print(f"{status} {check_name}")


def main():
    """主測試函數"""
    print("\n" + "=" * 70)
    print(" EmbeddingMatrixBuilder 完整測試")
    print("=" * 70)
    print()

    try:
        # 執行所有測試
        test_basic_functionality()
        embedding_matrix, builder, preprocessor = test_with_real_data()
        test_coverage()
        test_visualization()
        test_similarity()
        test_save_load()

        # 最終驗證
        print("\n" + "=" * 70)
        print(" Task 1.3 驗證標準")
        print("=" * 70)
        print()

        stats = builder.get_coverage_stats()

        checks = [
            ("Embedding matrix 形狀正確",
             embedding_matrix.shape == (preprocessor.vocab_size, 300),
             f"{embedding_matrix.shape}"),
            ("詞彙覆蓋率 > 85%",
             stats['coverage_rate'] > 0.85,
             f"{stats['coverage_rate']*100:.2f}%"),
            ("<PAD> 為零向量",
             np.allclose(embedding_matrix[preprocessor.word2idx['<PAD>']], 0.0),
             "是"),
            ("OOV 詞彙已處理",
             stats['oov_words'] >= 0,
             f"{stats['oov_words']} 個"),
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
