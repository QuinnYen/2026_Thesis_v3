"""
測試 GloVe 詞嵌入載入功能
"""

import sys
import os

# 將專案根目錄加入 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.embedding_loader import load_glove_embeddings


def main():
    print("=" * 60)
    print("GloVe 詞嵌入載入測試")
    print("=" * 60)

    # 載入詞嵌入
    print("\n[步驟 1] 載入 GloVe 詞嵌入...")
    print("-" * 60)
    embedding = load_glove_embeddings(
        embedding_dim=300,
        data_dir="data/embeddings",
        use_cache=True
    )

    # 顯示基本資訊
    print("\n[步驟 2] 詞嵌入基本資訊")
    print("-" * 60)
    info = embedding.get_info()
    print(f"詞彙量: {info['vocab_size']:,} 個詞")
    print(f"向量維度: {info['embedding_dim']} 維")
    print(f"範例詞彙: {', '.join(info['sample_words'])}")

    # 測試常見單詞查詢
    print("\n[步驟 3] 測試常見單詞查詢")
    print("-" * 60)
    common_words = ["the", "is", "computer", "science", "neural", "network", "learning"]

    for word in common_words:
        vector = embedding.get_vector(word)
        exists = embedding.has_word(word)
        status = "[O]" if exists else "[X]"
        print(f"{status} '{word}': 向量形狀 {vector.shape}, 前3維 [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]")

    # 測試 OOV 詞彙
    print("\n[步驟 4] 測試 OOV（Out of Vocabulary）詞彙")
    print("-" * 60)
    oov_words = ["UNKNOWNWORD123", "XyZ9999", "嗨"]

    for word in oov_words:
        vector = embedding.get_vector(word)
        exists = embedding.has_word(word)
        status = "[O]" if exists else "[X]"
        is_zero = (vector == 0).all()
        print(f"{status} '{word}': 存在={exists}, 零向量={is_zero}")

    # 測試大小寫處理
    print("\n[步驟 5] 測試大小寫處理")
    print("-" * 60)
    test_pairs = [("Computer", "computer"), ("NETWORK", "network"), ("Neural", "neural")]

    for upper_word, lower_word in test_pairs:
        has_upper = embedding.has_word(upper_word)
        has_lower = embedding.has_word(lower_word)
        vec_upper = embedding.get_vector(upper_word)
        vec_lower = embedding.get_vector(lower_word)
        same = (vec_upper == vec_lower).all()
        print(f"'{upper_word}' vs '{lower_word}': 大寫存在={has_upper}, 小寫存在={has_lower}, 向量相同={same}")

    # 測試建立嵌入矩陣
    print("\n[步驟 6] 測試建立嵌入矩陣")
    print("-" * 60)
    test_vocab = ["the", "cat", "sat", "on", "the", "mat", "with", "a", "neural", "network",
                  "UNKNOWN1", "UNKNOWN2", "XYZ999"]

    print(f"測試詞彙表: {test_vocab}")
    print(f"詞彙表大小: {len(test_vocab)}")

    for strategy in ['zeros', 'random', 'mean']:
        print(f"\n  策略: {strategy}")
        matrix, word2idx = embedding.get_embedding_matrix(
            test_vocab,
            oov_strategy=strategy,
            seed=42
        )
        print(f"    矩陣形狀: {matrix.shape}")
        print(f"    矩陣範圍: [{matrix.min():.4f}, {matrix.max():.4f}]")

    # 測試隨機向量生成
    print("\n[步驟 7] 測試隨機向量生成（用於 OOV）")
    print("-" * 60)
    random_vec1 = embedding.get_random_vector(seed=42)
    random_vec2 = embedding.get_random_vector(seed=42)
    random_vec3 = embedding.get_random_vector(seed=99)

    print(f"隨機向量1（seed=42）: 形狀 {random_vec1.shape}, 前3維 [{random_vec1[0]:.4f}, {random_vec1[1]:.4f}, {random_vec1[2]:.4f}]")
    print(f"隨機向量2（seed=42）: 形狀 {random_vec2.shape}, 前3維 [{random_vec2[0]:.4f}, {random_vec2[1]:.4f}, {random_vec2[2]:.4f}]")
    print(f"隨機向量3（seed=99）: 形狀 {random_vec3.shape}, 前3維 [{random_vec3[0]:.4f}, {random_vec3[1]:.4f}, {random_vec3[2]:.4f}]")
    print(f"向量1與向量2相同: {(random_vec1 == random_vec2).all()}")
    print(f"向量1與向量3相同: {(random_vec1 == random_vec3).all()}")

    # 測試詞向量相似度（額外功能）
    print("\n[步驟 8] 測試詞向量相似度（餘弦相似度）")
    print("-" * 60)
    from numpy.linalg import norm

    def cosine_similarity(v1, v2):
        """計算餘弦相似度"""
        return (v1 @ v2) / (norm(v1) * norm(v2))

    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("computer", "laptop"),
        ("cat", "dog"),
        ("good", "bad")
    ]

    for word1, word2 in word_pairs:
        if embedding.has_word(word1) and embedding.has_word(word2):
            vec1 = embedding.get_vector(word1)
            vec2 = embedding.get_vector(word2)
            similarity = cosine_similarity(vec1, vec2)
            print(f"'{word1}' vs '{word2}': 相似度 = {similarity:.4f}")
        else:
            print(f"'{word1}' vs '{word2}': 詞彙不存在")

    print("\n" + "=" * 60)
    print("測試完成！[OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
