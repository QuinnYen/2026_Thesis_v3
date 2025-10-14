"""
詞嵌入矩陣建構器
將 GloVe 預訓練向量與詞彙表對應，建立 embedding matrix
"""

import numpy as np
import pickle
import os
from typing import Dict, Optional, Tuple, List
from .embedding_loader import load_glove_embeddings


class EmbeddingMatrixBuilder:
    """
    詞嵌入矩陣建構器

    功能：
    1. 載入 GloVe 預訓練向量
    2. 根據詞彙表建立 embedding matrix
    3. 處理 OOV（Out-of-Vocabulary）詞彙
    4. 計算詞彙覆蓋率
    5. 儲存和載入功能
    """

    def __init__(self, embedding_dim: int = 300):
        """
        初始化詞嵌入矩陣建構器

        Args:
            embedding_dim: 詞向量維度
        """
        self.embedding_dim = embedding_dim
        self.glove = None
        self.embedding_matrix = None
        self.word2idx = None
        self.vocab_size = 0

        # 統計資訊
        self.coverage_stats = {
            'total_words': 0,
            'found_words': 0,
            'oov_words': 0,
            'coverage_rate': 0.0
        }

    def load_glove(self, data_dir: str = "data/embeddings", use_cache: bool = True) -> None:
        """
        載入 GloVe 預訓練向量

        Args:
            data_dir: GloVe 檔案目錄
            use_cache: 是否使用快取
        """
        print(f"載入 GloVe {self.embedding_dim}d 詞向量...")
        self.glove = load_glove_embeddings(
            embedding_dim=self.embedding_dim,
            data_dir=data_dir,
            use_cache=use_cache
        )
        print("GloVe 載入完成！\n")

    def build_embedding_matrix(self,
                               word2idx: Dict[str, int],
                               oov_strategy: str = 'random',
                               random_seed: int = 42) -> np.ndarray:
        """
        根據詞彙表建立 embedding matrix

        Args:
            word2idx: 詞彙到索引的映射（來自 Preprocessor）
            oov_strategy: OOV 詞彙處理策略
                - 'zeros': 使用零向量
                - 'random': 使用隨機向量
                - 'mean': 使用 GloVe 向量的平均值
            random_seed: 隨機種子

        Returns:
            embedding_matrix: shape (vocab_size, embedding_dim)
        """
        if self.glove is None:
            raise RuntimeError("請先呼叫 load_glove() 載入 GloVe 向量")

        self.word2idx = word2idx
        self.vocab_size = len(word2idx)

        print(f"建立詞嵌入矩陣...")
        print(f"  詞彙表大小: {self.vocab_size:,}")
        print(f"  詞向量維度: {self.embedding_dim}")
        print(f"  OOV 策略: {oov_strategy}")
        print()

        # 初始化矩陣
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float32)

        # 設定隨機種子
        if random_seed is not None:
            np.random.seed(random_seed)

        # 計算平均向量（如果需要）
        mean_vector = None
        if oov_strategy == 'mean':
            print("計算 GloVe 向量平均值...")
            all_vectors = np.array(list(self.glove.word2vec.values()))
            mean_vector = np.mean(all_vectors, axis=0).astype(np.float32)

        # 統計
        found_words = []
        oov_words = []

        # 填充矩陣
        for word, idx in word2idx.items():
            # 特殊處理：<PAD> 使用零向量
            if word == '<PAD>':
                self.embedding_matrix[idx] = np.zeros(self.embedding_dim, dtype=np.float32)
                continue

            # 檢查詞是否在 GloVe 中
            if self.glove.has_word(word):
                self.embedding_matrix[idx] = self.glove.get_vector(word)
                found_words.append(word)
            else:
                # OOV 詞彙處理
                oov_words.append(word)

                if oov_strategy == 'zeros':
                    self.embedding_matrix[idx] = np.zeros(self.embedding_dim, dtype=np.float32)
                elif oov_strategy == 'random':
                    # 隨機初始化（Xavier initialization）
                    self.embedding_matrix[idx] = np.random.uniform(
                        -0.25, 0.25, self.embedding_dim
                    ).astype(np.float32)
                elif oov_strategy == 'mean':
                    self.embedding_matrix[idx] = mean_vector
                else:
                    raise ValueError(f"未知的 OOV 策略: {oov_strategy}")

        # 更新統計資訊
        self.coverage_stats = {
            'total_words': self.vocab_size,
            'found_words': len(found_words),
            'oov_words': len(oov_words),
            'coverage_rate': len(found_words) / self.vocab_size if self.vocab_size > 0 else 0.0,
            'found_word_list': found_words[:100],  # 保留前 100 個
            'oov_word_list': oov_words[:100]       # 保留前 100 個
        }

        print("詞嵌入矩陣建立完成！")
        self._print_coverage_report()

        return self.embedding_matrix

    def _print_coverage_report(self) -> None:
        """列印詞彙覆蓋率報告"""
        stats = self.coverage_stats

        print("\n" + "=" * 70)
        print(" 詞彙覆蓋率報告")
        print("=" * 70)
        print(f"詞彙表大小: {stats['total_words']:,}")
        print(f"GloVe 覆蓋詞彙數: {stats['found_words']:,}")
        print(f"OOV 詞彙數: {stats['oov_words']:,}")
        print(f"覆蓋率: {stats['coverage_rate']*100:.2f}%")

        # 顯示覆蓋狀態
        if stats['coverage_rate'] >= 0.90:
            status = "優秀"
        elif stats['coverage_rate'] >= 0.85:
            status = "良好"
        elif stats['coverage_rate'] >= 0.75:
            status = "可接受"
        else:
            status = "需改進"

        print(f"狀態: {status}")

        # 顯示一些 OOV 詞彙範例
        if stats['oov_words'] > 0 and 'oov_word_list' in stats:
            print(f"\nOOV 詞彙範例（前 20 個）:")
            for word in stats['oov_word_list'][:20]:
                print(f"  - {word}")

        print("=" * 70)

    def get_coverage_stats(self) -> Dict:
        """
        取得詞彙覆蓋率統計資訊

        Returns:
            統計資訊字典
        """
        return self.coverage_stats

    def visualize_embeddings(self, words: Optional[List[str]] = None, n_dims: int = 10) -> None:
        """
        視覺化詞嵌入向量

        Args:
            words: 要顯示的詞彙列表（None 則隨機選擇 5 個）
            n_dims: 顯示向量的前 n 維
        """
        if self.embedding_matrix is None or self.word2idx is None:
            raise RuntimeError("請先建立 embedding matrix")

        print("\n" + "=" * 70)
        print(f" 詞嵌入向量視覺化（顯示前 {n_dims} 維）")
        print("=" * 70)

        # 如果沒有指定詞彙，隨機選擇
        if words is None:
            # 排除特殊 token
            normal_words = [w for w in self.word2idx.keys() if not w.startswith('<')]
            if len(normal_words) > 5:
                np.random.seed(42)
                words = list(np.random.choice(normal_words, size=5, replace=False))
            else:
                words = normal_words[:5]

        for word in words:
            if word not in self.word2idx:
                print(f"\n詞彙 '{word}' 不在詞彙表中")
                continue

            idx = self.word2idx[word]
            vector = self.embedding_matrix[idx]

            # 檢查是否在 GloVe 中
            in_glove = self.glove.has_word(word) if self.glove else False
            status = "[GloVe]" if in_glove else "[OOV]"

            print(f"\n詞彙: '{word}' {status}")
            print(f"  索引: {idx}")
            print(f"  向量前 {n_dims} 維:")
            vector_str = ", ".join([f"{v:.4f}" for v in vector[:n_dims]])
            print(f"    [{vector_str}]")
            print(f"  向量範數: {np.linalg.norm(vector):.4f}")

        print("=" * 70)

    def get_embedding_matrix(self) -> np.ndarray:
        """
        取得 embedding matrix

        Returns:
            embedding_matrix: shape (vocab_size, embedding_dim)
        """
        if self.embedding_matrix is None:
            raise RuntimeError("請先建立 embedding matrix")
        return self.embedding_matrix

    def save(self, filepath: str) -> None:
        """
        儲存 embedding matrix 和相關資訊

        Args:
            filepath: 儲存路徑
        """
        if self.embedding_matrix is None:
            raise RuntimeError("請先建立 embedding matrix")

        data = {
            'embedding_matrix': self.embedding_matrix,
            'word2idx': self.word2idx,
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'coverage_stats': self.coverage_stats
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nEmbedding matrix 已儲存到: {filepath}")
        print(f"  矩陣形狀: {self.embedding_matrix.shape}")
        print(f"  檔案大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

    def load(self, filepath: str) -> np.ndarray:
        """
        載入 embedding matrix

        Args:
            filepath: 載入路徑

        Returns:
            embedding_matrix
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.embedding_matrix = data['embedding_matrix']
        self.word2idx = data['word2idx']
        self.embedding_dim = data['embedding_dim']
        self.vocab_size = data['vocab_size']
        self.coverage_stats = data['coverage_stats']

        print(f"\nEmbedding matrix 已從 {filepath} 載入")
        print(f"  矩陣形狀: {self.embedding_matrix.shape}")
        print(f"  覆蓋率: {self.coverage_stats['coverage_rate']*100:.2f}%")

        return self.embedding_matrix

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        取得指定詞的向量

        Args:
            word: 詞彙

        Returns:
            向量或 None
        """
        if self.embedding_matrix is None or self.word2idx is None:
            raise RuntimeError("請先建立 embedding matrix")

        if word not in self.word2idx:
            return None

        idx = self.word2idx[word]
        return self.embedding_matrix[idx]

    def compute_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        計算兩個詞的餘弦相似度

        Args:
            word1: 第一個詞
            word2: 第二個詞

        Returns:
            相似度（0-1）或 None
        """
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)

        if vec1 is None or vec2 is None:
            return None

        # 餘弦相似度
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)


def build_embedding_matrix_from_preprocessor(preprocessor,
                                            embedding_dim: int = 300,
                                            oov_strategy: str = 'random',
                                            data_dir: str = "data/embeddings") -> Tuple[np.ndarray, EmbeddingMatrixBuilder]:
    """
    便捷函數：從 Preprocessor 建立 embedding matrix

    Args:
        preprocessor: ABSAPreprocessor 實例
        embedding_dim: 詞向量維度
        oov_strategy: OOV 策略
        data_dir: GloVe 資料目錄

    Returns:
        (embedding_matrix, builder)
    """
    # 建立 builder
    builder = EmbeddingMatrixBuilder(embedding_dim=embedding_dim)

    # 載入 GloVe
    builder.load_glove(data_dir=data_dir, use_cache=True)

    # 建立矩陣
    embedding_matrix = builder.build_embedding_matrix(
        word2idx=preprocessor.word2idx,
        oov_strategy=oov_strategy
    )

    return embedding_matrix, builder


if __name__ == "__main__":
    # 簡單測試
    print("EmbeddingMatrixBuilder 測試")
    print("=" * 70)

    # 建立測試詞彙表
    test_word2idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'food': 3,
        'is': 4,
        'great': 5,
        'UNKNOWNWORD123': 6  # OOV 詞
    }

    # 建立 builder
    builder = EmbeddingMatrixBuilder(embedding_dim=300)

    # 載入 GloVe
    builder.load_glove(data_dir="../../data/embeddings")

    # 建立矩陣
    embedding_matrix = builder.build_embedding_matrix(
        word2idx=test_word2idx,
        oov_strategy='random'
    )

    print(f"\nEmbedding matrix 形狀: {embedding_matrix.shape}")

    # 視覺化
    builder.visualize_embeddings(words=['the', 'food', 'UNKNOWNWORD123'])

    # 計算相似度
    print("\n相似度測試:")
    sim = builder.compute_similarity('the', 'food')
    print(f"  'the' vs 'food': {sim:.4f}")
