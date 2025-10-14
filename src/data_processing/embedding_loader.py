"""
GloVe 詞嵌入載入器
提供載入預訓練詞向量的功能，並處理 OOV（Out of Vocabulary）詞彙
"""

import numpy as np
import pickle
import os
from typing import Dict, Optional, Tuple
from pathlib import Path


class GloVeEmbedding:
    """GloVe 詞嵌入管理類別"""

    def __init__(self, embedding_dim: int = 300):
        """
        初始化 GloVe 詞嵌入載入器

        Args:
            embedding_dim: 詞向量維度，預設為 300
        """
        self.embedding_dim = embedding_dim
        self.word2vec: Dict[str, np.ndarray] = {}
        self.vocab_size = 0

    def load_glove(self, glove_path: str, cache_path: Optional[str] = None) -> None:
        """
        載入 GloVe 詞嵌入檔案

        Args:
            glove_path: GloVe 檔案路徑
            cache_path: 快取檔案路徑（pickle 格式），用於加速後續載入
        """
        # 檢查是否有快取檔案
        if cache_path and os.path.exists(cache_path):
            print(f"從快取載入詞嵌入: {cache_path}")
            self._load_from_cache(cache_path)
            return

        # 從原始 GloVe 檔案載入
        print(f"載入 GloVe 詞嵌入: {glove_path}")
        print("這可能需要幾分鐘時間...")

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num % 100000 == 0:
                        print(f"已載入 {line_num:,} 個詞彙...")

                    values = line.strip().split()
                    word = values[0]
                    vector = np.array(values[1:], dtype=np.float32)

                    # 驗證向量維度
                    if len(vector) != self.embedding_dim:
                        continue

                    self.word2vec[word] = vector

            self.vocab_size = len(self.word2vec)
            print(f"載入完成！共 {self.vocab_size:,} 個詞彙")

            # 儲存快取
            if cache_path:
                self._save_to_cache(cache_path)

        except Exception as e:
            raise RuntimeError(f"載入 GloVe 檔案時發生錯誤: {str(e)}")

    def _load_from_cache(self, cache_path: str) -> None:
        """從快取檔案載入詞嵌入"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            self.word2vec = cache_data['word2vec']
            self.embedding_dim = cache_data['embedding_dim']
            self.vocab_size = len(self.word2vec)
        print(f"快取載入完成！共 {self.vocab_size:,} 個詞彙")

    def _save_to_cache(self, cache_path: str) -> None:
        """儲存詞嵌入到快取檔案"""
        print(f"儲存快取到: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'word2vec': self.word2vec,
            'embedding_dim': self.embedding_dim
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print("快取儲存完成！")

    def get_vector(self, word: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        """
        取得單詞的詞向量

        Args:
            word: 要查詢的單詞
            default: 當單詞不存在時返回的預設向量

        Returns:
            該單詞的詞向量，如果不存在則返回 default
        """
        if word in self.word2vec:
            return self.word2vec[word]

        # 嘗試小寫
        word_lower = word.lower()
        if word_lower in self.word2vec:
            return self.word2vec[word_lower]

        # 返回預設值
        if default is not None:
            return default
        else:
            # 返回零向量
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def get_random_vector(self, seed: Optional[int] = None) -> np.ndarray:
        """
        生成隨機向量（用於 OOV 詞彙）

        Args:
            seed: 隨機種子

        Returns:
            隨機向量
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-0.25, 0.25, self.embedding_dim).astype(np.float32)

    def has_word(self, word: str) -> bool:
        """
        檢查單詞是否存在於詞表中

        Args:
            word: 要檢查的單詞

        Returns:
            True 如果單詞存在，否則 False
        """
        return word in self.word2vec or word.lower() in self.word2vec

    def get_embedding_matrix(self, vocab: list,
                            oov_strategy: str = 'zeros',
                            seed: Optional[int] = 42) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        為給定的詞彙表建立嵌入矩陣

        Args:
            vocab: 詞彙列表
            oov_strategy: OOV 詞彙處理策略
                - 'zeros': 使用零向量
                - 'random': 使用隨機向量
                - 'mean': 使用所有詞向量的平均值
            seed: 隨機種子（當使用 random 策略時）

        Returns:
            embedding_matrix: 嵌入矩陣 (vocab_size, embedding_dim)
            word2idx: 單詞到索引的映射
        """
        vocab_size = len(vocab)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        word2idx = {word: idx for idx, word in enumerate(vocab)}

        oov_count = 0

        # 計算平均向量（如果需要）
        mean_vector = None
        if oov_strategy == 'mean':
            all_vectors = np.array(list(self.word2vec.values()))
            mean_vector = np.mean(all_vectors, axis=0).astype(np.float32)

        # 填充嵌入矩陣
        for idx, word in enumerate(vocab):
            if self.has_word(word):
                embedding_matrix[idx] = self.get_vector(word)
            else:
                oov_count += 1
                if oov_strategy == 'random':
                    embedding_matrix[idx] = self.get_random_vector(seed=seed + idx if seed else None)
                elif oov_strategy == 'mean':
                    embedding_matrix[idx] = mean_vector
                # 預設為 zeros，已經初始化為零

        print(f"嵌入矩陣建立完成！")
        print(f"  詞彙表大小: {vocab_size:,}")
        print(f"  OOV 詞彙數: {oov_count:,} ({oov_count/vocab_size*100:.2f}%)")
        print(f"  OOV 策略: {oov_strategy}")

        return embedding_matrix, word2idx

    def get_info(self) -> Dict:
        """
        取得詞嵌入資訊

        Returns:
            包含詞嵌入資訊的字典
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'sample_words': list(self.word2vec.keys())[:10]
        }


def load_glove_embeddings(embedding_dim: int = 300,
                         data_dir: str = "data/embeddings",
                         use_cache: bool = True) -> GloVeEmbedding:
    """
    便捷函數：載入 GloVe 詞嵌入

    Args:
        embedding_dim: 詞向量維度
        data_dir: 資料目錄
        use_cache: 是否使用快取

    Returns:
        載入完成的 GloVeEmbedding 物件
    """
    glove_path = os.path.join(data_dir, f"glove.6B.{embedding_dim}d.txt")
    cache_path = os.path.join(data_dir, f"glove.6B.{embedding_dim}d.pkl") if use_cache else None

    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"找不到 GloVe 檔案: {glove_path}\n"
            f"請確認檔案是否存在於 {data_dir} 目錄中"
        )

    embedding = GloVeEmbedding(embedding_dim=embedding_dim)
    embedding.load_glove(glove_path, cache_path=cache_path)

    return embedding


if __name__ == "__main__":
    # 測試程式碼
    print("=== GloVe 詞嵌入載入器測試 ===\n")

    # 載入詞嵌入
    embedding = load_glove_embeddings(embedding_dim=300, data_dir="../../data/embeddings")

    # 顯示資訊
    info = embedding.get_info()
    print(f"\n詞嵌入資訊:")
    print(f"  詞彙量: {info['vocab_size']:,}")
    print(f"  向量維度: {info['embedding_dim']}")
    print(f"  範例詞彙: {', '.join(info['sample_words'])}")

    # 測試查詢單詞
    print("\n=== 測試詞向量查詢 ===")
    test_words = ["the", "computer", "neural", "network", "UNKNOWNWORD123"]

    for word in test_words:
        vector = embedding.get_vector(word)
        exists = embedding.has_word(word)
        print(f"\n單詞: '{word}'")
        print(f"  存在: {exists}")
        print(f"  向量形狀: {vector.shape}")
        print(f"  向量前5維: {vector[:5]}")

    # 測試建立嵌入矩陣
    print("\n=== 測試建立嵌入矩陣 ===")
    test_vocab = ["the", "cat", "sat", "on", "mat", "UNKNOWN1", "UNKNOWN2"]

    for strategy in ['zeros', 'random', 'mean']:
        print(f"\n使用策略: {strategy}")
        matrix, word2idx = embedding.get_embedding_matrix(test_vocab, oov_strategy=strategy)
        print(f"矩陣形狀: {matrix.shape}")
        print(f"詞彙索引: {word2idx}")
