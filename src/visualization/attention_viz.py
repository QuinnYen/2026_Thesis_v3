"""
Attention 權重可視化工具

用於可視化 Sentence-Level 模型的 Attention 權重
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# 添加項目根目錄到路徑
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class AttentionVisualizer:
    """Attention 權重可視化器"""

    def __init__(self, save_dir: str = 'outputs/visualizations/attention'):
        """
        初始化可視化器

        Args:
            save_dir: 儲存目錄
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 設定中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_attention_weights(self,
                               tokens: List[str],
                               attention_weights: np.ndarray,
                               title: str = 'Attention Weights',
                               save_name: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 3)):
        """
        繪製單個句子的 Attention 權重

        Args:
            tokens: Token 列表
            attention_weights: Attention 權重, shape: (seq_len,)
            title: 圖表標題
            save_name: 儲存檔名（不含副檔名）
            figsize: 圖表大小
        """
        # 過濾 padding token
        valid_len = min(len(tokens), len(attention_weights))
        tokens = tokens[:valid_len]
        attention_weights = attention_weights[:valid_len]

        # 移除 padding 部分（假設 padding 權重接近 0）
        threshold = 0.001
        mask = attention_weights > threshold
        if mask.any():
            tokens = [t for t, m in zip(tokens, mask) if m]
            attention_weights = attention_weights[mask]

        fig, ax = plt.subplots(figsize=figsize)

        # 繪製條形圖
        positions = np.arange(len(tokens))
        bars = ax.bar(positions, attention_weights, color='steelblue', alpha=0.7)

        # 高亮最高權重的 token
        max_idx = np.argmax(attention_weights)
        bars[max_idx].set_color('coral')

        # 設置標籤
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attention Weight')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # 儲存
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  儲存到: {save_path}")

        plt.close()

    def plot_attention_heatmap(self,
                              sentences: List[List[str]],
                              attention_weights: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = 'Attention Heatmap',
                              save_name: Optional[str] = None,
                              max_samples: int = 10):
        """
        繪製多個句子的 Attention 熱圖

        Args:
            sentences: 句子列表（token 列表的列表）
            attention_weights: Attention 權重, shape: (num_samples, seq_len)
            labels: 標籤列表
            title: 圖表標題
            save_name: 儲存檔名
            max_samples: 最多顯示樣本數
        """
        num_samples = min(len(sentences), max_samples, attention_weights.shape[0])

        # 找出最長的句子長度（用於對齊）
        max_len = max(len(sent) for sent in sentences[:num_samples])

        # 準備數據矩陣
        data = np.zeros((num_samples, max_len))
        yticks = []

        for i in range(num_samples):
            sent_len = min(len(sentences[i]), attention_weights.shape[1])
            data[i, :sent_len] = attention_weights[i, :sent_len]

            # Y軸標籤
            label_str = f"({labels[i]})" if labels else ""
            first_words = ' '.join(sentences[i][:3])
            yticks.append(f"{i+1}. {first_words}... {label_str}")

        # 繪製熱圖
        fig, ax = plt.subplots(figsize=(max_len * 0.5, num_samples * 0.5))

        sns.heatmap(data,
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Attention Weight'},
                    ax=ax,
                    vmin=0,
                    vmax=attention_weights.max())

        ax.set_yticks(np.arange(num_samples) + 0.5)
        ax.set_yticklabels(yticks, rotation=0, fontsize=8)
        ax.set_xlabel('Token Position')
        ax.set_title(title)

        plt.tight_layout()

        # 儲存
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  儲存到: {save_path}")

        plt.close()

    def plot_attention_comparison(self,
                                  tokens: List[str],
                                  pre_fusion_weights: np.ndarray,
                                  post_fusion_weights: np.ndarray,
                                  title: str = 'Attention Comparison',
                                  save_name: Optional[str] = None):
        """
        比較前融合和後融合的 Attention 權重

        Args:
            tokens: Token 列表
            pre_fusion_weights: 前融合 Attention 權重
            post_fusion_weights: 後融合 Attention 權重
            title: 圖表標題
            save_name: 儲存檔名
        """
        # 過濾 padding
        valid_len = min(len(tokens), len(pre_fusion_weights), len(post_fusion_weights))
        tokens = tokens[:valid_len]
        pre_fusion_weights = pre_fusion_weights[:valid_len]
        post_fusion_weights = post_fusion_weights[:valid_len]

        # 移除 padding
        threshold = 0.001
        mask = (pre_fusion_weights > threshold) | (post_fusion_weights > threshold)
        if mask.any():
            tokens = [t for t, m in zip(tokens, mask) if m]
            pre_fusion_weights = pre_fusion_weights[mask]
            post_fusion_weights = post_fusion_weights[mask]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        positions = np.arange(len(tokens))

        # 前融合
        axes[0].bar(positions, pre_fusion_weights, color='steelblue', alpha=0.7)
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(tokens, rotation=45, ha='right')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Pre-Fusion (Attention → BiLSTM)')
        axes[0].grid(axis='y', alpha=0.3)

        # 後融合
        axes[1].bar(positions, post_fusion_weights, color='coral', alpha=0.7)
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(tokens, rotation=45, ha='right')
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Post-Fusion (BiLSTM → Attention)')
        axes[1].grid(axis='y', alpha=0.3)

        fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 儲存
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  儲存到: {save_path}")

        plt.close()

    def plot_attention_distribution(self,
                                    attention_weights: np.ndarray,
                                    title: str = 'Attention Weight Distribution',
                                    save_name: Optional[str] = None):
        """
        繪製 Attention 權重分佈

        Args:
            attention_weights: Attention 權重, shape: (num_samples, seq_len)
            title: 圖表標題
            save_name: 儲存檔名
        """
        # 展平所有權重（排除 padding）
        flat_weights = attention_weights[attention_weights > 0.001].flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 直方圖
        axes[0].hist(flat_weights, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution')
        axes[0].grid(alpha=0.3)

        # 箱形圖
        axes[1].boxplot(flat_weights, vert=True)
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Box Plot')
        axes[1].grid(alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 儲存
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  儲存到: {save_path}")

        plt.close()

        # 列印統計資訊
        print(f"\nAttention 權重統計:")
        print(f"  平均值: {flat_weights.mean():.6f}")
        print(f"  中位數: {np.median(flat_weights):.6f}")
        print(f"  標準差: {flat_weights.std():.6f}")
        print(f"  最小值: {flat_weights.min():.6f}")
        print(f"  最大值: {flat_weights.max():.6f}")


def extract_attention_from_model(model,
                                 dataloader,
                                 device,
                                 model_type: str,
                                 num_samples: int = 10):
    """
    從模型中提取 Attention 權重

    Args:
        model: 模型
        dataloader: 資料載入器
        device: 設備
        model_type: 模型類型
        num_samples: 提取樣本數

    Returns:
        sentences: 句子列表
        attention_weights: Attention 權重
        labels: 標籤列表
    """
    model.eval()

    sentences = []
    attention_weights = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].squeeze().to(device)
            texts = batch['text']

            # 前向傳播
            _, attn = model(input_ids, attention_mask)

            # 收集結果
            for i in range(len(texts)):
                if len(sentences) >= num_samples:
                    break

                # 分詞
                tokens = texts[i].lower().split()[:50]  # 限制長度
                sentences.append(tokens)
                attention_weights.append(attn[i].cpu().numpy())
                labels.append(batch_labels[i].item())

            if len(sentences) >= num_samples:
                break

    attention_weights = np.array(attention_weights)

    return sentences, attention_weights, labels


if __name__ == '__main__':
    # 測試可視化
    print("測試 Attention 可視化工具")

    visualizer = AttentionVisualizer()

    # 測試單個句子
    tokens = ['the', 'food', 'was', 'amazing', 'but', 'service', 'was', 'terrible']
    attention_weights = np.array([0.05, 0.30, 0.05, 0.35, 0.05, 0.15, 0.02, 0.03])

    visualizer.plot_attention_weights(
        tokens=tokens,
        attention_weights=attention_weights,
        title='Example Attention Weights',
        save_name='test_attention'
    )

    print("\n[OK] Attention 可視化工具測試完成")
