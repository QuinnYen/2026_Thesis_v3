"""
Attention 視覺化腳本

展示 Baseline 和 Aspect-Level 模型的 attention 差異
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加入 src 路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "models"))

from baseline import BaselineModel
from aspect_level import AspectLevelModel

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = BASE_DIR / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_attention_comparison():
    """比較 Baseline 和 Aspect-Level 模型的 attention"""

    print("=" * 80)
    print("Attention 視覺化比較")
    print("=" * 80)

    # 設定參數
    vocab_size = 10000
    embedding_dim = 300
    hidden_size = 128
    num_classes = 3
    seq_len = 40

    # 建立模型
    print("\n1. 載入模型...")
    baseline_model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=0.0  # 評估模式不使用 dropout
    )

    aspect_model = AspectLevelModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        position_dim=50,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=0.0
    )

    baseline_model.eval()
    aspect_model.eval()

    # 2. 測試不同 aspect 位置的 attention
    print("\n2. 測試不同 aspect 位置...")

    test_cases = [
        {"name": "前段 Aspect", "aspect_pos": (5, 10)},
        {"name": "中段 Aspect", "aspect_pos": (18, 23)},
        {"name": "後段 Aspect", "aspect_pos": (30, 35)},
    ]

    fig, axes = plt.subplots(len(test_cases), 2, figsize=(16, 4 * len(test_cases)))

    for idx, test_case in enumerate(test_cases):
        name = test_case["name"]
        start, end = test_case["aspect_pos"]

        print(f"\n測試案例: {name} (位置 {start}-{end})")

        # 建立測試資料
        dummy_input = torch.randint(0, vocab_size, (1, seq_len))
        dummy_mask = torch.zeros(1, seq_len)
        dummy_mask[0, start:end] = 1

        # Baseline 模型預測
        with torch.no_grad():
            _, baseline_attn = baseline_model(dummy_input, dummy_mask)

        # Aspect-Level 模型預測
        with torch.no_grad():
            _, aspect_attn = aspect_model(dummy_input, dummy_mask)

        # 轉換為 numpy
        baseline_attn = baseline_attn[0].cpu().numpy()
        aspect_attn = aspect_attn[0].cpu().numpy()

        # 視覺化 Baseline
        ax1 = axes[idx, 0] if len(test_cases) > 1 else axes[0]
        positions = np.arange(seq_len)
        ax1.bar(positions, baseline_attn, alpha=0.7, color='steelblue')
        ax1.axvspan(start, end, alpha=0.3, color='red', label='Aspect 位置')
        ax1.set_xlabel('詞位置', fontsize=12)
        ax1.set_ylabel('Attention 權重', fontsize=12)
        ax1.set_title(f'Baseline 模型 - {name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 統計資訊
        aspect_weight_baseline = baseline_attn[start:end].sum()
        ax1.text(0.02, 0.95, f'Aspect 區域權重: {aspect_weight_baseline:.3f}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 視覺化 Aspect-Level
        ax2 = axes[idx, 1] if len(test_cases) > 1 else axes[1]
        ax2.bar(positions, aspect_attn, alpha=0.7, color='coral')
        ax2.axvspan(start, end, alpha=0.3, color='red', label='Aspect 位置')
        ax2.set_xlabel('詞位置', fontsize=12)
        ax2.set_ylabel('Attention 權重', fontsize=12)
        ax2.set_title(f'Aspect-Level 模型 - {name}', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 統計資訊
        aspect_weight_aspect = aspect_attn[start:end].sum()
        nearby_weight_aspect = aspect_attn[max(0, start-3):min(seq_len, end+3)].sum()
        ax2.text(0.02, 0.95,
                f'Aspect 區域權重: {aspect_weight_aspect:.3f}\n'
                f'Aspect±3 區域權重: {nearby_weight_aspect:.3f}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        print(f"  Baseline - Aspect 區域權重: {aspect_weight_baseline:.4f}")
        print(f"  Aspect-Level - Aspect 區域權重: {aspect_weight_aspect:.4f}")
        print(f"  Aspect-Level - Aspect±3 區域權重: {nearby_weight_aspect:.4f}")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "attention_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n圖表已儲存: {output_path}")
    plt.close()


def visualize_position_effect():
    """視覺化位置嵌入的效果"""

    print("\n" + "=" * 80)
    print("位置嵌入效果視覺化")
    print("=" * 80)

    vocab_size = 10000
    embedding_dim = 300
    hidden_size = 128
    seq_len = 30

    # 建立模型
    model = AspectLevelModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        position_dim=50,
        hidden_size=hidden_size,
        num_classes=3,
        dropout=0.0
    )
    model.eval()

    # 測試相同句子，不同 aspect 位置
    test_input = torch.randint(0, vocab_size, (1, seq_len))

    aspect_positions = [5, 10, 15, 20, 25]
    attentions = []

    for pos in aspect_positions:
        mask = torch.zeros(1, seq_len)
        mask[0, pos:pos+3] = 1

        with torch.no_grad():
            _, attn = model(test_input, mask)

        attentions.append(attn[0].cpu().numpy())

    # 繪製熱圖
    attentions = np.array(attentions)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        attentions,
        cmap='YlOrRd',
        xticklabels=range(seq_len),
        yticklabels=[f'Aspect@{p}' for p in aspect_positions],
        cbar_kws={'label': 'Attention 權重'},
        linewidths=0.5,
        linecolor='gray'
    )

    # 標記 aspect 位置
    for i, pos in enumerate(aspect_positions):
        plt.gca().add_patch(plt.Rectangle(
            (pos, i), 3, 1,
            fill=False, edgecolor='blue', lw=3
        ))

    plt.xlabel('詞位置', fontsize=12, fontweight='bold')
    plt.ylabel('Aspect 位置', fontsize=12, fontweight='bold')
    plt.title('不同 Aspect 位置的 Attention 分布\n（藍框標記 Aspect 位置）',
             fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "position_effect_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n熱圖已儲存: {output_path}")
    plt.close()


def visualize_distance_decay():
    """視覺化距離衰減效果"""

    print("\n" + "=" * 80)
    print("距離衰減效果視覺化")
    print("=" * 80)

    vocab_size = 10000
    embedding_dim = 300
    hidden_size = 128
    seq_len = 50

    model = AspectLevelModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        position_dim=50,
        hidden_size=hidden_size,
        num_classes=3,
        dropout=0.0
    )
    model.eval()

    # Aspect 在中間位置
    test_input = torch.randint(0, vocab_size, (1, seq_len))
    aspect_center = 25
    mask = torch.zeros(1, seq_len)
    mask[0, aspect_center-2:aspect_center+3] = 1  # aspect 寬度為 5

    with torch.no_grad():
        _, attn = model(test_input, mask)

    attn = attn[0].cpu().numpy()

    # 計算距離
    distances = np.abs(np.arange(seq_len) - aspect_center)

    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上圖：Attention 權重 vs 位置
    ax1.bar(range(seq_len), attn, alpha=0.7, color='steelblue')
    ax1.axvspan(aspect_center-2, aspect_center+3, alpha=0.3, color='red',
               label='Aspect 位置')
    ax1.set_xlabel('詞位置', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention 權重', fontsize=12, fontweight='bold')
    ax1.set_title('Attention 權重分布', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 下圖：Attention 權重 vs 距離 Aspect 的距離
    # 按距離分組計算平均權重
    max_dist = int(distances.max())
    dist_weights = []
    for d in range(max_dist + 1):
        mask_dist = (distances == d)
        if mask_dist.sum() > 0:
            dist_weights.append(attn[mask_dist].mean())
        else:
            dist_weights.append(0)

    ax2.plot(range(len(dist_weights)), dist_weights, marker='o',
            linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('距離 Aspect 的距離（詞數）', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均 Attention 權重', fontsize=12, fontweight='bold')
    ax2.set_title('距離衰減效果（距離越遠，權重越低）', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    # 添加擬合曲線（指數衰減）
    from scipy.optimize import curve_fit

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    try:
        x_data = np.array(range(len(dist_weights)))
        y_data = np.array(dist_weights)
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[0.1, 0.1])
        x_fit = np.linspace(0, max_dist, 100)
        y_fit = exp_decay(x_fit, *popt)
        ax2.plot(x_fit, y_fit, '--', color='red', linewidth=2,
                label=f'擬合曲線: {popt[0]:.3f} × exp(-{popt[1]:.3f} × d)')
        ax2.legend()
    except Exception as e:
        print(f"  擬合曲線失敗: {e}")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "distance_decay.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n距離衰減圖已儲存: {output_path}")
    plt.close()


def main():
    print("開始生成 Attention 視覺化...")

    # 1. Baseline vs Aspect-Level 比較
    visualize_attention_comparison()

    # 2. 位置嵌入效果
    visualize_position_effect()

    # 3. 距離衰減效果
    visualize_distance_decay()

    print("\n" + "=" * 80)
    print("所有視覺化完成！")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
