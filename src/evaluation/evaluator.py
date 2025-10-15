"""
ABSA 評估器模組

此模組提供完整的模型評估工具，包含：
- 基礎指標計算
- 混淆矩陣
- 錯誤分析
- 注意力視覺化
- 評估報告生成
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# 匯入自定義 AUC 計算模組
from .metrics import (
    calculate_multiclass_auc,
    calculate_roc_curves,
    calculate_micro_auc
)


class ABSAEvaluator:
    """
    ABSA 模型評估器

    提供完整的評估功能，包括指標計算、視覺化和報告生成。

    參數:
        model (nn.Module): 要評估的模型
        data_loader (DataLoader): 測試資料載入器
        device (str, optional): 評估裝置，預設自動選擇
        class_names (List[str], optional): 類別名稱，預設 ['負面', '中性', '正面']
        save_dir (str): 評估結果儲存目錄，預設 'evaluation_results'
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        save_dir: str = 'evaluation_results'
    ):
        # 模型和資料
        self.model = model
        self.data_loader = data_loader

        # 裝置設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # 類別名稱
        if class_names is None:
            self.class_names = ['負面', '中性', '正面']
        else:
            self.class_names = class_names

        # 儲存目錄
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 評估結果
        self.predictions = []
        self.true_labels = []
        self.probabilities = []  # 新增：儲存預測機率用於 AUC 計算
        self.attention_weights = []
        self.input_ids = []
        self.metrics = {}
        self.roc_data = {}  # 新增：儲存 ROC 曲線數據

        print(f"評估器初始化完成")
        print(f"  - 裝置: {self.device}")
        print(f"  - 類別數量: {len(self.class_names)}")
        print(f"  - 類別名稱: {self.class_names}")
        print(f"  - 儲存目錄: {self.save_dir}")

    def evaluate(self) -> Dict[str, float]:
        """
        評估模型並計算所有指標

        返回:
            Dict[str, float]: 包含所有評估指標的字典
        """
        print(f"\n{'='*60}")
        print(f"開始評估")
        print(f"{'='*60}\n")

        self.model.eval()
        self.predictions = []
        self.true_labels = []
        self.probabilities = []  # 重置機率列表
        self.attention_weights = []
        self.input_ids = []

        # 進度條
        pbar = tqdm(self.data_loader, desc='評估中')

        with torch.no_grad():
            for batch in pbar:
                # 取得批次資料
                input_ids = batch['input_ids'].to(self.device)
                aspect_mask = batch['aspect_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向傳播
                logits, attn_weights = self.model(input_ids, aspect_mask)

                # 預測
                predictions = torch.argmax(logits, dim=1)
                # 計算機率（用於 AUC）
                probabilities = torch.softmax(logits, dim=1)

                # 記錄結果
                self.predictions.extend(predictions.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())
                self.probabilities.extend(probabilities.cpu().numpy())  # 記錄機率
                self.attention_weights.extend(attn_weights.cpu().numpy())
                self.input_ids.extend(input_ids.cpu().numpy())

        # 計算指標
        self.metrics = self._calculate_metrics()

        print(f"\n{'='*60}")
        print(f"評估完成")
        print(f"{'='*60}\n")

        self._print_metrics()

        return self.metrics

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        計算所有評估指標

        返回:
            Dict[str, float]: 包含所有指標的字典
        """
        metrics = {}

        # 基礎指標
        metrics['accuracy'] = accuracy_score(self.true_labels, self.predictions)
        metrics['macro_f1'] = f1_score(self.true_labels, self.predictions, average='macro')
        metrics['micro_f1'] = f1_score(self.true_labels, self.predictions, average='micro')
        metrics['macro_precision'] = precision_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(self.true_labels, self.predictions, average='macro', zero_division=0)

        # 每類別指標
        precision_per_class = precision_score(self.true_labels, self.predictions, average=None, zero_division=0)
        recall_per_class = recall_score(self.true_labels, self.predictions, average=None, zero_division=0)
        f1_per_class = f1_score(self.true_labels, self.predictions, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision_per_class[i]
            metrics[f'{class_name}_recall'] = recall_per_class[i]
            metrics[f'{class_name}_f1'] = f1_per_class[i]

        # 計算 AUC 指標
        if len(self.probabilities) > 0:
            y_true = np.array(self.true_labels)
            y_prob = np.array(self.probabilities)

            # 計算 multiclass AUC
            auc_scores = calculate_multiclass_auc(y_true, y_prob, self.class_names)
            metrics.update(auc_scores)

            # 計算 ROC 曲線數據（儲存供後續繪製）
            self.roc_data = calculate_roc_curves(y_true, y_prob, self.class_names)

            # 計算 micro-average AUC
            micro_auc, _, _ = calculate_micro_auc(y_true, y_prob)
            metrics['auc_micro'] = micro_auc

        return metrics

    def _print_metrics(self):
        """顯示評估指標"""
        print("基礎指標:")
        print(f"  - 準確度 (Accuracy): {self.metrics['accuracy']:.4f}")
        print(f"  - Macro-F1: {self.metrics['macro_f1']:.4f}")
        print(f"  - Micro-F1: {self.metrics['micro_f1']:.4f}")
        print(f"  - Macro-Precision: {self.metrics['macro_precision']:.4f}")
        print(f"  - Macro-Recall: {self.metrics['macro_recall']:.4f}")

        # 顯示 AUC 指標
        if 'auc_macro' in self.metrics:
            print(f"\nAUC 指標:")
            print(f"  - Macro-AUC: {self.metrics['auc_macro']:.4f}")
            if 'auc_micro' in self.metrics:
                print(f"  - Micro-AUC: {self.metrics['auc_micro']:.4f}")
            for class_name in self.class_names:
                auc_key = f'auc_{class_name}'
                if auc_key in self.metrics and not np.isnan(self.metrics[auc_key]):
                    print(f"  - {class_name} AUC: {self.metrics[auc_key]:.4f}")

        print("\n每類別指標:")
        for class_name in self.class_names:
            print(f"  {class_name}:")
            print(f"    - Precision: {self.metrics[f'{class_name}_precision']:.4f}")
            print(f"    - Recall: {self.metrics[f'{class_name}_recall']:.4f}")
            print(f"    - F1: {self.metrics[f'{class_name}_f1']:.4f}")

    def plot_confusion_matrix(self, normalize: bool = False) -> Path:
        """
        繪製混淆矩陣

        參數:
            normalize (bool): 是否正規化，預設 False

        返回:
            Path: 圖片儲存路徑
        """
        # 計算混淆矩陣
        cm = confusion_matrix(self.true_labels, self.predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = '正規化混淆矩陣'
        else:
            fmt = 'd'
            title = '混淆矩陣'

        # 繪製
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': '數量' if not normalize else '比例'}
        )
        plt.title(title, fontsize=16, pad=20)
        plt.ylabel('真實標籤', fontsize=12)
        plt.xlabel('預測標籤', fontsize=12)
        plt.tight_layout()

        # 儲存
        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已儲存: {save_path}")
        plt.close()

        return save_path

    def plot_roc_curves(self) -> Path:
        """
        繪製 ROC 曲線（所有類別 + macro-average）

        返回:
            Path: 圖片儲存路徑
        """
        if not self.roc_data:
            print("警告: 沒有 ROC 數據可繪製")
            return None

        # 建立圖形
        plt.figure(figsize=(10, 8))

        # 定義顏色
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']

        # 繪製每個類別的 ROC 曲線
        for idx, class_name in enumerate(self.class_names):
            if class_name in self.roc_data:
                data = self.roc_data[class_name]
                plt.plot(
                    data['fpr'],
                    data['tpr'],
                    color=colors[idx % len(colors)],
                    lw=2,
                    label=f'{class_name} (AUC = {data["auc"]:.4f})'
                )

        # 繪製 macro-average ROC 曲線
        if 'macro' in self.roc_data:
            data = self.roc_data['macro']
            plt.plot(
                data['fpr'],
                data['tpr'],
                color='black',
                lw=3,
                linestyle='--',
                label=f'Macro-average (AUC = {data["auc"]:.4f})'
            )

        # 繪製對角線（隨機猜測）
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3, label='Random Guess')

        # 設定圖形
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 儲存
        save_path = self.save_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC 曲線已儲存: {save_path}")
        plt.close()

        return save_path

    def analyze_errors(self, vocab: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        """
        分析預測錯誤的樣本

        參數:
            vocab (Dict[int, str], optional): 詞彙表，用於將 ID 轉換為文字

        返回:
            pd.DataFrame: 包含錯誤樣本的 DataFrame
        """
        errors = []

        for i, (pred, true) in enumerate(zip(self.predictions, self.true_labels)):
            if pred != true:
                error_info = {
                    'index': i,
                    'true_label': self.class_names[true],
                    'predicted_label': self.class_names[pred],
                    'error_type': f'{self.class_names[true]} -> {self.class_names[pred]}'
                }

                # 如果提供詞彙表，轉換文字
                if vocab is not None:
                    tokens = [vocab.get(idx, '<UNK>') for idx in self.input_ids[i] if idx != 0]
                    error_info['text'] = ' '.join(tokens)

                errors.append(error_info)

        # 建立 DataFrame
        errors_df = pd.DataFrame(errors)

        if len(errors_df) > 0:
            # 統計錯誤類型
            error_counts = errors_df['error_type'].value_counts()
            print(f"\n錯誤分析:")
            print(f"  - 總錯誤數: {len(errors_df)}")
            print(f"  - 錯誤率: {len(errors_df) / len(self.predictions):.2%}")
            print(f"\n錯誤類型分布:")
            for error_type, count in error_counts.items():
                print(f"    {error_type}: {count} ({count/len(errors_df):.2%})")

            # 儲存到 CSV
            csv_path = self.save_dir / 'error_analysis.csv'
            errors_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n錯誤分析已儲存: {csv_path}")
        else:
            print("\n沒有預測錯誤！")

        return errors_df

    def visualize_attention(
        self,
        sample_idx: int,
        vocab: Optional[Dict[int, str]] = None,
        max_len: int = 50
    ) -> Path:
        """
        視覺化注意力權重

        參數:
            sample_idx (int): 樣本索引
            vocab (Dict[int, str], optional): 詞彙表
            max_len (int): 最大顯示長度

        返回:
            Path: 圖片儲存路徑
        """
        # 取得資料
        input_ids = self.input_ids[sample_idx]
        attention = self.attention_weights[sample_idx]
        true_label = self.class_names[self.true_labels[sample_idx]]
        pred_label = self.class_names[self.predictions[sample_idx]]

        # 過濾 padding
        valid_mask = input_ids != 0
        input_ids = input_ids[valid_mask][:max_len]
        attention = attention[valid_mask][:max_len]

        # 轉換為文字
        if vocab is not None:
            tokens = [vocab.get(idx, '<UNK>') for idx in input_ids]
        else:
            tokens = [f'ID_{idx}' for idx in input_ids]

        # 繪製
        fig, ax = plt.subplots(figsize=(15, 3))

        # 繪製注意力權重
        colors = plt.cm.RdYlGn(attention)
        y_pos = np.arange(len(tokens))

        ax.barh(y_pos, attention, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=10)
        ax.set_xlabel('注意力權重', fontsize=12)
        ax.set_title(
            f'注意力視覺化 (樣本 {sample_idx})\n'
            f'真實標籤: {true_label} | 預測標籤: {pred_label}',
            fontsize=14,
            pad=20
        )
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # 儲存
        save_path = self.save_dir / f'attention_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力視覺化已儲存: {save_path}")
        plt.close()

        return save_path

    def visualize_attention_heatmap(
        self,
        sample_indices: List[int],
        vocab: Optional[Dict[int, str]] = None,
        max_len: int = 30
    ) -> Path:
        """
        視覺化多個樣本的注意力熱圖

        參數:
            sample_indices (List[int]): 樣本索引列表
            vocab (Dict[int, str], optional): 詞彙表
            max_len (int): 最大顯示長度

        返回:
            Path: 圖片儲存路徑
        """
        n_samples = len(sample_indices)
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 2 * n_samples))

        if n_samples == 1:
            axes = [axes]

        for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
            # 取得資料
            input_ids = self.input_ids[sample_idx]
            attention = self.attention_weights[sample_idx]
            true_label = self.class_names[self.true_labels[sample_idx]]
            pred_label = self.class_names[self.predictions[sample_idx]]

            # 過濾 padding
            valid_mask = input_ids != 0
            input_ids = input_ids[valid_mask][:max_len]
            attention = attention[valid_mask][:max_len]

            # 轉換為文字
            if vocab is not None:
                tokens = [vocab.get(idx, '<UNK>') for idx in input_ids]
            else:
                tokens = [f'ID_{idx}' for idx in input_ids]

            # 繪製熱圖
            attention_2d = attention.reshape(1, -1)
            im = ax.imshow(attention_2d, cmap='YlOrRd', aspect='auto')

            # 設定刻度
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticks([])
            ax.set_title(
                f'樣本 {sample_idx}: {true_label} -> {pred_label}',
                fontsize=10
            )

            # 添加顏色條
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        # 儲存
        save_path = self.save_dir / 'attention_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力熱圖已儲存: {save_path}")
        plt.close()

        return save_path

    def generate_report(self, vocab: Optional[Dict[int, str]] = None) -> Path:
        """
        生成完整的評估報告（Markdown 格式）

        參數:
            vocab (Dict[int, str], optional): 詞彙表

        返回:
            Path: 報告儲存路徑
        """
        report_lines = []

        # 標題
        report_lines.append("# ABSA 模型評估報告")
        report_lines.append(f"\n生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 基礎資訊
        report_lines.append("## 1. 基礎資訊\n")
        report_lines.append(f"- 測試樣本數: {len(self.predictions)}")
        report_lines.append(f"- 類別數量: {len(self.class_names)}")
        report_lines.append(f"- 類別名稱: {', '.join(self.class_names)}\n")

        # 整體指標
        report_lines.append("## 2. 整體指標\n")
        report_lines.append("| 指標 | 數值 |")
        report_lines.append("|------|------|")
        report_lines.append(f"| 準確度 (Accuracy) | {self.metrics['accuracy']:.4f} |")
        report_lines.append(f"| Macro-F1 | {self.metrics['macro_f1']:.4f} |")
        report_lines.append(f"| Micro-F1 | {self.metrics['micro_f1']:.4f} |")
        report_lines.append(f"| Macro-Precision | {self.metrics['macro_precision']:.4f} |")
        report_lines.append(f"| Macro-Recall | {self.metrics['macro_recall']:.4f} |\n")

        # 每類別指標
        report_lines.append("## 3. 每類別指標\n")
        report_lines.append("| 類別 | Precision | Recall | F1 |")
        report_lines.append("|------|-----------|--------|-----|")
        for class_name in self.class_names:
            report_lines.append(
                f"| {class_name} | "
                f"{self.metrics[f'{class_name}_precision']:.4f} | "
                f"{self.metrics[f'{class_name}_recall']:.4f} | "
                f"{self.metrics[f'{class_name}_f1']:.4f} |"
            )
        report_lines.append("")

        # 混淆矩陣
        report_lines.append("## 4. 混淆矩陣\n")
        report_lines.append("### 4.1 原始計數\n")
        report_lines.append("![混淆矩陣](confusion_matrix.png)\n")
        report_lines.append("### 4.2 正規化\n")
        report_lines.append("![正規化混淆矩陣](confusion_matrix_normalized.png)\n")

        # 錯誤分析
        errors_df = self.analyze_errors(vocab)
        if len(errors_df) > 0:
            report_lines.append("## 5. 錯誤分析\n")
            report_lines.append(f"- 總錯誤數: {len(errors_df)}")
            report_lines.append(f"- 錯誤率: {len(errors_df) / len(self.predictions):.2%}\n")

            report_lines.append("### 錯誤類型分布\n")
            error_counts = errors_df['error_type'].value_counts()
            report_lines.append("| 錯誤類型 | 數量 | 比例 |")
            report_lines.append("|----------|------|------|")
            for error_type, count in error_counts.items():
                report_lines.append(f"| {error_type} | {count} | {count/len(errors_df):.2%} |")
            report_lines.append("")

            report_lines.append("詳細錯誤分析請參考: `error_analysis.csv`\n")

        # 注意力視覺化
        report_lines.append("## 6. 注意力視覺化\n")
        report_lines.append("注意力視覺化範例請參考生成的圖片檔案。\n")

        # 結論
        report_lines.append("## 7. 結論\n")
        best_class = max(self.class_names, key=lambda c: self.metrics[f'{c}_f1'])
        worst_class = min(self.class_names, key=lambda c: self.metrics[f'{c}_f1'])
        report_lines.append(f"- 表現最佳類別: **{best_class}** (F1: {self.metrics[f'{best_class}_f1']:.4f})")
        report_lines.append(f"- 表現最差類別: **{worst_class}** (F1: {self.metrics[f'{worst_class}_f1']:.4f})")
        report_lines.append(f"- 整體準確度: **{self.metrics['accuracy']:.4f}**")
        report_lines.append(f"- 整體 Macro-F1: **{self.metrics['macro_f1']:.4f}**\n")

        # 儲存報告
        report_content = '\n'.join(report_lines)
        report_path = self.save_dir / 'evaluation_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n評估報告已生成: {report_path}")

        # 同時儲存 JSON 格式
        json_path = self.save_dir / 'metrics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"指標已儲存: {json_path}")

        return report_path

    def run_full_evaluation(self, vocab: Optional[Dict[int, str]] = None) -> Dict:
        """
        執行完整評估流程

        參數:
            vocab (Dict[int, str], optional): 詞彙表

        返回:
            Dict: 包含所有評估結果的字典
        """
        # 1. 評估模型
        self.evaluate()

        # 2. 繪製混淆矩陣
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)

        # 3. 繪製 ROC 曲線
        self.plot_roc_curves()

        # 4. 錯誤分析
        errors_df = self.analyze_errors(vocab)

        # 4. 注意力視覺化（隨機選擇幾個樣本）
        if len(self.predictions) > 0:
            # 選擇一些有代表性的樣本
            n_samples = min(5, len(self.predictions))
            sample_indices = np.random.choice(len(self.predictions), n_samples, replace=False)

            for idx in sample_indices[:3]:  # 前3個單獨視覺化
                self.visualize_attention(idx, vocab)

            # 熱圖視覺化
            self.visualize_attention_heatmap(sample_indices.tolist(), vocab)

        # 5. 生成報告
        self.generate_report(vocab)

        return {
            'metrics': self.metrics,
            'errors': errors_df,
            'save_dir': str(self.save_dir)
        }


# 測試程式碼
if __name__ == "__main__":
    """
    測試評估器功能
    """
    print("="*60)
    print("測試 ABSA 評估器")
    print("="*60)

    # 建立假資料集
    from torch.utils.data import TensorDataset

    # 參數設定
    num_samples = 100
    vocab_size = 1000
    seq_len = 50
    batch_size = 16

    # 建立假資料
    test_input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
    test_aspect_mask = torch.zeros(num_samples, seq_len)
    test_aspect_mask[:, 10:15] = 1
    test_labels = torch.randint(0, 3, (num_samples,))

    # 建立 DataLoader
    test_dataset = TensorDataset(test_input_ids, test_aspect_mask, test_labels)

    def collate_fn(batch):
        input_ids, aspect_mask, labels = zip(*batch)
        return {
            'input_ids': torch.stack(input_ids),
            'aspect_mask': torch.stack(aspect_mask),
            'label': torch.stack(labels)
        }

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 建立模型
    import sys
    sys.path.append('..')
    from models.baseline import BaselineModel

    model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_size=64,
        num_classes=3,
        dropout=0.3
    )

    # 建立假詞彙表
    vocab = {i: f'word_{i}' for i in range(vocab_size)}

    # 建立評估器
    evaluator = ABSAEvaluator(
        model=model,
        data_loader=test_loader,
        class_names=['負面', '中性', '正面'],
        save_dir='test_evaluation'
    )

    print("\n[PASS] 評估器初始化成功")

    # 執行完整評估
    print("\n" + "="*60)
    print("執行完整評估")
    print("="*60)

    results = evaluator.run_full_evaluation(vocab)

    print("\n[PASS] 評估完成")
    print(f"\n結果已儲存至: {results['save_dir']}")

    print("\n" + "="*60)
    print("所有測試通過！")
    print("="*60)
