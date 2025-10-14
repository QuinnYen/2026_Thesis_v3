"""
系統性實驗批次訓練腳本

自動化訓練所有模型組合：
- 多個數據集
- 多種模型架構（Baseline, Post-Fusion, Pre-Fusion）
- 多種層數（2-5層）

功能：
1. 自動訓練所有模型組合
2. 記錄所有指標和訓練過程
3. 儲存最佳模型
4. 生成獨立報告
5. 生成總結報告
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 切換到項目根目錄（確保相對路徑正確）
os.chdir(project_root)

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import gc

from src.data_processing.dataset import create_data_loaders
from src.models.baseline import BaselineModel
from src.models.post_fusion import (
    PostFusion_2Layer, PostFusion_3Layer,
    PostFusion_4Layer, PostFusion_5Layer
)
from src.models.pre_fusion import (
    PreFusion_2Layer, PreFusion_3Layer,
    PreFusion_4Layer, PreFusion_5Layer
)
from src.training.trainer import ABSATrainer
from src.evaluation.evaluator import ABSAEvaluator


# 實驗配置
EXPERIMENTS_CONFIG = {
    'datasets': [
        {
            'name': 'SemEval2014_Restaurant',
            'train_path': 'data/raw/SemEval-2014/Restaurants_Train_v2.xml',
            'test_path': 'data/raw/SemEval-2014/Restaurants_Test_Data_phaseB.xml',
        },
        {
            'name': 'SemEval2014_Laptop',
            'train_path': 'data/raw/SemEval-2014/Laptops_Train_v2.xml',
            'test_path': 'data/raw/SemEval-2014/Laptops_Test_Data_phaseB.xml',
        },
    ],
    'models': [
        {'name': 'baseline', 'class': BaselineModel, 'layers': 1},
        {'name': 'post_fusion_2', 'class': PostFusion_2Layer, 'layers': 2},
        {'name': 'post_fusion_3', 'class': PostFusion_3Layer, 'layers': 3},
        {'name': 'post_fusion_4', 'class': PostFusion_4Layer, 'layers': 4},
        {'name': 'post_fusion_5', 'class': PostFusion_5Layer, 'layers': 5},
        {'name': 'pre_fusion_2', 'class': PreFusion_2Layer, 'layers': 2},
        {'name': 'pre_fusion_3', 'class': PreFusion_3Layer, 'layers': 3},
        {'name': 'pre_fusion_4', 'class': PreFusion_4Layer, 'layers': 4},
        {'name': 'pre_fusion_5', 'class': PreFusion_5Layer, 'layers': 5},
    ],
    'training': {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'hidden_size': 128,
        'embedding_dim': 300,
        'dropout': 0.3,
        'patience': 10,
        'val_split': 0.2,
    }
}


class ExperimentRunner:
    """實驗運行器"""

    def __init__(self, config, output_base_dir='outputs/experiments'):
        self.config = config
        self.output_base_dir = Path(output_base_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []

    def run_all_experiments(self):
        """運行所有實驗"""
        print("="*80)
        print("系統性實驗批次訓練")
        print("="*80)
        print(f"設備: {self.device}")
        print(f"數據集數量: {len(self.config['datasets'])}")
        print(f"模型數量: {len(self.config['models'])}")
        print(f"總實驗數: {len(self.config['datasets']) * len(self.config['models'])}")
        print("="*80 + "\n")

        start_time = time.time()

        # 計算總實驗數
        total_experiments = len(self.config['datasets']) * len(self.config['models'])
        experiment_idx = 0

        # 遍歷所有數據集
        for dataset_config in self.config['datasets']:
            dataset_name = dataset_config['name']
            print(f"\n{'='*80}")
            print(f"數據集: {dataset_name}")
            print(f"{'='*80}")

            # 載入數據
            try:
                train_loader, val_loader, test_loader, preprocessor = self._load_dataset(dataset_config)

                print(f"訓練集大小: {len(train_loader.dataset)}")
                print(f"驗證集大小: {len(val_loader.dataset)}")
                if test_loader is not None:
                    print(f"測試集大小: {len(test_loader.dataset)}")
                print(f"詞彙表大小: {preprocessor.vocab_size}")

            except Exception as e:
                print(f"[錯誤] 載入數據集失敗: {e}")
                # 跳過此數據集的所有模型
                experiment_idx += len(self.config['models'])
                continue

            # 遍歷所有模型
            for model_config in self.config['models']:
                experiment_idx += 1
                model_name = model_config['name']

                print(f"\n{'-'*80}")
                print(f"[{experiment_idx}/{total_experiments}] 模型: {model_name}")
                print(f"{'-'*80}")

                try:
                    # 運行單個實驗
                    result = self._run_single_experiment(
                        dataset_config=dataset_config,
                        model_config=model_config,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        preprocessor=preprocessor
                    )

                    self.results.append(result)

                    # 顯示結果摘要
                    print(f"\n實驗完成:")
                    print(f"  準確度: {result['test_metrics']['accuracy']:.4f}")
                    print(f"  Macro-F1: {result['test_metrics']['macro_f1']:.4f}")
                    print(f"  訓練時間: {result['training_time']:.2f}s")

                except Exception as e:
                    print(f"[錯誤] 實驗失敗: {e}")
                    import traceback
                    traceback.print_exc()

                    # 記錄失敗
                    self.results.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'status': 'failed',
                        'error': str(e)
                    })

                # 清理記憶體
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_time = time.time() - start_time

        # 生成總結報告
        self._generate_summary_report(total_time)

        print(f"\n{'='*80}")
        print(f"所有實驗完成！")
        print(f"總時間: {total_time/60:.2f} 分鐘")
        print(f"成功: {sum(1 for r in self.results if r.get('status') != 'failed')}/{len(self.results)}")
        print(f"失敗: {sum(1 for r in self.results if r.get('status') == 'failed')}/{len(self.results)}")
        print(f"{'='*80}")

    def _load_dataset(self, dataset_config):
        """載入數據集"""
        dataset_name = dataset_config['name']

        dataloaders = create_data_loaders(
            dataset_name=dataset_name,
            batch_size=self.config['training']['batch_size'],
            val_split=self.config['training']['val_split'],
            random_seed=42,
            data_dir='data/raw',
            use_official_split=True,
            min_freq=2
        )

        return dataloaders

    def _run_single_experiment(
        self,
        dataset_config,
        model_config,
        train_loader,
        val_loader,
        test_loader,
        preprocessor
    ):
        """運行單個實驗"""
        dataset_name = dataset_config['name']
        model_name = model_config['name']

        # 創建輸出目錄
        output_dir = self.output_base_dir / dataset_name / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = output_dir / 'checkpoints'
        results_dir = output_dir / 'results'
        viz_dir = output_dir / 'visualizations'

        for d in [checkpoint_dir, results_dir, viz_dir]:
            d.mkdir(exist_ok=True)

        # 創建模型
        num_classes = len(preprocessor.POLARITY_MAP)
        vocab_size = preprocessor.vocab_size

        model = model_config['class'](
            vocab_size=vocab_size,
            embedding_dim=self.config['training']['embedding_dim'],
            hidden_size=self.config['training']['hidden_size'],
            num_classes=num_classes,
            dropout=self.config['training']['dropout']
        )

        model = model.to(self.device)

        # 計算類別權重（處理類別不平衡）
        from collections import Counter
        label_counts = Counter()
        for batch in train_loader:
            labels = batch['label'].numpy()
            label_counts.update(labels)

        total_samples = sum(label_counts.values())
        class_weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)

        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # 創建損失函數和優化器
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate']
        )

        # 創建訓練器
        trainer = ABSATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            save_dir=str(checkpoint_dir),
            patience=self.config['training']['patience']
        )

        # 訓練模型
        print(f"開始訓練...")
        train_start = time.time()
        history = trainer.train(num_epochs=self.config['training']['num_epochs'])
        training_time = time.time() - train_start

        # 載入最佳模型
        best_model_path = checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"載入最佳模型 (Epoch {checkpoint['epoch']}, Val F1: {checkpoint['best_val_f1']:.4f})")

        # 評估模型
        print(f"評估模型...")
        # 只使用實際的類別名稱（排除 conflict，因為它映射到 neutral）
        class_names = ['negative', 'neutral', 'positive']
        evaluator = ABSAEvaluator(
            model=model,
            data_loader=val_loader,
            device=self.device,
            class_names=class_names,
            save_dir=str(results_dir)
        )

        test_metrics = evaluator.evaluate()
        evaluator.plot_confusion_matrix()
        evaluator.generate_report()

        # 準備結果
        result = {
            'dataset': dataset_name,
            'model': model_name,
            'model_layers': model_config['layers'],
            'status': 'success',
            'training_time': training_time,
            'best_epoch': checkpoint.get('epoch', -1) if best_model_path.exists() else -1,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset) if test_loader is not None else 0,
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'macro_f1': test_metrics['macro_f1'],
                'micro_f1': test_metrics['micro_f1'],
                'macro_precision': test_metrics['macro_precision'],
                'macro_recall': test_metrics['macro_recall'],
                'negative_f1': test_metrics.get('negative_f1', 0.0),
                'neutral_f1': test_metrics.get('neutral_f1', 0.0),
                'positive_f1': test_metrics.get('positive_f1', 0.0)
            },
            'history': history,
            'output_dir': str(output_dir)
        }

        # 儲存實驗結果
        result_file = results_dir / 'experiment_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            # 將不可序列化的對象轉換
            serializable_result = {
                k: v for k, v in result.items()
                if k not in ['history']  # history 包含 numpy arrays
            }
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        return result

    def _generate_summary_report(self, total_time):
        """生成總結報告"""
        print(f"\n{'='*80}")
        print("生成總結報告...")
        print(f"{'='*80}")

        summary_dir = self.output_base_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)

        # 準備總結數據
        # 創建可序列化的配置副本（移除 class 物件）
        serializable_config = {
            'datasets': self.config['datasets'],
            'models': [
                {
                    'name': m['name'],
                    'layers': m['layers']
                }
                for m in self.config['models']
            ],
            'training': self.config['training']
        }

        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_experiments': len(self.results),
            'successful_experiments': sum(1 for r in self.results if r.get('status') != 'failed'),
            'failed_experiments': sum(1 for r in self.results if r.get('status') == 'failed'),
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'config': serializable_config,
            'results': []
        }

        # 收集成功的實驗結果
        for result in self.results:
            if result.get('status') == 'failed':
                summary['results'].append({
                    'dataset': result['dataset'],
                    'model': result['model'],
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                })
            else:
                summary['results'].append({
                    'dataset': result['dataset'],
                    'model': result['model'],
                    'model_layers': result['model_layers'],
                    'status': 'success',
                    'accuracy': result['test_metrics']['accuracy'],
                    'macro_f1': result['test_metrics']['macro_f1'],
                    'micro_f1': result['test_metrics']['micro_f1'],
                    'training_time': result['training_time'],
                    'best_epoch': result['best_epoch'],
                    'output_dir': result['output_dir']
                })

        # 儲存 JSON 報告
        summary_file = summary_dir / 'experiments_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"總結報告已儲存: {summary_file}")

        # 生成 Markdown 報告
        self._generate_markdown_report(summary, summary_dir)

        # 生成性能比較表
        self._generate_performance_table(summary, summary_dir)

    def _generate_markdown_report(self, summary, summary_dir):
        """生成 Markdown 格式的報告"""
        md_file = summary_dir / 'experiments_report.md'

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# 系統性實驗總結報告\n\n")
            f.write(f"**生成時間:** {summary['timestamp']}\n\n")
            f.write(f"**總實驗數:** {summary['total_experiments']}\n")
            f.write(f"**成功:** {summary['successful_experiments']}\n")
            f.write(f"**失敗:** {summary['failed_experiments']}\n")
            f.write(f"**總時間:** {summary['total_time_minutes']:.2f} 分鐘\n\n")

            f.write("## 實驗配置\n\n")
            f.write(f"- **訓練 Epochs:** {self.config['training']['num_epochs']}\n")
            f.write(f"- **批次大小:** {self.config['training']['batch_size']}\n")
            f.write(f"- **學習率:** {self.config['training']['learning_rate']}\n")
            f.write(f"- **隱藏層大小:** {self.config['training']['hidden_size']}\n")
            f.write(f"- **Dropout:** {self.config['training']['dropout']}\n")
            f.write(f"- **Early Stopping Patience:** {self.config['training']['patience']}\n\n")

            # 按數據集分組
            datasets = {}
            for result in summary['results']:
                if result['status'] != 'failed':
                    dataset = result['dataset']
                    if dataset not in datasets:
                        datasets[dataset] = []
                    datasets[dataset].append(result)

            for dataset_name, results in datasets.items():
                f.write(f"## {dataset_name}\n\n")
                f.write("| 模型 | 層數 | 準確度 | Macro-F1 | Micro-F1 | 訓練時間(s) | Best Epoch |\n")
                f.write("|------|------|--------|----------|----------|-------------|------------|\n")

                # 按 F1 分數排序
                results_sorted = sorted(results, key=lambda x: x['macro_f1'], reverse=True)

                for result in results_sorted:
                    f.write(
                        f"| {result['model']} | "
                        f"{result['model_layers']} | "
                        f"{result['accuracy']:.4f} | "
                        f"{result['macro_f1']:.4f} | "
                        f"{result['micro_f1']:.4f} | "
                        f"{result['training_time']:.2f} | "
                        f"{result['best_epoch']} |\n"
                    )

                f.write("\n")

            # 失敗的實驗
            failed_results = [r for r in summary['results'] if r['status'] == 'failed']
            if failed_results:
                f.write("## 失敗的實驗\n\n")
                f.write("| 數據集 | 模型 | 錯誤信息 |\n")
                f.write("|--------|------|----------|\n")
                for result in failed_results:
                    error_msg = result.get('error', 'Unknown')[:100]  # 限制長度
                    f.write(f"| {result['dataset']} | {result['model']} | {error_msg} |\n")
                f.write("\n")

        print(f"Markdown 報告已儲存: {md_file}")

    def _generate_performance_table(self, summary, summary_dir):
        """生成性能比較表（CSV格式）"""
        csv_file = summary_dir / 'performance_comparison.csv'

        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Dataset,Model,Layers,Accuracy,Macro-F1,Micro-F1,Training_Time,Best_Epoch\n")

            for result in summary['results']:
                if result['status'] != 'failed':
                    f.write(
                        f"{result['dataset']},"
                        f"{result['model']},"
                        f"{result['model_layers']},"
                        f"{result['accuracy']:.4f},"
                        f"{result['macro_f1']:.4f},"
                        f"{result['micro_f1']:.4f},"
                        f"{result['training_time']:.2f},"
                        f"{result['best_epoch']}\n"
                    )

        print(f"性能比較表已儲存: {csv_file}")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='系統性實驗批次訓練')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/experiments',
        help='輸出目錄'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['restaurant', 'laptop', 'all'],
        default=['all'],
        help='要訓練的數據集'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['baseline', 'post_fusion', 'pre_fusion', 'all'],
        default=['all'],
        help='要訓練的模型類型'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='訓練 epochs 數'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小'
    )

    args = parser.parse_args()

    # 根據參數過濾配置
    config = EXPERIMENTS_CONFIG.copy()

    # 過濾數據集
    if 'all' not in args.datasets:
        filtered_datasets = []
        for ds in config['datasets']:
            if 'restaurant' in args.datasets and 'Restaurant' in ds['name']:
                filtered_datasets.append(ds)
            elif 'laptop' in args.datasets and 'Laptop' in ds['name']:
                filtered_datasets.append(ds)
        config['datasets'] = filtered_datasets

    # 過濾模型
    if 'all' not in args.models:
        filtered_models = []
        for model in config['models']:
            if 'baseline' in args.models and model['name'] == 'baseline':
                filtered_models.append(model)
            elif 'post_fusion' in args.models and 'post_fusion' in model['name']:
                filtered_models.append(model)
            elif 'pre_fusion' in args.models and 'pre_fusion' in model['name']:
                filtered_models.append(model)
        config['models'] = filtered_models

    # 更新訓練參數
    config['training']['num_epochs'] = args.num_epochs
    config['training']['batch_size'] = args.batch_size

    # 運行實驗
    runner = ExperimentRunner(config, output_base_dir=args.output_dir)
    runner.run_all_experiments()


if __name__ == '__main__':
    main()
