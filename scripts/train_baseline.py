"""
Baseline 模型訓練腳本

此腳本實現完整的訓練流程：
1. 載入 SemEval-2014 Restaurant 資料集
2. 建立 DataLoader
3. 初始化 Baseline 模型
4. 訓練模型（含 early stopping）
5. 在測試集上評估
6. 生成完整報告
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
from datetime import datetime
import pickle

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.data_loader import SemEvalDataLoader
from src.data_processing.preprocessor import ABSAPreprocessor
from src.data_processing.dataset import create_data_loaders
from src.models.baseline import BaselineModel
from src.training.trainer import ABSATrainer
from src.evaluation.evaluator import ABSAEvaluator


def set_seed(seed: int = 42):
    """設定隨機種子以確保可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embedding_matrix(embedding_path: str, vocab_size: int, embedding_dim: int):
    """
    載入預訓練的詞嵌入矩陣

    Args:
        embedding_path: 詞嵌入矩陣路徑
        vocab_size: 詞彙表大小
        embedding_dim: 嵌入維度

    Returns:
        torch.Tensor: 詞嵌入矩陣
    """
    if os.path.exists(embedding_path):
        print(f"載入預訓練詞嵌入: {embedding_path}")
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)

        # 檢查是字典還是矩陣
        if isinstance(data, dict):
            embedding_matrix = data['embedding_matrix']
            print(f"  載入的詞嵌入維度: {data.get('embedding_dim', 'N/A')}")
            print(f"  載入的詞彙表大小: {data.get('vocab_size', 'N/A')}")
        else:
            embedding_matrix = data

        # 確保形狀正確
        if embedding_matrix.shape[0] == vocab_size and embedding_matrix.shape[1] == embedding_dim:
            print(f"  詞嵌入矩陣形狀: {embedding_matrix.shape} [OK]")
            return torch.FloatTensor(embedding_matrix)
        else:
            print(f"警告：詞嵌入矩陣形狀不匹配")
            print(f"  預期: ({vocab_size}, {embedding_dim})")
            print(f"  實際: {embedding_matrix.shape}")
            print(f"  使用隨機初始化")
            return None
    else:
        print(f"警告：找不到詞嵌入檔案: {embedding_path}")
        print(f"  使用隨機初始化")
        return None


def create_output_directories(base_dir: str = 'outputs'):
    """
    建立輸出目錄結構

    Args:
        base_dir: 基礎輸出目錄

    Returns:
        Dict: 包含各個目錄路徑的字典
    """
    base_path = Path(base_dir)
    dirs = {
        'base': base_path,
        'checkpoints': base_path / 'checkpoints',
        'visualizations': base_path / 'visualizations',
        'results': base_path / 'results',
        'logs': base_path / 'logs'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def train_baseline(args):
    """
    訓練 Baseline 模型

    Args:
        args: 命令列參數
    """
    print("="*80)
    print("Baseline 模型訓練")
    print("="*80)
    print()

    # 設定隨機種子
    set_seed(args.seed)
    print(f"隨機種子: {args.seed}")

    # 建立輸出目錄
    print("\n[步驟 1] 建立輸出目錄")
    print("-"*80)
    output_dirs = create_output_directories(args.output_dir)
    print(f"輸出目錄: {output_dirs['base']}")
    for name, path in output_dirs.items():
        if name != 'base':
            print(f"  - {name}: {path}")

    # 載入資料並建立 DataLoaders
    print("\n[步驟 2] 載入資料集並建立 DataLoaders")
    print("-"*80)
    train_loader, val_loader, test_loader, preprocessor = create_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        random_seed=args.seed,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        use_official_split=args.use_official_split,
        min_freq=args.min_freq
    )

    # 儲存預處理器（用於後續推理）
    preprocessor_path = output_dirs['checkpoints'] / 'preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"\n預處理器已儲存: {preprocessor_path}")

    # 取得詞彙表資訊
    vocab_size = preprocessor.vocab_size
    print(f"\n詞彙表大小: {vocab_size}")

    # 載入預訓練詞嵌入（如果可用）
    print("\n[步驟 3] 載入預訓練詞嵌入")
    print("-"*80)
    embedding_matrix = load_embedding_matrix(
        args.embedding_path,
        vocab_size,
        args.embedding_dim
    )

    # 初始化模型
    print("\n[步驟 4] 初始化 Baseline 模型")
    print("-"*80)
    # 取得實際類別數量（去重後的標籤值數量）
    num_classes = len(set(preprocessor.POLARITY_MAP.values()))  # 3: 負面(0)、中性(1)、正面(2)
    print(f"類別數量: {num_classes}")

    model = BaselineModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=args.freeze_embeddings
    )

    total_params, trainable_params = model.get_num_params()
    print(f"模型參數統計:")
    print(f"  - 總參數量: {total_params:,}")
    print(f"  - 可訓練參數量: {trainable_params:,}")
    print(f"  - 凍結參數量: {total_params - trainable_params:,}")

    # 計算類別權重（處理類別不平衡）
    print("\n計算類別權重...")
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

    class_weights = torch.FloatTensor(class_weights).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"類別分布: {dict(label_counts)}")
    print(f"類別權重: {class_weights.cpu().numpy()}")

    # 使用加權損失函數
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 建立優化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 學習率調度器（可選）
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        print("使用學習率調度器：ReduceLROnPlateau（模式：max，降低因子：0.5，patience：3）")

    # 建立訓練器
    print("\n[步驟 5] 建立訓練器")
    print("-"*80)
    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,  # 傳入加權損失函數
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        save_dir=str(output_dirs['checkpoints']),
        scheduler=scheduler
    )

    # 訓練模型
    print("\n[步驟 6] 開始訓練")
    print("-"*80)
    start_time = datetime.now()
    history = trainer.train(num_epochs=args.num_epochs)
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print(f"\n訓練完成！總耗時: {training_time:.2f} 秒 ({training_time/60:.2f} 分鐘)")

    # 移動訓練曲線到視覺化目錄
    training_curve_src = output_dirs['checkpoints'] / 'training_curves.png'
    training_curve_dst = output_dirs['visualizations'] / 'baseline_training_curve.png'
    if training_curve_src.exists():
        import shutil
        shutil.move(str(training_curve_src), str(training_curve_dst))
        print(f"訓練曲線已儲存: {training_curve_dst}")

    # 在測試集上評估（如果有的話）
    if test_loader is not None:
        eval_loader = test_loader
        eval_name = "測試集"
    else:
        eval_loader = val_loader
        eval_name = "驗證集"

    print(f"\n[步驟 7] 在{eval_name}上評估")
    print("-"*80)

    # 載入最佳模型
    print("載入最佳模型...")
    checkpoint = trainer.load_checkpoint('best_model.pt')
    best_epoch = checkpoint['epoch']
    best_f1 = checkpoint['metrics']['f1']
    print(f"最佳模型來自 Epoch {best_epoch}，驗證 F1: {best_f1:.4f}")

    # 建立評估器
    class_names = ['負面', '中性', '正面', '衝突'][:num_classes]  # 根據實際類別數量
    evaluator = ABSAEvaluator(
        model=model,
        data_loader=eval_loader,
        class_names=class_names,
        save_dir=str(output_dirs['results'])
    )

    # 執行完整評估
    print(f"\n執行完整評估...")
    vocab_dict = {v: k for k, v in preprocessor.word2idx.items()}  # 反向詞彙表
    results = evaluator.run_full_evaluation(vocab=vocab_dict)

    # 移動混淆矩陣到視覺化目錄
    confusion_matrix_src = output_dirs['results'] / 'confusion_matrix.png'
    confusion_matrix_dst = output_dirs['visualizations'] / 'baseline_confusion_matrix.png'
    if confusion_matrix_src.exists():
        import shutil
        shutil.copy(str(confusion_matrix_src), str(confusion_matrix_dst))
        print(f"混淆矩陣已複製: {confusion_matrix_dst}")

    # 移動評估報告
    report_src = output_dirs['results'] / 'evaluation_report.md'
    report_dst = output_dirs['results'] / 'baseline_report.md'
    if report_src.exists():
        import shutil
        shutil.move(str(report_src), str(report_dst))
        print(f"評估報告已儲存: {report_dst}")

    # 移動最佳模型
    best_model_src = output_dirs['checkpoints'] / 'best_model.pt'
    best_model_dst = output_dirs['checkpoints'] / 'baseline_best.pt'
    if best_model_src.exists():
        import shutil
        shutil.copy(str(best_model_src), str(best_model_dst))
        print(f"最佳模型已複製: {best_model_dst}")

    # 生成訓練摘要
    print("\n[步驟 8] 生成訓練摘要")
    print("-"*80)

    summary = {
        'dataset': args.dataset,
        'model': 'Baseline (BiLSTM + Attention)',
        'training': {
            'num_epochs': args.num_epochs,
            'actual_epochs': best_epoch,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'optimizer': 'Adam',
            'early_stopping_patience': args.patience,
            'training_time_seconds': training_time,
            'best_epoch': best_epoch
        },
        'model_config': {
            'vocab_size': vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_size': args.hidden_size,
            'dropout': args.dropout,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'data': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(eval_loader.dataset) if test_loader else 0
        },
        'results': {
            'accuracy': results['metrics']['accuracy'],
            'macro_f1': results['metrics']['macro_f1'],
            'micro_f1': results['metrics']['micro_f1'],
            'macro_precision': results['metrics']['macro_precision'],
            'macro_recall': results['metrics']['macro_recall']
        },
        'per_class': {
            name: {
                'precision': results['metrics'][f'{name}_precision'],
                'recall': results['metrics'][f'{name}_recall'],
                'f1': results['metrics'][f'{name}_f1']
            }
            for name in class_names
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 儲存訓練摘要
    summary_path = output_dirs['results'] / 'baseline_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"訓練摘要已儲存: {summary_path}")

    # 顯示最終結果
    print("\n" + "="*80)
    print("訓練完成摘要")
    print("="*80)
    print(f"\n資料集: {args.dataset}")
    print(f"訓練樣本數: {len(train_loader.dataset)}")
    print(f"驗證樣本數: {len(val_loader.dataset)}")
    print(f"測試樣本數: {len(eval_loader.dataset) if test_loader else 0}")
    print(f"\n訓練配置:")
    print(f"  - Epochs: {best_epoch} (最佳) / {args.num_epochs} (最大)")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Training Time: {training_time:.2f}s ({training_time/60:.2f}m)")
    print(f"\n模型配置:")
    print(f"  - Embedding Dim: {args.embedding_dim}")
    print(f"  - Hidden Size: {args.hidden_size}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Total Params: {total_params:,}")
    print(f"  - Trainable Params: {trainable_params:,}")
    print(f"\n{eval_name}結果:")
    print(f"  - 準確度: {results['metrics']['accuracy']:.4f} ({results['metrics']['accuracy']*100:.2f}%)")
    print(f"  - Macro-F1: {results['metrics']['macro_f1']:.4f}")
    print(f"  - Micro-F1: {results['metrics']['micro_f1']:.4f}")
    print(f"  - Precision: {results['metrics']['macro_precision']:.4f}")
    print(f"  - Recall: {results['metrics']['macro_recall']:.4f}")
    print(f"\n每類別 F1:")
    for name in class_names:
        print(f"  - {name}: {results['metrics'][f'{name}_f1']:.4f}")
    print(f"\n輸出檔案:")
    print(f"  - 訓練曲線: {training_curve_dst}")
    print(f"  - 混淆矩陣: {confusion_matrix_dst}")
    print(f"  - 評估報告: {report_dst}")
    print(f"  - 最佳模型: {best_model_dst}")
    print(f"  - 訓練摘要: {summary_path}")

    # 檢查是否達標
    print("\n" + "="*80)
    print("驗證標準檢查")
    print("="*80)

    accuracy_threshold = 0.75
    f1_threshold = 0.70

    accuracy_pass = results['metrics']['accuracy'] >= accuracy_threshold
    f1_pass = results['metrics']['macro_f1'] >= f1_threshold

    print(f"\n準確度: {results['metrics']['accuracy']:.4f} ", end='')
    if accuracy_pass:
        print(f"[PASS] (>= {accuracy_threshold})")
    else:
        print(f"[FAIL] (< {accuracy_threshold})")

    print(f"Macro-F1: {results['metrics']['macro_f1']:.4f} ", end='')
    if f1_pass:
        print(f"[PASS] (>= {f1_threshold})")
    else:
        print(f"[FAIL] (< {f1_threshold})")

    if accuracy_pass and f1_pass:
        print("\n[成功] 所有驗證標準已達標！")
    else:
        print("\n[警告] 未達驗證標準，建議檢查以下項目：")
        print("  - 增加訓練 epochs")
        print("  - 調整學習率")
        print("  - 使用預訓練詞嵌入")
        print("  - 調整模型超參數（hidden_size, dropout 等）")

    print("\n" + "="*80)
    print("訓練流程完成！")
    print("="*80)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='訓練 Baseline ABSA 模型')

    # 資料相關
    parser.add_argument('--dataset', type=str, default='SemEval2014_Restaurant',
                      help='資料集名稱')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='資料目錄')
    parser.add_argument('--embedding_path', type=str, default='data/processed/embedding_matrix_final.pkl',
                      help='預訓練詞嵌入路徑')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='輸出目錄')

    # 資料處理相關
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='驗證集比例')
    parser.add_argument('--use_official_split', action='store_true',
                      help='使用官方測試集作為驗證集')
    parser.add_argument('--min_freq', type=int, default=2,
                      help='詞彙表最小詞頻')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='DataLoader 工作進程數')

    # 模型相關
    parser.add_argument('--embedding_dim', type=int, default=300,
                      help='詞嵌入維度')
    parser.add_argument('--hidden_size', type=int, default=128,
                      help='LSTM 隱藏層維度')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout 比率')
    parser.add_argument('--freeze_embeddings', action='store_true',
                      help='凍結詞嵌入層')

    # 訓練相關
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='訓練 epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='學習率')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                      help='梯度裁剪最大範數')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early Stopping 耐心值')
    parser.add_argument('--use_scheduler', action='store_true',
                      help='使用學習率調度器')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                      help='隨機種子')

    args = parser.parse_args()

    # 執行訓練
    train_baseline(args)


if __name__ == "__main__":
    main()
