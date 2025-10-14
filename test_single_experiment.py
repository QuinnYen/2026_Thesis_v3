"""
測試單個實驗的訓練流程
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from src.data_processing.dataset import create_data_loaders
from src.models.baseline import BaselineModel
from src.models.post_fusion import PostFusion_2Layer
from src.training.trainer import ABSATrainer
from src.evaluation.evaluator import ABSAEvaluator


def test_single_experiment():
    """測試單個實驗流程"""
    print("="*80)
    print("測試單個實驗流程")
    print("="*80)

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"設備: {device}")

    # 載入數據
    print("\n[步驟 1] 載入數據...")
    dataloaders = create_data_loaders(
        dataset_name='SemEval2014_Restaurant',
        batch_size=32,
        val_split=0.2,
        random_seed=42,
        data_dir='data/raw',
        use_official_split=True,
        min_freq=2
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    preprocessor = dataloaders['preprocessor']

    print(f"訓練集大小: {len(train_loader.dataset)}")
    print(f"驗證集大小: {len(val_loader.dataset)}")
    print(f"測試集大小: {len(test_loader.dataset)}")
    print(f"詞彙表大小: {preprocessor.vocab_size}")

    # 創建模型
    print("\n[步驟 2] 創建模型...")
    num_classes = len(preprocessor.POLARITY_MAP)
    vocab_size = preprocessor.vocab_size

    model = PostFusion_2Layer(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_size=128,
        num_classes=num_classes,
        dropout=0.3
    )

    model = model.to(device)
    total_params, trainable_params = model.get_num_params()
    print(f"模型參數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")

    # 計算類別權重
    print("\n[步驟 3] 計算類別權重...")
    label_counts = Counter()
    for batch in train_loader:
        labels = batch['label'].numpy()
        label_counts.update(labels)

    print(f"類別分布: {dict(label_counts)}")

    total_samples = sum(label_counts.values())
    class_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"類別權重: {class_weights.cpu().numpy()}")

    # 創建優化器和損失函數
    print("\n[步驟 4] 創建訓練組件...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 創建輸出目錄
    output_dir = Path('outputs/test_experiment')
    checkpoint_dir = output_dir / 'checkpoints'
    viz_dir = output_dir / 'visualizations'
    results_dir = output_dir / 'results'

    for d in [checkpoint_dir, viz_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 創建訓練器
    print("\n[步驟 5] 創建訓練器...")
    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        visualization_dir=str(viz_dir),
        patience=5
    )

    # 訓練模型（只訓練 3 個 epoch 用於測試）
    print("\n[步驟 6] 訓練模型（3 epochs）...")
    history = trainer.train(num_epochs=3)

    print(f"\n訓練完成！")
    print(f"最佳驗證 F1: {max(history['val_f1']):.4f}")
    print(f"最佳驗證 Acc: {max(history['val_acc']):.4f}")

    # 載入最佳模型
    print("\n[步驟 7] 載入最佳模型...")
    best_model_path = checkpoint_dir / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"載入最佳模型 (Epoch {checkpoint['epoch']}, Val F1: {checkpoint['val_f1']:.4f})")
    else:
        print("警告：找不到最佳模型檢查點")

    # 評估模型
    print("\n[步驟 8] 評估模型...")
    evaluator = ABSAEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=list(preprocessor.POLARITY_MAP.keys()),
        save_dir=str(results_dir)
    )

    test_metrics = evaluator.evaluate()

    print(f"\n測試集結果:")
    print(f"  準確度: {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1: {test_metrics['micro_f1']:.4f}")

    # 生成報告
    print("\n[步驟 9] 生成報告...")
    evaluator.plot_confusion_matrix()
    evaluator.generate_report()

    print(f"\n輸出目錄: {output_dir}")
    print(f"  - 檢查點: {checkpoint_dir}")
    print(f"  - 可視化: {viz_dir}")
    print(f"  - 結果: {results_dir}")

    print("\n" + "="*80)
    print("單個實驗測試完成！[OK]")
    print("="*80)


if __name__ == '__main__':
    test_single_experiment()
