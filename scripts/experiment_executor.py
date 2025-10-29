"""
實驗執行器模組

負責將多維度實驗配置轉換為實際的訓練任務並執行
橋接 experiment_dimensions.py 和實際的訓練邏輯
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 加入專案路徑
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "models"))
sys.path.insert(0, str(BASE_DIR / "src" / "data_processing"))
sys.path.insert(0, str(BASE_DIR / "configs"))

from models.baseline import BaselineModel
from models.baseline_no_attention import BaselineNoAttention
from models.pre_fusion import PreFusionModel
from models.post_fusion import PostFusionModel
from data_processing.cleaned_data_loader import create_cleaned_data_loaders, get_vocab_words
from data_processing.embedding_loader import load_glove_embeddings
from experiment_config import get_dataset_path, BASELINE_CONFIG, PREFUSION_CONFIG, POSTFUSION_CONFIG


# ============================================================================
# 配置解析器
# ============================================================================

class ConfigParser:
    """配置解析器：將維度配置轉換為訓練配置"""

    @staticmethod
    def parse_dimension_config(experiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析維度實驗配置，轉換為訓練配置

        Args:
            experiment: 維度實驗配置

        Returns:
            訓練配置字典
        """
        dimension = experiment['dimension']
        config = experiment['config'].copy()
        dataset = experiment['dataset']

        # 基礎配置
        train_config = {
            'dataset': dataset,
            'dimension': dimension,
            'experiment_group': experiment['group']
        }

        # 根據維度解析配置
        if dimension == 1:
            # 維度1：基礎架構
            train_config.update(ConfigParser._parse_dimension_1(config))

        elif dimension == 2:
            # 維度2：詞向量
            train_config.update(ConfigParser._parse_dimension_2(config))

        elif dimension == 3:
            # 維度3：編碼器
            train_config.update(ConfigParser._parse_dimension_3(config))

        elif dimension == 4:
            # 維度4：句法信息
            train_config.update(ConfigParser._parse_dimension_4(config))

        elif dimension == 5:
            # 維度5：多種子
            train_config.update(ConfigParser._parse_dimension_5(config))

        return train_config

    @staticmethod
    def _parse_dimension_1(config: Dict[str, Any]) -> Dict[str, Any]:
        """解析維度1：基礎架構配置"""
        model_type = config.get('model_type')
        encoder_type = config.get('encoder_type', 'bilstm')
        bidirectional = config.get('bidirectional', True)
        layers = config.get('layers', [2])

        # 基礎配置（從 experiment_config 繼承）
        if model_type == 'baseline_no_attention':
            base_config = BASELINE_CONFIG.copy()
            base_config['use_attention'] = False
        elif model_type in ['prefusion', 'pre_fusion']:
            base_config = PREFUSION_CONFIG.copy()
        elif model_type in ['postfusion', 'post_fusion']:
            base_config = POSTFUSION_CONFIG.copy()
        else:
            base_config = BASELINE_CONFIG.copy()

        return {
            'model_type': model_type,
            'encoder_type': encoder_type,
            'bidirectional': bidirectional,
            'layers': layers,
            'base_config': base_config
        }

    @staticmethod
    def _parse_dimension_2(config: Dict[str, Any]) -> Dict[str, Any]:
        """解析維度2：詞向量配置"""
        embedding_type = config.get('embedding_type', 'random')
        embedding_dim = config.get('embedding_dim', 300)
        freeze_embeddings = config.get('freeze_embeddings', False)
        models = config.get('models', ['baseline'])
        layers = config.get('layers', 2)

        return {
            'embedding_type': embedding_type,
            'embedding_dim': embedding_dim,
            'freeze_embeddings': freeze_embeddings,
            'models': models,
            'layers': layers,
            'base_config': BASELINE_CONFIG.copy()
        }

    @staticmethod
    def _parse_dimension_3(config: Dict[str, Any]) -> Dict[str, Any]:
        """解析維度3：編碼器配置"""
        encoder_type = config.get('encoder_type', 'bilstm')
        models = config.get('models', ['prefusion'])
        layers = config.get('layers', 2)

        bert_config = {}
        if encoder_type == 'bert':
            bert_config = {
                'bert_model': config.get('bert_model', 'bert-base-uncased'),
                'freeze_bert': config.get('freeze_bert', False)
            }

        return {
            'encoder_type': encoder_type,
            'models': models,
            'layers': layers,
            'bert_config': bert_config,
            'base_config': PREFUSION_CONFIG.copy()
        }

    @staticmethod
    def _parse_dimension_4(config: Dict[str, Any]) -> Dict[str, Any]:
        """解析維度4：句法信息配置"""
        use_syntax = config.get('use_syntax', False)
        models = config.get('models', ['prefusion'])
        layers = config.get('layers', 2)

        syntax_config = {}
        if use_syntax:
            syntax_config = {
                'syntax_type': config.get('syntax_type', 'dependency'),
                'gnn_type': config.get('gnn_type', 'gcn'),
                'gnn_layers': config.get('gnn_layers', 2)
            }

        return {
            'use_syntax': use_syntax,
            'models': models,
            'layers': layers,
            'syntax_config': syntax_config,
            'base_config': PREFUSION_CONFIG.copy()
        }

    @staticmethod
    def _parse_dimension_5(config: Dict[str, Any]) -> Dict[str, Any]:
        """解析維度5：多種子配置"""
        seeds = config.get('seeds', [42])
        report_stats = config.get('report_stats', False)
        run_significance_test = config.get('run_significance_test', False)

        return {
            'seeds': seeds,
            'report_stats': report_stats,
            'run_significance_test': run_significance_test,
            'base_config': BASELINE_CONFIG.copy()
        }


# ============================================================================
# 訓練執行器
# ============================================================================

class TrainingExecutor:
    """訓練執行器"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def execute(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        執行訓練

        Args:
            train_config: 訓練配置

        Returns:
            訓練結果
        """
        dimension = train_config['dimension']

        # 根據維度選擇執行方法
        if dimension == 1:
            return self._execute_dimension_1(train_config)
        elif dimension == 2:
            return self._execute_dimension_2(train_config)
        elif dimension == 3:
            return self._execute_dimension_3(train_config)
        elif dimension == 4:
            return self._execute_dimension_4(train_config)
        elif dimension == 5:
            return self._execute_dimension_5(train_config)
        else:
            raise ValueError(f"不支援的維度: {dimension}")

    def _execute_dimension_1(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """執行維度1：基礎架構實驗"""
        dataset = train_config['dataset']
        model_type = train_config['model_type']
        layers_list = train_config['layers']
        base_config = train_config['base_config']
        bidirectional = train_config.get('bidirectional', True)

        print(f"\n執行維度1實驗：{model_type}")
        print(f"數據集：{dataset}")
        print(f"層數：{layers_list}")
        print(f"雙向：{bidirectional}")

        # 載入數據（對於所有層數共用）
        train_loader, val_loader, vocab, label_map, class_weights = self._load_data(
            dataset, base_config
        )

        # 訓練每個層數配置
        results = []
        for num_layers in (layers_list if isinstance(layers_list, list) else [layers_list]):
            print(f"\n--- 訓練 {num_layers} 層 ---")

            result = self._train_single_model(
                dataset=dataset,
                model_type=model_type,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                vocab=vocab,
                class_weights=class_weights,
                config=base_config,
                bidirectional=bidirectional
            )

            if result:
                results.append(result)

        # 返回最佳結果
        if results:
            best_result = max(results, key=lambda x: x['final_metrics']['macro_f1'])
            # 只保留所有層數的摘要信息，避免嵌套過深
            best_result['all_layers_summary'] = [
                {
                    'num_layers': r['num_layers'],
                    'macro_f1': r['final_metrics']['macro_f1'],
                    'accuracy': r['final_metrics']['accuracy']
                }
                for r in results
            ]
            return best_result

        return None

    def _execute_dimension_2(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """執行維度2：詞向量實驗"""
        print(f"\n[待實現] 執行維度2實驗")
        # TODO: 實現詞向量實驗邏輯
        return None

    def _execute_dimension_3(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """執行維度3：編碼器實驗"""
        print(f"\n[待實現] 執行維度3實驗")
        # TODO: 實現編碼器實驗邏輯
        return None

    def _execute_dimension_4(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """執行維度4：句法信息實驗"""
        print(f"\n[待實現] 執行維度4實驗")
        # TODO: 實現句法信息實驗邏輯
        return None

    def _execute_dimension_5(self, train_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """執行維度5：多種子穩定性實驗"""
        print(f"\n[待實現] 執行維度5實驗")
        # TODO: 實現多種子實驗邏輯
        return None

    def _load_data(self, dataset: str, config: Dict[str, Any]) -> Tuple:
        """載入數據"""
        print(f"\n載入數據：{dataset}")

        csv_path = get_dataset_path(dataset, "train")

        train_loader, val_loader, vocab, label_map, class_weights = create_cleaned_data_loaders(
            train_csv=csv_path,
            batch_size=config.get('batch_size', 32),
            val_split=0.2,
            max_length=config.get('max_length', 128),
            min_freq=2,
            random_seed=42
        )

        print(f"訓練樣本：{len(train_loader.dataset)}")
        print(f"驗證樣本：{len(val_loader.dataset)}")
        print(f"詞彙量：{len(vocab)}")

        return train_loader, val_loader, vocab, label_map, class_weights

    def _train_single_model(
        self,
        dataset: str,
        model_type: str,
        num_layers: int,
        train_loader,
        val_loader,
        vocab,
        class_weights,
        config: Dict[str, Any],
        bidirectional: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        訓練單個模型

        Args:
            dataset: 數據集名稱
            model_type: 模型類型
            num_layers: 層數
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            vocab: 詞彙表
            class_weights: 類別權重
            config: 配置
            bidirectional: 是否雙向

        Returns:
            訓練結果
        """
        # 載入詞向量（如果啟用）
        pretrained_embeddings = self._load_embeddings(vocab, config)

        # 建立模型
        model = self._create_model(
            model_type=model_type,
            vocab_size=len(vocab),
            num_layers=num_layers,
            config=config,
            pretrained_embeddings=pretrained_embeddings,
            bidirectional=bidirectional
        )

        if model is None:
            print(f"[錯誤] 無法建立模型：{model_type}")
            return None

        model = model.to(self.device)
        total_params, trainable_params = model.get_num_params()
        print(f"模型參數量：{total_params:,}（可訓練：{trainable_params:,}）")

        # 訓練配置
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-3))

        # 訓練
        num_epochs = config.get('num_epochs', 30)
        best_f1 = 0
        best_epoch = 0
        patience_counter = 0
        patience = 5

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 訓練一個 epoch
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)

            # 評估
            val_loss, val_acc, val_precision, val_recall, val_f1 = self._evaluate(
                model, val_loader, criterion
            )

            # Early stopping
            if val_f1 > best_f1 + 0.001:
                best_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # 印出進度
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{num_epochs}: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        training_time = time.time() - start_time

        # 返回結果
        result = {
            'experiment_id': f"{dataset}_{model_type}_{num_layers}layers",
            'dataset': dataset,
            'model_type': model_type,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'best_epoch': best_epoch,
            'final_metrics': {
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'macro_f1': val_f1
            },
            'training_time_seconds': training_time,
            'model_info': {
                'total_params': total_params,
                'trainable_params': trainable_params
            }
        }

        print(f"\n最終結果：F1={val_f1:.4f}, Acc={val_acc:.4f}")

        return result

    def _load_embeddings(self, vocab, config: Dict[str, Any]) -> Optional[torch.Tensor]:
        """載入詞向量"""
        if not config.get('use_pretrained_embeddings', False):
            return None

        try:
            print("載入 GloVe 詞向量...")
            glove_dir = BASE_DIR / "data" / "embeddings"
            glove_embedding = load_glove_embeddings(
                embedding_dim=300,
                data_dir=str(glove_dir),
                use_cache=True
            )

            vocab_words = get_vocab_words(vocab)
            embedding_matrix, _ = glove_embedding.get_embedding_matrix(
                vocab=vocab_words,
                oov_strategy=config.get('oov_strategy', 'random'),
                seed=42
            )

            return torch.FloatTensor(embedding_matrix)

        except Exception as e:
            print(f"警告：無法載入詞向量：{e}")
            return None

    def _create_model(
        self,
        model_type: str,
        vocab_size: int,
        num_layers: int,
        config: Dict[str, Any],
        pretrained_embeddings: Optional[torch.Tensor] = None,
        bidirectional: bool = True
    ):
        """建立模型"""
        embedding_dim = config.get('embedding_dim', 300)
        hidden_size = config.get('hidden_size', 128)
        dropout = config.get('dropout', 0.3)
        freeze_embeddings = config.get('freeze_embeddings', False)

        if model_type == 'baseline_no_attention':
            return BaselineNoAttention(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=3,
                dropout=dropout,
                pretrained_embeddings=pretrained_embeddings,
                freeze_embeddings=freeze_embeddings
            )

        elif model_type in ['prefusion', 'pre_fusion']:
            return PreFusionModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_lstm_layers=num_layers,
                num_classes=3,
                dropout=dropout,
                pretrained_embeddings=pretrained_embeddings,
                freeze_embeddings=freeze_embeddings,
                bidirectional=bidirectional
            )

        elif model_type in ['postfusion', 'post_fusion']:
            return PostFusionModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_lstm_layers=num_layers,
                num_classes=3,
                dropout=dropout,
                pretrained_embeddings=pretrained_embeddings,
                freeze_embeddings=freeze_embeddings,
                bidirectional=bidirectional
            )

        else:
            print(f"[錯誤] 不支援的模型類型：{model_type}")
            return None

    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """訓練一個 epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            aspect_mask = batch['aspect_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, aspect_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    def _evaluate(self, model, val_loader, criterion):
        """評估模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                aspect_mask = batch['aspect_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits, _ = model(input_ids, aspect_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        return avg_loss, acc, precision, recall, f1
