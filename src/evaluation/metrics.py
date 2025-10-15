"""
ABSA 評估指標模組

此模組提供各種評估指標的計算，包含：
- 多分類 AUC (One-vs-Rest)
- ROC 曲線相關計算
- 其他自定義指標
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings


def calculate_multiclass_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    計算多分類 AUC (One-vs-Rest)

    參數:
        y_true (np.ndarray): 真實標籤，shape (n_samples,)
        y_prob (np.ndarray): 預測機率，shape (n_samples, n_classes)
        class_names (List[str], optional): 類別名稱列表

    返回:
        Dict[str, float]: 包含各類別 AUC 和 macro-average AUC 的字典
    """
    n_classes = y_prob.shape[1]

    # 如果沒有提供類別名稱，使用預設名稱
    if class_names is None:
        class_names = [f'class_{i}' for i in range(n_classes)]

    # 將標籤二值化 (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 如果只有兩個類別，label_binarize 會返回單列，需要調整
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    auc_scores = {}

    # 計算每個類別的 AUC
    for i, class_name in enumerate(class_names):
        try:
            # 檢查該類別是否存在於 y_true 中
            if i in y_true:
                score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                auc_scores[f'auc_{class_name}'] = score
            else:
                # 如果該類別不存在於測試集中，設為 NaN
                auc_scores[f'auc_{class_name}'] = np.nan
                warnings.warn(f"類別 '{class_name}' 不存在於測試集中，AUC 設為 NaN")
        except Exception as e:
            auc_scores[f'auc_{class_name}'] = np.nan
            warnings.warn(f"計算類別 '{class_name}' 的 AUC 時發生錯誤: {e}")

    # 計算 macro-average AUC (忽略 NaN)
    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    if valid_aucs:
        auc_scores['auc_macro'] = np.mean(valid_aucs)
    else:
        auc_scores['auc_macro'] = np.nan

    return auc_scores


def calculate_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    計算每個類別的 ROC 曲線數據

    參數:
        y_true (np.ndarray): 真實標籤，shape (n_samples,)
        y_prob (np.ndarray): 預測機率，shape (n_samples, n_classes)
        class_names (List[str], optional): 類別名稱列表

    返回:
        Dict[str, Dict[str, np.ndarray]]: 包含每個類別的 FPR, TPR, AUC
            格式: {
                'class_name': {
                    'fpr': np.ndarray,
                    'tpr': np.ndarray,
                    'auc': float
                }
            }
    """
    n_classes = y_prob.shape[1]

    # 如果沒有提供類別名稱，使用預設名稱
    if class_names is None:
        class_names = [f'class_{i}' for i in range(n_classes)]

    # 將標籤二值化 (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 如果只有兩個類別，label_binarize 會返回單列，需要調整
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    roc_data = {}

    # 計算每個類別的 ROC 曲線
    for i, class_name in enumerate(class_names):
        try:
            if i in y_true:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)

                roc_data[class_name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                }
            else:
                warnings.warn(f"類別 '{class_name}' 不存在於測試集中，跳過 ROC 計算")
        except Exception as e:
            warnings.warn(f"計算類別 '{class_name}' 的 ROC 曲線時發生錯誤: {e}")

    # 計算 macro-average ROC
    if roc_data:
        # 收集所有的 FPR 點
        all_fpr = np.unique(np.concatenate([roc_data[cn]['fpr'] for cn in roc_data.keys()]))

        # 在所有 FPR 點上插值 TPR
        mean_tpr = np.zeros_like(all_fpr)
        for class_name in roc_data.keys():
            mean_tpr += np.interp(all_fpr, roc_data[class_name]['fpr'], roc_data[class_name]['tpr'])

        # 計算平均值
        mean_tpr /= len(roc_data)

        # 添加起始點和結束點
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0

        # 計算 macro-average AUC
        macro_auc = auc(all_fpr, mean_tpr)

        roc_data['macro'] = {
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'auc': macro_auc
        }

    return roc_data


def calculate_micro_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    計算 micro-average AUC

    參數:
        y_true (np.ndarray): 真實標籤，shape (n_samples,)
        y_prob (np.ndarray): 預測機率，shape (n_samples, n_classes)

    返回:
        Tuple[float, np.ndarray, np.ndarray]: (micro_auc, fpr, tpr)
    """
    n_classes = y_prob.shape[1]

    # 將標籤二值化
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # 如果只有兩個類別，label_binarize 會返回單列，需要調整
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # 將所有類別的預測攤平
    y_true_flat = y_true_bin.ravel()
    y_prob_flat = y_prob.ravel()

    # 計算 micro-average ROC 曲線
    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    micro_auc = auc(fpr, tpr)

    return micro_auc, fpr, tpr


def verify_auc_calculation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None
) -> bool:
    """
    驗證 AUC 計算結果與 sklearn 的一致性

    參數:
        y_true (np.ndarray): 真實標籤
        y_prob (np.ndarray): 預測機率
        class_names (List[str], optional): 類別名稱

    返回:
        bool: 驗證是否通過
    """
    # 計算我們的 AUC
    our_auc = calculate_multiclass_auc(y_true, y_prob, class_names)

    # 使用 sklearn 計算 AUC (One-vs-Rest, macro-average)
    try:
        sklearn_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class='ovr',
            average='macro'
        )

        # 比較結果（允許小誤差）
        diff = abs(our_auc['auc_macro'] - sklearn_auc)
        tolerance = 1e-6

        if diff < tolerance:
            print(f"[OK] AUC 驗證通過！")
            print(f"  我們的計算: {our_auc['auc_macro']:.6f}")
            print(f"  sklearn 計算: {sklearn_auc:.6f}")
            print(f"  差異: {diff:.10f}")
            return True
        else:
            print(f"[X] AUC 驗證失敗！")
            print(f"  我們的計算: {our_auc['auc_macro']:.6f}")
            print(f"  sklearn 計算: {sklearn_auc:.6f}")
            print(f"  差異: {diff:.10f}")
            return False
    except Exception as e:
        print(f"[X] 驗證過程發生錯誤: {e}")
        return False


# 測試程式碼
if __name__ == "__main__":
    """
    測試 AUC 計算功能
    """
    print("="*60)
    print("測試 AUC 指標計算")
    print("="*60)

    # 建立測試資料
    np.random.seed(42)
    n_samples = 100
    n_classes = 3

    # 模擬真實標籤
    y_true = np.random.randint(0, n_classes, n_samples)

    # 模擬預測機率（確保每行和為1）
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    class_names = ['negative', 'neutral', 'positive']

    # 測試 1: 計算 multiclass AUC
    print("\n測試 1: 計算 Multiclass AUC")
    print("-" * 60)
    auc_scores = calculate_multiclass_auc(y_true, y_prob, class_names)
    for key, value in auc_scores.items():
        print(f"{key}: {value:.4f}")

    # 測試 2: 計算 ROC 曲線
    print("\n測試 2: 計算 ROC 曲線")
    print("-" * 60)
    roc_data = calculate_roc_curves(y_true, y_prob, class_names)
    for class_name, data in roc_data.items():
        print(f"{class_name}: AUC = {data['auc']:.4f}, FPR 點數 = {len(data['fpr'])}")

    # 測試 3: 計算 micro-average AUC
    print("\n測試 3: 計算 Micro-average AUC")
    print("-" * 60)
    micro_auc, fpr, tpr = calculate_micro_auc(y_true, y_prob)
    print(f"Micro-average AUC: {micro_auc:.4f}")
    print(f"FPR 點數: {len(fpr)}")

    # 測試 4: 驗證 AUC 計算
    print("\n測試 4: 驗證 AUC 計算")
    print("-" * 60)
    is_valid = verify_auc_calculation(y_true, y_prob, class_names)

    # 測試 5: 邊界情況測試
    print("\n測試 5: 邊界情況 - 只有兩個類別")
    print("-" * 60)
    y_true_binary = np.random.randint(0, 2, n_samples)
    y_prob_binary = np.random.rand(n_samples, 2)
    y_prob_binary = y_prob_binary / y_prob_binary.sum(axis=1, keepdims=True)

    auc_binary = calculate_multiclass_auc(y_true_binary, y_prob_binary, ['negative', 'positive'])
    print(f"二分類 Macro-AUC: {auc_binary['auc_macro']:.4f}")

    # 測試 6: 完美預測
    print("\n測試 6: 完美預測情況")
    print("-" * 60)
    y_true_perfect = np.array([0, 1, 2, 0, 1, 2])
    y_prob_perfect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    auc_perfect = calculate_multiclass_auc(y_true_perfect, y_prob_perfect, class_names)
    print(f"完美預測 Macro-AUC: {auc_perfect['auc_macro']:.4f}")
    print("(應該接近 1.0)")

    print("\n" + "="*60)
    print("所有測試完成！" if is_valid else "驗證失敗，請檢查實作")
    print("="*60)
