# GPU 設置指南

## ✅ 當前狀態

你的系統配置：
- **GPU**: NVIDIA GeForce RTX 3090 (24GB)
- **CUDA**: 12.6
- **虛擬環境 PyTorch**: 2.8.0+cu126 ✅ (支持 CUDA)
- **系統 Python PyTorch**: 2.8.0+cpu ❌ (不支持 CUDA)

## ⚠️ 問題

如果直接運行 `python scripts/train_xxx.py`，會使用系統 Python（CPU版本），**不會使用 GPU**。

## ✅ 解決方案

### 方法一：使用批次腳本（最簡單）⭐

直接雙擊或運行：

```bash
# 完整實驗流程（使用 GPU）
run_with_gpu.bat

# 或者只訓練模型
train_gpu.bat
```

### 方法二：激活虛擬環境

```bash
# 激活虛擬環境
env\Scripts\activate

# 驗證 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"

# 執行訓練
python scripts/train_all_datasets.py --model all
```

### 方法三：直接指定虛擬環境 Python

```bash
# 不激活虛擬環境，直接使用
env\Scripts\python.exe scripts\train_all_datasets.py --model all
env\Scripts\python.exe scripts\run_full_experiment.py --skip-cleaning
```

## 🔍 驗證 GPU 是否啟用

### 快速檢查

```bash
# 在虛擬環境中
env\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

應該顯示：
```
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

### 訓練時檢查

訓練腳本開始時會顯示：
```
使用裝置: cuda
```

如果顯示 `cpu`，說明沒有正確使用虛擬環境。

### 監控 GPU 使用

在訓練過程中，另開一個終端執行：

```bash
# 實時監控 GPU
nvidia-smi -l 1
```

應該看到：
- GPU-Util: 80-100%（訓練時）
- Memory-Usage: 逐漸增加

## 📊 性能對比

使用 RTX 3090 訓練：

| 設備 | 每個 epoch 時間 | 完整訓練時間 |
|------|----------------|-------------|
| **CPU** | ~10-15 分鐘 | ~5-8 小時 |
| **RTX 3090** | ~30-60 秒 | ~30-60 分鐘 |

**加速比：約 10-20 倍** 🚀

## 🛠️ 常見問題

### Q: 為什麼顯示 "使用裝置: cpu"？

A: 你沒有使用虛擬環境的 Python。請使用：
```bash
env\Scripts\python.exe scripts/train_xxx.py
```

### Q: 如何強制使用 GPU？

A: PyTorch 會自動檢測。只要：
1. 使用虛擬環境的 Python
2. `torch.cuda.is_available()` 返回 True
3. 訓練腳本會自動使用 GPU

### Q: 訓練時 GPU 使用率很低？

A: 可能的原因：
1. Batch size 太小 → 增大到 64 或 128
2. 數據加載瓶頸 → 增加 DataLoader workers
3. 模型太小 → 正常情況

### Q: 記憶體不足 (OOM) 錯誤？

A: 減小 batch size：
```python
# configs/experiment_config.py
BASELINE_CONFIG = {
    "batch_size": 16,  # 從 32 減到 16
    ...
}
```

## 🚀 推薦設置

針對 RTX 3090 (24GB) 的最佳配置：

```python
# configs/experiment_config.py

BASELINE_CONFIG = {
    "batch_size": 64,      # 利用大顯存
    "num_epochs": 50,      # GPU 快，可以訓練更多輪
    "learning_rate": 2e-5,
    ...
}
```

## 📝 快速啟動命令

```bash
# 1. 檢查 GPU（應該顯示 True）
env\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# 2. 查看配置
env\Scripts\python.exe configs\experiment_config.py

# 3. 開始訓練（使用 GPU）
train_gpu.bat

# 或者
env\Scripts\python.exe scripts\train_all_datasets.py --model all
```

## 💡 提示

1. **總是使用虛擬環境**：`env\Scripts\python.exe` 或激活後使用 `python`
2. **批次腳本最簡單**：直接運行 `train_gpu.bat`
3. **監控 GPU**：訓練時開 `nvidia-smi -l 1` 查看使用情況
4. **調整 batch size**：根據 GPU 記憶體調整（RTX 3090 可以用 64-128）

## ⚡ 開始訓練

最簡單的方式：

```bash
# 雙擊運行
train_gpu.bat
```

或命令行：

```bash
env\Scripts\python.exe scripts\run_full_experiment.py --skip-cleaning
```

享受 GPU 加速帶來的效率提升！🚀
