@echo off
REM 使用虛擬環境的 Python 執行訓練（啟用 GPU）

echo ========================================
echo 檢查 GPU 狀態
echo ========================================
env\Scripts\python.exe -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo 開始訓練
echo ========================================
echo.

REM 執行完整實驗（跳過數據清理）
env\Scripts\python.exe scripts\run_full_experiment.py --skip-cleaning

pause
