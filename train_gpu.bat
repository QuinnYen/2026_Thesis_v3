@echo off
REM 快速啟動腳本 - 使用 GPU 訓練所有數據集

echo ========================================
echo GPU 訓練腳本
echo ========================================
echo.

REM 檢查 GPU
echo 檢查 GPU 狀態...
env\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM 顯示配置
echo 當前配置:
env\Scripts\python.exe configs\experiment_config.py
echo.

REM 詢問確認
set /p confirm="是否開始訓練? (y/n): "
if /i not "%confirm%"=="y" (
    echo 已取消
    pause
    exit /b
)

REM 執行訓練
echo.
echo 開始訓練...
env\Scripts\python.exe scripts\train_all_datasets.py --model all

echo.
echo ========================================
echo 訓練完成!
echo ========================================
pause
