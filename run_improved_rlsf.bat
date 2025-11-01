@echo off
REM RLSF改进算法一键运行脚本 (Windows批处理版本)
REM 使用方法: run_improved_rlsf.bat [test|train|ablation|visualize]

echo ============================================================
echo RLSF改进算法运行脚本
echo ============================================================
echo.

REM 检查参数
if "%1"=="" (
    echo 使用方法: run_improved_rlsf.bat [test^|train^|ablation^|visualize]
    echo.
    echo 可用命令:
    echo   test       - 运行功能测试
    echo   train      - 运行完整训练
    echo   ablation   - 运行消融研究
    echo   visualize  - 可视化训练结果
    echo.
    exit /b 1
)

REM 激活虚拟环境（如果存在）
if exist "myenv\Scripts\activate.bat" (
    echo [1/3] 激活虚拟环境...
    call myenv\Scripts\activate.bat
) else (
    echo [1/3] 未找到虚拟环境，使用系统Python
)

REM 检查依赖
echo [2/3] 检查依赖...
python -c "import torch, numpy, matplotlib, scipy" 2>nul
if errorlevel 1 (
    echo ❌ 缺少必要的依赖包
    echo 请运行: pip install torch numpy matplotlib scipy wandb safety-gymnasium
    exit /b 1
)
echo ✅ 依赖检查通过

REM 执行命令
echo [3/3] 执行命令: %1
echo.

if "%1"=="test" (
    echo 🧪 运行功能测试...
    python test_improvements.py
    goto :end
)

if "%1"=="train" (
    echo 🚀 运行完整训练...
    python run_improved_rlsf.py --mode single --env_name SafetyPointCircle1-v0 --num_training_step 100000 --wandb_log False
    goto :end
)

if "%1"=="ablation" (
    echo 🔬 运行消融研究...
    python run_improved_rlsf.py --mode ablation --environments SafetyPointCircle1-v0 --seeds 0 1 2 --num_training_step 50000
    goto :end
)

if "%1"=="visualize" (
    echo 📊 可视化训练结果...
    if "%2"=="" (
        echo 请指定日志目录: run_improved_rlsf.bat visualize [log_dir]
        exit /b 1
    )
    python visualize_training.py --log_dir %2 --output_dir ./visualizations
    goto :end
)

echo ❌ 未知命令: %1
echo 可用命令: test, train, ablation, visualize
exit /b 1

:end
echo.
echo ============================================================
echo 完成！
echo ============================================================

