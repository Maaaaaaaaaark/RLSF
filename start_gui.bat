@echo off
REM IMPROVED RLSF GUI 启动脚本 (Windows)
REM 自动激活虚拟环境并启动 Streamlit 界面

echo ========================================
echo   IMPROVED RLSF Training GUI
echo ========================================
echo.

REM 检查虚拟环境
if exist "..\myenv\Scripts\activate.bat" (
    echo [1/3] 激活虚拟环境...
    call ..\myenv\Scripts\activate.bat
) else (
    echo 警告: 未找到虚拟环境 myenv
    echo 请确保已创建虚拟环境或手动激活
    echo.
)

REM 检查依赖
echo [2/3] 检查依赖...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit 未安装，正在安装...
    pip install streamlit plotly
)

REM 启动 GUI
echo [3/3] 启动界面...
echo.
echo 界面将在浏览器中自动打开
echo 默认地址: http://localhost:8501
echo.
echo 按 Ctrl+C 可停止服务器
echo ========================================
echo.

streamlit run gui_app.py

pause

