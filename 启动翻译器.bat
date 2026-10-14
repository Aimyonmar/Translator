@echo off
chcp 936 >nul
cd /d "%~dp0"

rem Python 环境在本项目 .runtime 下（不存在时自动回退系统 Python）
set "PY=%~dp0.runtime\venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=pythonw"

set HF_ENDPOINT=https://hf-mirror.com
rem 模型缓存默认在项目内 .runtime\hf；若已在别处缓存，可设 HF_HOME 指向它
set "PIP_CACHE_DIR=%~dp0.runtime\pipcache"

echo.
echo   正在启动 VTuber 实时翻译器，请稍候（首次约 5-10 秒）...
echo.

start "" "%PY%" main.py

ping -n 9 127.0.0.1 >nul
tasklist /fi "imagename eq pythonw.exe" 2>nul | find /i "pythonw.exe" >nul
if errorlevel 1 (
    echo   ============================================
    echo    启动失败，程序没有跑起来。
    echo    请双击「调试模式启动.bat」看具体报错。
    echo   ============================================
    echo.
    if exist run.log start "" notepad run.log
    pause
) else (
    echo   ============================================
    echo    已启动！
    echo    1. 控制面板已打开，点上面的「开始翻译」
    echo    2. 字幕窗在屏幕底部
    echo    3. 屏幕右下角会弹一条提示气泡
    echo   ============================================
    echo.
    echo   看不到窗口的话：右键屏幕右下角托盘图标，
    echo   选「显示控制面板」。
    echo.
    ping -n 6 127.0.0.1 >nul
)
