@echo off
chcp 936 >nul
cd /d "%~dp0"

set "PY=%~dp0.runtime\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set HF_ENDPOINT=https://hf-mirror.com
set "PIP_CACHE_DIR=%~dp0.runtime\pipcache"

echo   调试模式：这个黑窗口会一直显示程序报错信息，请不要关它。
echo.
"%PY%" main.py
echo.
echo   程序已退出。上面若有红色报错，截图发给我即可。按任意键关闭。
pause >nul
