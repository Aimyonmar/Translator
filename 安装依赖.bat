@echo off
chcp 936 >nul
cd /d "%~dp0"

set "PY=%~dp0.runtime\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "PIP_CACHE_DIR=%~dp0.runtime\pipcache"

echo   正在安装依赖（国内走阿里云镜像），大约几分钟，请耐心等待...
echo.
"%PY%" -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
echo.
echo   安装结束。按任意键关闭，之后双击「启动翻译器.bat」即可。
pause >nul
