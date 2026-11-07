@echo off
chcp 936 >nul
cd /d "%~dp0"
title 安装依赖 - VTuber 实时翻译器

rem 依赖全部装在项目内 .runtime\venv，不污染系统 Python，也不占 C 盘。
set "VENV=%~dp0.runtime\venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP_CACHE_DIR=%~dp0.runtime\pipcache"
set "MIRROR=https://mirrors.aliyun.com/pypi/simple/"

echo ============================================
echo   安装依赖（只需做一次，通常 3~8 分钟）
echo   位置：项目内 .runtime\venv
echo   需要：Python 3.10 ~ 3.13（64 位）
echo ============================================
echo.

if exist "%PY%" (
    echo   检测到已有虚拟环境，直接检查/补齐依赖。
    goto :install
)

rem ---------- 1) 找一个可用的 Python（优先 3.12/3.11/3.10/3.13） ----------
set "BASE="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined BASE (
        py -%%V -c "import sys" >nul 2>&1 && set "BASE=py -%%V"
    )
)
if not defined BASE (
    python -c "import sys;raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>&1 && set "BASE=python"
)
if not defined BASE goto :nopython

echo   使用的 Python：
%BASE% -V
echo.
echo   正在创建虚拟环境 .runtime\venv（约 10 秒）…
%BASE% -m venv "%VENV%"
if not exist "%PY%" goto :venvfail
echo.

:install
echo   正在安装依赖（走阿里云镜像，可断点重试）…
echo.
"%PY%" -m pip install --upgrade pip -i %MIRROR% >nul 2>&1
"%PY%" -m pip install -r requirements.txt -i %MIRROR%
if errorlevel 1 (
    echo.
    echo   镜像安装失败，改用官方源再试一次…
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :pipfail
)

echo.
echo   校验关键依赖…
"%PY%" -c "import PySide6, faster_whisper, av, sounddevice; print('  依赖校验通过')"
if errorlevel 1 goto :pipfail

rem ---------- 2) NVIDIA 显卡：按需补 cuBLAS 运行库 ----------
rem ctranslate2 的 Windows 版不自带 cublas，缺了 GPU 推理会直接失败。
rem 软件拿到异常后会静默退回 CPU，表现为「离线转写很久不出结果」——
rem 所以这里主动补装，而不是写进 requirements.txt 让没显卡的人也下 550 MB。
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :nogpu

echo.
echo   检测到 NVIDIA 显卡，检查 CUDA 运行库…
"%PY%" -c "import os,glob,sys; d=os.path.join(sys.prefix,'Lib','site-packages','ctranslate2'); sys.exit(0 if glob.glob(os.path.join(d,'cublas64*.dll')) else 1)"
if errorlevel 1 (
    echo   缺少 cuBLAS，正在安装 nvidia-cublas-cu12（约 550 MB）…
    "%PY%" -m pip install nvidia-cublas-cu12 -i %MIRROR%
    if errorlevel 1 (
        echo   镜像安装失败，改用官方源再试一次…
        "%PY%" -m pip install nvidia-cublas-cu12
    )
)
"%PY%" enable_gpu.py --copy-only
if errorlevel 1 goto :gpufail
echo   CUDA 运行库就绪，下次启动会直接尝试 GPU。
goto :done

:nogpu
echo.
echo   没有检测到 NVIDIA 显卡，跳过 CUDA 运行库（CPU 可用，只是慢些）。

:done
echo.
echo ============================================
echo   依赖装好了。下一步：
echo   1. 双击「启动翻译器.bat」
echo   2. 首次启动会自动下载识别模型
echo      （默认 small 约 480 MB，只下一次）
echo ============================================
pause
exit /b 0

:gpufail
echo.
echo ============================================
echo   [!] 检测到 NVIDIA 显卡，但 CUDA 运行库没准备好。
echo.
echo   影响：GPU 识别会报找不到 cublas64_12.dll，
echo   软件会退回 CPU —— 实时字幕变慢，离线转写看起来像卡住。
echo.
echo   联网后手动补救：
echo     .runtime\venv\Scripts\python.exe -m pip install nvidia-cublas-cu12
echo     .runtime\venv\Scripts\python.exe enable_gpu.py
echo.
echo   暂时不想处理也行：CPU 模式功能完整，只是慢一些。
echo ============================================
echo.
pause
exit /b 1

:nopython
echo.
echo ============================================
echo   [!] 没有找到可用的 Python。
echo.
echo   请先安装 Python 3.10 ~ 3.13 的 64 位版本：
echo     https://www.python.org/downloads/windows/
echo   安装时务必勾选 "Add python.exe to PATH"。
echo.
echo   装好后重新双击本文件；已经有 3.14 也能装，
echo   但 PySide6 / ctranslate2 目前还没有 3.14 的预编译包。
echo ============================================
echo.
pause
exit /b 1

:venvfail
echo.
echo ============================================
echo   [!] 创建虚拟环境失败。
echo   常见原因：Python 安装不完整（缺 venv 模块）、
echo   磁盘空间不足、或 .runtime 目录被占用。
echo   建议重装 Python（勾选 pip 与 venv）后重试。
echo ============================================
echo.
pause
exit /b 1

:pipfail
echo.
echo ============================================
echo   [!] 依赖安装失败。
echo   1. 检查网络/代理：能打开 https://mirrors.aliyun.com 吗？
echo   2. 有代理时先执行：set HTTPS_PROXY=http://127.0.0.1:7890
echo   3. 磁盘至少留 3 GB 空间（依赖约 2.5 GB）。
echo   4. 修好后重新双击本文件，已下载的包不会重复下。
echo ============================================
echo.
pause
exit /b 1
