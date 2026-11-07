@echo off
chcp 936 >nul
cd /d "%~dp0"
title VTuber 实时翻译器

set "PY=%~dp0.runtime\venv\Scripts\pythonw.exe"
set "PYCON=%~dp0.runtime\venv\Scripts\python.exe"

rem 模型缓存固定在项目内 .runtime\hf：系统里可能残留别的项目设的 HF_HOME，
rem 这里显式覆盖，保证 准备模型.py / asr / 面板看到的是同一份缓存
set "HF_HOME=%~dp0.runtime\hf"

rem 依赖装在项目内 .runtime\venv；没有就先引导去安装，避免 pythonw 静默失败
if not exist "%PYCON%" (
    echo   ============================================
    echo    还没安装依赖。
    echo    请先双击「安装依赖.bat」（只需做一次，3~8 分钟）。
    echo   ============================================
    echo.
    pause
    exit /b 1
)

rem 首次启动：模型没缓存过就先下载，进度直接显示在这个窗口里
rem 缓存判定交给 准备模型.py：turbo / kotoba 的仓库名不是 Systran，
rem 写死 models--Systran--faster-whisper-* 会把已缓存的它们误判成「首次启动」
"%PYCON%" "%~dp0准备模型.py" --check >nul 2>&1
if errorlevel 1 (
    echo   ============================================
    echo    首次启动：需要下载语音识别模型
    echo    只下载一次，之后完全离线可用
    echo   ============================================
    echo.
    "%PYCON%" "%~dp0准备模型.py"
    if errorlevel 1 (
        echo.
        echo   模型没有下载成功，先解决网络问题再重试：
        echo   重新双击本文件即可，会从断点继续下载。
        echo.
        pause
        exit /b 1
    )
    echo.
)

set HF_ENDPOINT=https://hf-mirror.com
rem 缓存目录已在上方固定为项目内 .runtime\hf，HF_ENDPOINT 只负责下载源
set "PIP_CACHE_DIR=%~dp0.runtime\pipcache"

echo.
echo   正在启动 VTuber 实时翻译器，请稍候（约 3~10 秒）...
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
