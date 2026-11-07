# -*- coding: utf-8 -*-
"""首次运行前把 Whisper 模型下载到项目内缓存，并显示可见进度。

为什么要单独跑这一步：程序本体用 pythonw 启动（没有控制台窗口），
faster-whisper 的下载进度条会被吞掉，用户只看到「正在加载模型…」卡着。
本脚本用控制台 python 运行，进度、体积、耗时都看得见。

用法（在项目根目录）：
    .runtime\\venv\\Scripts\\python.exe 准备模型.py
    .runtime\\venv\\Scripts\\python.exe 准备模型.py medium
    .runtime\\venv\\Scripts\\python.exe 准备模型.py --check

不带参数时读 config.json 的 model_size；没有配置就用默认 small。
已缓存时直接跳过，不会重复下载。
`--check` 只判断当前配置的模型是否已完整缓存：已缓存返回 0，否则返回 1，
供「启动翻译器.bat」决定要不要弹下载 —— 缓存目录名由 core.model_store 统一决定，
turbo / kotoba 的仓库名不是 Systran，写死前缀会误判。
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")

# 控制台编码（chcp 936）偶发编不出的字符不该让脚本崩掉，宁可显示成 ?
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# 模型仓库体积（MB）与缓存判定统一由 core.model_store 提供：面板、asr、
# 本脚本共用一套，避免两处表各写一份、越走越偏
from core.model_store import EST_SIZE_MB as SIZE_MB, repo_id, snapshot_dir


def _pick_model() -> str:
    # 跳过 --check 这类开关，别把参数名当成型号
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            return arg.strip()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            size = (json.load(f) or {}).get("model_size")
        if size:
            return str(size)
    except Exception:
        pass
    return "small"


def _fail(msg: str) -> int:
    print()
    print("=" * 58)
    print(msg)
    print("=" * 58)
    return 1


def _cached_path(size: str) -> str | None:
    """按目录结构判断模型是否已经完整下载。

    不用 snapshot_download(local_files_only=True) 判断：镜像下载没写 refs 时它会
    误报「未缓存」，于是每次启动都重新联网，看起来像卡住。
    """
    return snapshot_dir(size)


def main() -> int:
    inherited = os.environ.get("HF_HOME")
    # 必须直接覆盖：环境里可能残留别的项目设的 HF_HOME，setdefault 只会沿用那份旧路径，
    # 结果模型下到旧目录、启动时按项目目录判定「没缓存」，每次都要重下。
    os.environ["HF_HOME"] = os.path.join(ROOT, ".runtime", "hf")
    # 国内直连 huggingface.co 多半不通，默认走镜像；想用官方源就自己设 HF_ENDPOINT
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Windows 没开发者模式时 hub 会警告软链不可用，对使用无影响，关掉免得吓人
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    size = _pick_model()
    # --check：只回答「当前配置的模型是否已完整缓存」，不打印、不联网。
    # 启动脚本用它判断要不要走下载流程；仓库名/缓存目录都交给 model_store，
    # 避免在这里硬拼 Systran 前缀（turbo、kotoba 不在 Systran 名下）。
    if "--check" in sys.argv:
        cached = _cached_path(size)
        if cached:
            print(cached)
        return 0 if cached else 1

    repo = repo_id(size)
    mb = SIZE_MB.get(size)
    cache = os.path.join(os.environ["HF_HOME"], "hub")

    print(f"模型：{repo}" + (f"（约 {mb} MB）" if mb else ""))
    print(f"缓存：{cache}")
    if inherited and os.path.abspath(inherited) != os.path.abspath(os.environ["HF_HOME"]):
        print(f"（已忽略环境里的 HF_HOME={inherited}，统一用项目内缓存）")
    print(f"下载源：{os.environ['HF_ENDPOINT']}")
    if mb:
        print(f"参考耗时：10 MB/s 约 {max(1, round(mb / 10 / 60))} 分钟，"
              f"1~2 MB/s 约 {max(1, round(mb / 1.5 / 60))} 分钟。")
    print()

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return _fail(
            "缺少 huggingface_hub，无法下载模型。\n"
            "请先双击「安装依赖.bat」，或手动执行：\n"
            "    .runtime\\venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )

    # 先看缓存：已经下过就直接结束，不联网、不重复下载
    cached = _cached_path(size)
    if cached:
        print(f"已在本地缓存，无需下载：{cached}")
        return 0

    print("开始下载（进度条为已下载文件数与总量，网络中断可以重新运行本脚本续传）…")
    print()
    t0 = time.time()
    try:
        path = snapshot_download(repo)
    except KeyboardInterrupt:
        return _fail("已取消下载。下次运行本脚本会从断点继续。")
    except Exception as exc:
        return _fail(
            "模型下载失败，程序暂时无法识别语音。\n"
            f"错误信息：{exc}\n\n"
            "可以这样排查：\n"
            "1. 确认能打开 https://hf-mirror.com（浏览器打不开就是网络/防火墙问题）；\n"
            "2. 有代理就设一下再重试：set HTTPS_PROXY=http://127.0.0.1:7890；\n"
            "3. 想换下载源：set HF_ENDPOINT=https://huggingface.co（需要能直连）；\n"
            "4. 公司网络拦住时，用手机热点重试一次。\n"
            "修好网络后重新双击「启动翻译器.bat」，会自动再尝试下载。"
        )

    cost = time.time() - t0
    print()
    print(f"下载完成，用时 {cost:.0f} 秒，之后离线可用。")
    print(f"位置：{path}")
    print("现在可以双击「启动翻译器.bat」开始使用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
