# -*- coding: utf-8 -*-
"""Whisper 模型缓存：查状态、算体积、下载、卸载。

判定一律看磁盘上的真实文件（snapshots/<rev>/model.bin 等），不看 refs、
不看环境变量、不看内存里的标志位：下载中断时目录仍然存在，只看目录会把
「只下了一半」误判成「已下载」，接着被 asr 切成离线模式，加载必然失败。
"""
from __future__ import annotations

import os
import shutil
from typing import Callable, Optional

REPO_PREFIX = "models--Systran--faster-whisper-"

# 首次下载的估算体积（MB），用于在 UI 上先告诉用户要花多少空间
EST_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 484,
    "medium": 1530,
    "large-v2": 3090,
    "large-v3": 3090,
}

# 缺任一文件都不算可用（faster-whisper 加载必需）
_REQUIRED = ("model.bin", "config.json", "tokenizer.json")


def cache_root() -> str:
    """与 asr.py 完全一致：优先 HF_HOME，其次用户目录默认缓存。"""
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")


def model_dir(size: str) -> str:
    return os.path.join(cache_root(), "hub", REPO_PREFIX + size)


def snapshot_dir(size: str) -> Optional[str]:
    """返回可用的快照目录；没有或不完整返回 None。"""
    base = os.path.join(model_dir(size), "snapshots")
    try:
        names = sorted(os.listdir(base), reverse=True)
    except OSError:
        return None
    for name in names:
        path = os.path.join(base, name)
        if all(os.path.exists(os.path.join(path, f)) for f in _REQUIRED):
            return path
    return None


def is_downloaded(size: str) -> bool:
    return snapshot_dir(size) is not None


def is_partial(size: str) -> bool:
    """目录存在但没有完整快照：多半是上次下载中断。"""
    return os.path.isdir(model_dir(size)) and not is_downloaded(size)


def size_bytes(size: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(model_dir(size)):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def total_bytes() -> int:
    """所有 Whisper 模型缓存合计占用的字节数。"""
    hub = os.path.join(cache_root(), "hub")
    try:
        names = [n for n in os.listdir(hub) if n.startswith(REPO_PREFIX)]
    except OSError:
        return 0
    return sum(size_bytes(n[len(REPO_PREFIX):]) for n in names)


def remove(size: str) -> int:
    """删除该模型的整个缓存目录，返回释放的字节数。"""
    root = model_dir(size)
    if not os.path.isdir(root):
        return 0
    freed = size_bytes(size)
    shutil.rmtree(root)
    return freed


def download(size: str, progress: Optional[Callable[[str], None]] = None) -> str:
    """下载指定模型；已完整存在时直接返回，不联网。"""
    if is_downloaded(size):
        return snapshot_dir(size) or ""

    os.environ.setdefault("HF_HOME", cache_root())
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # asr 加载旧模型时会置 HF_HUB_OFFLINE=1；不清掉的话本次下载会直接按离线失败
    os.environ.pop("HF_HUB_OFFLINE", None)

    from huggingface_hub import snapshot_download

    repo = f"Systran/faster-whisper-{size}"
    if progress:
        progress(f"开始下载 {size}（约 {EST_SIZE_MB.get(size, 0)} MB）…")
    snapshot_download(repo)
    if not is_downloaded(size):
        raise RuntimeError("下载结束但缓存文件不完整，请重试或检查磁盘空间")

    # 下载完立刻清掉离线标记，否则本次会话仍会按“没网”加载
    os.environ.pop("HF_HUB_OFFLINE", None)
    if progress:
        progress(f"{size} 下载完成")
    return snapshot_dir(size) or ""
