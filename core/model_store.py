# -*- coding: utf-8 -*-
"""Whisper 模型缓存：查状态、算体积、下载、卸载。

判定一律看磁盘上的真实文件（snapshots/<rev>/model.bin 等），不看 refs、
不看环境变量、不看内存里的标志位：下载中断时目录仍然存在，只看目录会把
「只下了一半」误判成「已下载」，接着被 asr 切成离线模式，加载必然失败。
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Optional

REPO_PREFIX = "models--Systran--faster-whisper-"

# 不是 Systran 发布的型号要单独写仓库名；否则按 Systran/faster-whisper-<size> 拼。
# large-v3-turbo：Systran 没有这个仓库，这里用的正是 faster-whisper 内置别名
# "large-v3-turbo" 指向的那一份，保证「缓存判定」和「实际加载」看到同一个目录。
_REPO_OVERRIDES = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    # 日语专用蒸馏档：官方发布了 CTranslate2 版本，faster-whisper 可直接加载
    "kotoba-whisper-v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
}

# 首次下载的估算体积（MB），用于在 UI 上先告诉用户要花多少空间
EST_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 484,
    "medium": 1530,
    "large-v3-turbo": 1620,
    "kotoba-whisper-v2.0": 1560,
    "large-v2": 3090,
    "large-v3": 3090,
}

# 缺任一文件都不算可用（faster-whisper 加载必需）
_REQUIRED = ("model.bin", "config.json", "tokenizer.json")

# 修复副本：kotoba 原件的 alignment_heads 指向解码器 7~25 层，但它只有 2 层解码器，
# ct2 的 align 不校验层号 → 开字词时间码直接原生崩溃。副本只改了 config.json
# 里的 alignment_heads（层 1 的 20 个头），model.bin 与原快照逐字节相同。
# 目录名特意保留 "kotoba" 便于辨认，所以能力判断必须按真实路径走，不能按名字。
ALIGN_FIXED = {"kotoba-whisper-v2.0": "kotoba-align-fix"}
_FIXED_REQUIRED = ("model.bin", "config.json", "tokenizer.json",
                   "vocabulary.json", "preprocessor_config.json")
# 该模型的解码器层数与每层头数（model.bin 实测），用来校验副本里的层号/头号
_FIXED_DECODER_LAYERS = 2
_FIXED_DECODER_HEADS = 20

_RUNTIME_HF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runtime", "hf")


def cache_root() -> str:
    """缓存目录固定为项目内 .runtime/hf，与 asr.py 一致。

    不读环境里的 HF_HOME：用户环境里可能残留别的项目设的旧路径
    （实测 <旧工作副本>\\settings\\hf），那样面板的
    「已下载 / 占用」会和 asr 实际加载的目录对不上。
    """
    return _RUNTIME_HF


def repo_id(size: str) -> str:
    return _REPO_OVERRIDES.get(size, f"Systran/faster-whisper-{size}")


def is_custom(size: str) -> bool:
    """是否是不走 Systran 命名的型号（需要按仓库/本地路径加载）。"""
    return size in _REPO_OVERRIDES


def load_ref(size: str) -> str:
    """给 WhisperModel 用的型号参数。

    自定义型号的名字（如 kotoba-whisper-v2.0）不在 faster-whisper 的内置表里，
    直接传名字会 ValueError；已下载时改为传快照目录，顺带避免任何联网检查。
    有通过校验的修复副本时优先用它（字词时间码才不会崩）。"""
    if not is_custom(size):
        return size
    return align_fix_dir(size) or snapshot_dir(size) or repo_id(size)


def align_fix_dir(size: str) -> Optional[str]:
    """kotoba 的修复副本目录；不存在、不完整或 config 非法时返回 None。

    校验必须足够严：半拷贝的 model.bin 或坏掉的 config.json 一旦被交给
    WhisperModel，要么加载失败（静默掉到 small），要么在 align 时再次崩进程。
    所以文件齐全、model.bin 体积与原快照一致、alignment_heads 的层号/头号
    都在解码器范围内，四条全过才算可用。"""
    name = ALIGN_FIXED.get(size)
    if not name:
        return None
    path = os.path.join(os.path.dirname(_RUNTIME_HF), "models", name)
    if not all(os.path.exists(os.path.join(path, f)) for f in _FIXED_REQUIRED):
        return None
    snap = snapshot_dir(size)
    try:
        if snap and os.path.getsize(os.path.join(path, "model.bin")) != \
                os.path.getsize(os.path.join(snap, "model.bin")):
            return None
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            heads = json.load(f).get("alignment_heads")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(heads, list) or not heads:
        return None
    for pair in heads:
        if not (isinstance(pair, list) and len(pair) == 2
                and all(isinstance(v, int) for v in pair)):
            return None
        layer, head = pair
        if not (0 <= layer < _FIXED_DECODER_LAYERS and 0 <= head < _FIXED_DECODER_HEADS):
            return None
    return path


def is_align_fixed(ref: str) -> bool:
    """ref 是否就是通过校验的修复副本目录（能力判断专用）。"""
    if not ref:
        return False
    try:
        target = os.path.normcase(os.path.realpath(ref))
    except OSError:
        return False
    for size in ALIGN_FIXED:
        fixed = align_fix_dir(size)
        if fixed and os.path.normcase(os.path.realpath(fixed)) == target:
            return True
    return False


def model_dir(size: str) -> str:
    # hub 的缓存目录名 = "models--" + 仓库名里的 "/" 换成 "--"
    return os.path.join(cache_root(), "hub", "models--" + repo_id(size).replace("/", "--"))


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


def _path_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def size_bytes(size: str) -> int:
    return _path_bytes(model_dir(size))


def total_bytes() -> int:
    """所有 Whisper 模型缓存合计占用的字节数。

    不能只匹配 Systran 前缀：turbo 在 mobiuslabsgmbh 名下、kotoba 在 kotoba-tech
    名下（见 _REPO_OVERRIDES），按前缀扫会漏算，界面上的「已占用」就偏小。"""
    hub = os.path.join(cache_root(), "hub")
    try:
        names = [n for n in os.listdir(hub)
                 if n.startswith("models--") and "whisper" in n.lower()]
    except OSError:
        return 0
    return sum(_path_bytes(os.path.join(hub, n)) for n in names)


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

    os.environ["HF_HOME"] = cache_root()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # asr 加载旧模型时会置 HF_HUB_OFFLINE=1；不清掉的话本次下载会直接按离线失败
    os.environ.pop("HF_HUB_OFFLINE", None)

    from huggingface_hub import snapshot_download

    repo = repo_id(size)
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
