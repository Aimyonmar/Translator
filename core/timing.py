# -*- coding: utf-8 -*-
"""分段计时：只做诊断，默认关闭。

为什么要单独一个模块：
- 用户要的是「每句的时间到底花在哪一环」，这是临时诊断，不该写进 config.json，
  也不该在正常使用时产生日志和开销；
- 通过环境变量开启，关掉后各计时点只剩一次 perf_counter（纳秒级）。

开启方式（PowerShell）：
    $env:TRANSLATOR_TIMING = "1"
    $env:TRANSLATOR_TIMING_LOG = "<项目目录>\\.runtime\\logs\\timing.jsonl"  # 可选
每句一行 JSON，字段见 pipeline._flush_ordered 里的 timing.record(...)。
"""
from __future__ import annotations

import json
import os
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG = os.path.join(ROOT, ".runtime", "logs", "timing.jsonl")

_ENABLED = None
_LOCK = threading.Lock()


def enabled() -> bool:
    """是否开启计时（只读一次环境变量，避免每句都解析）。"""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = (os.environ.get("TRANSLATOR_TIMING", "") or "").strip().lower() \
            not in ("", "0", "false", "no", "off")
    return _ENABLED


def log_path() -> str:
    return os.environ.get("TRANSLATOR_TIMING_LOG") or DEFAULT_LOG


def record(**fields) -> None:
    """追加一行 JSON。任何异常都吞掉：诊断代码绝不能影响字幕主流程。"""
    if not enabled():
        return
    row = {"ts": round(time.time(), 3)}
    for k, v in fields.items():
        row[k] = round(v, 1) if isinstance(v, float) else v
    try:
        path = log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def reset_log() -> None:
    """测试脚本用：开跑前清空，别把上一轮的数据混进来。"""
    try:
        with open(log_path(), "w", encoding="utf-8"):
            pass
    except Exception:
        pass
