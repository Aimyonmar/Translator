# -*- coding: utf-8 -*-
"""主备竞速：先发主通道，超时未回或先失败再补发备胎，取先到的可用译文。

只做线程调度与「谁先到」的判定，通道是否可用、失败后拉黑等策略由调用方提供。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Callable, List, Optional, Tuple


def race(chs: List[str], work: Callable[[str], Optional[str]],
         accept: Callable[[Optional[str]], bool], timeout: float,
         delay_ms: int) -> Tuple[Optional[tuple], dict, int]:
    """跑 1~2 条线路。

    work(ch) 发请求并返回译文（失败返回 None，由调用方决定是否拉黑）；
    accept(out) 判定译文是否可用（比如过滤乱码）。
    返回 (best, done_at, fired)：best=(译文, 通道) 或 None，
    done_at 是各通道完成时刻（相对起跑），fired 是实际发起的线路数。
    """
    t0 = time.perf_counter()
    done_at: dict = {}

    def run(ch):
        try:
            return work(ch)
        finally:
            done_at[ch] = time.perf_counter() - t0

    ex = ThreadPoolExecutor(max_workers=2)
    futs = {ex.submit(run, chs[0]): chs[0]}
    fired = 1
    if delay_ms <= 0 and len(chs) > 1:
        futs[ex.submit(run, chs[1])] = chs[1]
        fired = 2
    best = None
    while futs and best is None:
        if fired < len(chs) and delay_ms > 0:
            left = delay_ms / 1000.0 - (time.perf_counter() - t0)
            if left > 0:
                done, _ = wait(list(futs), timeout=left, return_when=FIRST_COMPLETED)
            else:
                done = set()
        else:
            done, _ = wait(list(futs), timeout=timeout + 0.5,
                           return_when=FIRST_COMPLETED)
        if not done:
            if fired < len(chs):       # 主通道到点还没回：补发备份
                futs[ex.submit(run, chs[1])] = chs[1]
                fired += 1
                continue
            break
        for f in done:
            ch = futs.pop(f)
            try:
                out = f.result()
            except Exception:
                out = None
            if out and out.strip() and accept(out):
                best = (out, ch)
                break
            if fired < len(chs):       # 主通道先失败了：立刻补发备份
                futs[ex.submit(run, chs[1])] = chs[1]
                fired += 1
    ex.shutdown(wait=False)
    return best, done_at, fired
