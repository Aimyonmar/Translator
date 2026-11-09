# -*- coding: utf-8 -*-
"""翻译保活：空闲时每 15 秒向免费通道发一次极短请求，保住 HTTPS 连接。

只管线程生命周期；通道选择、是否该发由 owner（Translator）决定。
默认关闭（cfg.enable_keepalive=False）：关闭时不创建线程、零出站请求。
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class Keepalive:
    """同一时刻最多一条保活线程；stop 后阻塞在 Event.wait 上的线程立即返回。"""

    def __init__(self, owner) -> None:
        self._owner = owner
        # 每轮换一个新的 Event，避免 stop→start 之间把上一轮线程「复活」
        self._lock = threading.Lock()
        self._started = False
        self._suspended = False
        self._stop: Optional[threading.Event] = None

    def start(self) -> None:
        o = self._owner
        if not getattr(o.cfg, "enable_keepalive", False):
            return
        o._last_req = getattr(o, "_last_req", 0.0) or time.time()
        with self._lock:
            if self._started or self._suspended:
                return
            stop_evt = threading.Event()
            self._stop = stop_evt
            self._started = True
            threading.Thread(target=self._loop, args=(stop_evt,),
                             daemon=True, name="translator-keepalive").start()
        if not o._custom_ready():
            threading.Thread(target=self._ping_once, args=(stop_evt,),
                             daemon=True, name="translator-keepalive-ping").start()

    def _ping(self, ch: str) -> None:
        """一次极短请求。不写 _last_req：那是「真实翻译最后活跃时刻」的口径，
        保活自己更新它会让下一轮 tick 判定为「刚翻译过」而跳过，间隔翻倍到 30 秒。"""
        o = self._owner
        if o._custom_ready() or ch in o._dead:
            return
        try:
            o._dispatch(ch, "あ", timeout=2.5)
        except Exception:
            pass

    def _ping_once(self, stop_evt: threading.Event) -> None:
        """开关刚打开时立刻温一下首选通道；停止请求已下达就不再发。"""
        o = self._owner
        for ch in ("tencent", "caiyun"):
            if stop_evt.is_set():
                return
            if ch not in o._dead and not o._custom_ready():
                self._ping(ch)
                return

    def _loop(self, stop_evt: threading.Event) -> None:
        """每 15 秒 ping 一个可用通道（轮换），一分钟约 4 次，不双发浪费额度。"""
        o = self._owner
        turn = 0
        try:
            while not stop_evt.wait(15):
                if not getattr(o.cfg, "enable_keepalive", False):
                    return
                if o._custom_ready():       # 用户自己配了 API，不必占免费通道额度
                    continue
                if time.time() - getattr(o, "_last_req", 0.0) < 15:
                    continue
                chans = [ch for ch in ("tencent", "caiyun") if ch not in o._dead]
                if not chans:
                    continue
                self._ping(chans[turn % len(chans)])
                turn += 1
        finally:
            with self._lock:
                # 只清理自己这一轮；stop→start 已被新 Event 接管时不能动新线程的状态
                if self._stop is stop_evt:
                    self._started = False
                    self._stop = None

    def stop(self) -> None:
        """停掉保活并阻止再次自启（直到 resume）。

        线程阻塞在 Event.wait 上，set() 后立即返回并退出；Pipeline.stop() 也走这里，
        所以程序停止后不会再有任何 ping。
        """
        with self._lock:
            self._suspended = True
            stop_evt, self._stop = self._stop, None
            self._started = False
        if stop_evt is not None:
            stop_evt.set()

    def resume(self) -> None:
        """解除 stop 的抑制（新一轮开始翻译时调用），不主动起线程。"""
        with self._lock:
            self._suspended = False
