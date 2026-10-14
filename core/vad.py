# -*- coding: utf-8 -*-
"""语音活动检测（VAD）与音频分段。

采用「自适应能量阈值」：持续跟踪背景噪声底噪，说话时能量高于底噪即判定为语音，
连续静音超过阈值则把整段语音切出来送去识别。对直播里带 BGM 的场景也能工作。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class SpeechSegmenter:
    """不断喂入 16k 单声道音频块，吐出一段段完整语音。"""

    def __init__(self, sample_rate: int = 16000, sensitivity: float = 0.55,
                 min_speech_ms: int = 250, min_silence_ms: int = 450,
                 max_speech_ms: int = 15000, pre_roll_ms: int = 300):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_speech_ms = max_speech_ms
        self.pre_roll_ms = pre_roll_ms
        self.last_level = 0.0
        self.threshold = 0.01
        self.in_speech = False
        self._reset()

    def _reset(self) -> None:
        self._noise = 0.01
        self._buf: List[np.ndarray] = []
        self._levels: List[float] = []     # 与 _buf 平行的每块电平（pre-roll 块记 +inf）
        self._speech_ms = 0
        self._silence_ms = 0
        self._pre: Deque[np.ndarray] = deque()
        self._pre_ms = 0
        self.in_speech = False
        # 最近若干帧的电平历史：底噪取其中的低分位，避免被说话声/BGM 带跑
        self._hist: Deque[float] = deque(maxlen=240)

    def update_params(self, sensitivity: Optional[float] = None, min_speech_ms: Optional[int] = None,
                      min_silence_ms: Optional[int] = None, max_speech_ms: Optional[int] = None) -> None:
        if sensitivity is not None:
            self.sensitivity = sensitivity
        if min_speech_ms is not None:
            self.min_speech_ms = min_speech_ms
        if min_silence_ms is not None:
            self.min_silence_ms = min_silence_ms
        if max_speech_ms is not None:
            self.max_speech_ms = max_speech_ms

    def reset(self) -> None:
        self._reset()

    # -- 核心
    def feed(self, chunk: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """返回 (语音段或 None, 当前音量)。"""
        if chunk is None or len(chunk) == 0:
            return None, self.last_level
        x = chunk.astype(np.float32, copy=False)
        level = float(np.sqrt(np.mean(np.square(x))))
        self.last_level = level
        chunk_ms = len(x) * 1000.0 / self.sample_rate

        ratio = 5.0 - 4.0 * float(self.sensitivity)
        floor = 0.0016 * ratio
        self.threshold = max(self._noise * ratio, floor)
        voiced = level > self.threshold
        self.voiced = voiced

        # 底噪跟踪：只在「既不处于语音中、当前块也不响」的静音期更新。
        # 说话/游戏音/BGM 持续期间必须冻结——否则持续声会把底噪估计一路推高，
        # 阈值跟着涨到语音电平，连续语音被剁成一串 0.6s 碎段
        # （实测 16s 连续语音被切成 21 段，碎段全是 ASR 固定开销，又慢又碎）。
        if not voiced and not self.in_speech:
            self._hist.append(level)
            if len(self._hist) >= 16:
                target = float(np.percentile(np.fromiter(self._hist, dtype=np.float64), 10))
            else:
                target = float(min(self._hist))
            self._noise += (target - self._noise) * 0.08
            self._noise = float(min(max(self._noise, 1e-5), 0.08))

        if not self.in_speech:
            self._pre.append(x)
            self._pre_ms += chunk_ms
            while self._pre_ms > self.pre_roll_ms and len(self._pre) > 1:
                self._pre_ms -= len(self._pre[0]) * 1000.0 / self.sample_rate
                self._pre.popleft()

        if voiced:
            if not self.in_speech:
                self.in_speech = True
                self._buf = list(self._pre)
                self._levels = [float("inf")] * len(self._buf)
                self._speech_ms = 0
                self._silence_ms = 0
            self._buf.append(x)
            self._levels.append(level)
            self._speech_ms += chunk_ms
            self._silence_ms = 0
            if self._speech_ms >= self.max_speech_ms:
                return self._cut_at_dip(), level
            return None, level

        # 静音
        if self.in_speech:
            self._buf.append(x)
            self._levels.append(level)
            self._silence_ms += chunk_ms
            if self._silence_ms >= self.min_silence_ms:
                if self._speech_ms >= self.min_speech_ms:
                    return self._emit(), level
                self._reset()
        return None, level

    def _cut_at_dip(self) -> Optional[np.ndarray]:
        """max_speech 到点时的强切：在尾部窗口里挑「块能量最低」的边界下刀。

        直接在 max_speech_ms 处硬切，刀口会落在词中间（日语音素跨块），
        首尾两段的识别质量都受影响。改为在尾部 ~600ms 里找能量低谷（多半是
        词间空隙）下刀；刀后的音频原样保留为下一段的开头，不丢音、不加延迟。
        找不到像样的低谷就退回原来的行为（整段吐出）。"""
        if not self._buf:
            self._reset()
            return None
        data = np.concatenate(self._buf)
        n = len(data)
        sr = self.sample_rate
        if n < int(0.6 * sr):                 # 段太短，没有挑选余地
            self._reset()
            return data
        # 候选窗口：最后一个块往前 ~600ms（跳过最后一个块——切在末尾等于没挑）
        starts: List[int] = []
        pos = 0
        for c in self._buf:
            starts.append(pos)
            pos += len(c)
        win_start = max(0, n - int(0.6 * sr))
        lo = 1
        while lo < len(starts) and starts[lo] < win_start:
            lo += 1
        hi = len(starts) - 1                   # 不含最后一个块
        if lo >= hi:                           # 窗口里没有可挑的块间边界
            self._reset()
            return data
        best_i, best_lv = lo, self._levels[lo]
        for i in range(lo + 1, hi):
            if self._levels[i] < best_lv:
                best_i, best_lv = i, self._levels[i]
        if best_lv >= self.threshold:          # 窗口里没有低于阈值的安全点，放弃挑选
            self._reset()
            return data
        p = starts[best_i]
        head, tail = data[:p], data[p:]
        self._reset()
        if len(tail) > 0:
            # 余下音频滚进下一段：不丢音，下一段的 max_speech 从刀口重新计时
            self.in_speech = True
            self._buf = [tail]
            tl = float(np.sqrt(np.mean(np.square(tail))))
            self._levels = [tl]
            self._speech_ms = len(tail) * 1000.0 / self.sample_rate
            self._silence_ms = 0
        return head

    def _emit(self, force: bool = False) -> Optional[np.ndarray]:
        if not self._buf:
            self._reset()
            return None
        tail = self.min_silence_ms if not force else 0
        data = np.concatenate(self._buf)
        # 去掉尾部多余的静音，保留一点点自然收尾
        cut = int(tail * self.sample_rate / 1000.0)
        if cut > 0 and len(data) > cut + self.sample_rate // 10:
            data = data[: len(data) - cut + self.sample_rate // 20]
        self._reset()
        return data

    def flush(self) -> Optional[np.ndarray]:
        """停止时把残留的一段吐出来。"""
        if self.in_speech and self._speech_ms >= self.min_speech_ms and self._buf:
            return self._emit(force=True)
        self._reset()
        return None
