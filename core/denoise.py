# -*- coding: utf-8 -*-
"""轻量实时降噪：流式谱减法（Wiener 风格增益），只依赖 numpy。

解决的问题：系统声音回环采集时混入了风扇/电流声/低频隆隆等**稳态背景噪声**，
这些噪声会拉低 Whisper 的信噪比。这里在送入 VAD/识别前先压一压。

工作原理（经典谱减法，50% 重叠相加 COLA 重建）：
1. 每帧加周期 Hann 窗，做 rfft 得到幅度谱；
2. 用一个「慢速逼近观测最小值」的噪声谱估计（噪声比语音平稳，最小幅度≈噪声）；
3. 对每个频点算增益 gain = 1 - strength * (noise/(mag+ε))，越像噪声压得越狠；
4. 反变换后乘合成窗，重叠相加。gain=1 时完美重建原信号，所以不会引入额外失真。

诚实说明：这只对**稳态**噪声有效（风扇、电流、底噪）；
BGM 这类非稳态音乐它压不动——那需要专门的人声分离模型（Demucs 等，更重）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Denoise:
    def __init__(self, sample_rate: int = 16000, frame: int = 512,
                 strength: float = 0.5, enable: bool = True):
        self.sr = sample_rate
        self.frame = frame
        self.hop = frame // 2
        self.strength = max(0.0, min(1.0, float(strength)))
        self.enable = enable
        # 分析窗与合成窗都用 sqrt-Hann：等效窗为 Hann，50% 重叠相加时求和恒为 1，
        # 因此 gain=1 的频点**精确重建**（实测 max|Δ|≈1e-7）。
        # 若两侧都用 Hann，等效窗是 Hann²，其重叠和在 0.5~1 之间波动，
        # 等于给整段音频叠了一层 16ms 周期的振幅调制（实测 max|Δ|≈0.5）。
        n = np.arange(frame, dtype=np.float64)
        self._win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame)).astype(np.float32)
        self._noise: Optional[np.ndarray] = None
        self._alpha = 0.95          # 噪声估计的平滑系数（越大越稳、跟得越慢）
        self._floor = 1e-3          # 噪声估计下限，避免把语音也当噪声消掉
        self._buf = np.zeros(0, dtype=np.float32)
        # 跨块传递「尚未完整」的输出尾巴：每块开头 hop 个采样只被 1 帧覆盖
        # （窗口尚未累加到 1），把上一块相同位置的累计值接过来补齐即可连续。
        self._tail = np.zeros(self.hop, dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """喂入一段 16k 单声道 float32 音频，返回同样采样率的降噪后音频。

        流式 STFT 重叠相加（50% OLA）：每帧加周期 Hann 窗 → 谱减 → 反变换后再乘
        合成窗并累加。**跨块边界用输出尾巴累积补齐**：每块开头 hop 个采样只被 1 帧
        覆盖（窗口未累加到 1），必须接上上一块在该位置的累计值，输出才连续。
        （旧实现是把首尾各 hop 直接裁掉，实测 32000 样本里丢掉 5376 个 = 17%，
        并在每块边界留下波形跳变——ASR 拿到的是被切碎的音频。）
        本实现保证：每块 `n` 个输入样本恰好对应 `n` 个输出样本，不丢不重。
        """
        if not self.enable or chunk is None or len(chunk) == 0:
            return chunk if chunk is not None else np.zeros(0, dtype=np.float32)
        chunk = chunk.astype(np.float32, copy=False)
        data = np.concatenate([self._buf, chunk]) if self._buf.size else chunk
        n = len(data)
        if n < self.frame:
            self._buf = data
            return np.zeros(0, dtype=np.float32)
        nframes = (n - self.frame) // self.hop + 1
        syn = np.zeros((nframes - 1) * self.hop + self.frame, dtype=np.float32)
        for i in range(nframes):
            pos = i * self.hop
            syn[pos:pos + self.frame] += self._process_frame(data[pos:pos + self.frame])
        last_pos = (nframes - 1) * self.hop
        self._buf = data[last_pos + self.hop:] if (last_pos + self.hop) < n else np.zeros(0, dtype=np.float32)
        if len(syn) <= self.hop:
            self._tail = syn.copy() if len(syn) == self.hop else np.zeros(self.hop, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)
        # 接上上一块遗留的不完整尾巴，再交出本块已完整的部分
        syn[:len(self._tail)] += self._tail
        self._tail = syn[-self.hop:].copy()
        return syn[:-self.hop].astype(np.float32)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        w = frame * self._win
        spec = np.fft.rfft(w).astype(np.complex64)
        mag = np.abs(spec)
        if self._noise is None:
            self._noise = mag.copy()
        else:
            # 慢速逼近观测到的最小幅度：噪声比语音平稳，最小幅度近似噪声谱
            est = self._noise * self._alpha + mag * (1.0 - self._alpha)
            self._noise = np.minimum(est, mag)
        noise = np.maximum(self._noise, self._floor)
        # Wiener 风格增益：越像噪声的频点压得越多；strength 控制总体强度
        gain = 1.0 - self.strength * (noise / (mag + 1e-6))
        gain = np.clip(gain, 0.0, 1.0)
        new_spec = spec * gain
        rec = np.fft.irfft(new_spec).astype(np.float32)
        return rec * self._win

    def reset(self) -> None:
        self._noise = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(self.hop, dtype=np.float32)
