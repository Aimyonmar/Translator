# -*- coding: utf-8 -*-
"""人声提取（中置声道提取）：把宽声场的 BGM 压低，保留正中的说话声。

**为什么是这条路**：模型级人声分离（Demucs / MDX）效果最好，但本机实测
显存只剩 ~1.5GB（htdemucs 需 2GB+），且 torch/demucs 都未安装
（要额外下 ~2.5GB 依赖）——当前环境下不可行。而采集端本来就是**立体声回环**，
立体声里天然带着「位置信息」：真实混音里人声几乎总是钉在正中（L≈R），
而 BGM/伴奏常被做宽（混响、立体声展开、左右不同相）。本模块就利用这一点。

**算法**（逐帧 STFT，50% 重叠相加，与 `denoise.py` 同一套 OLA 框架）：

    mid   = (L + R) / 2          # 正中成分
    side  = (L - R) / 2          # 侧向（宽声场）成分
    coh   = |mid| / (|mid| + |side| + ε)     # 每个频点的「中置度」∈ [0,1]
    gain  = (1 - strength) + strength * coh
    out   = irfft(mid * gain)

**关键性质（可测）**：L 与 R 完全相同的成分（纯正中 / 本来就是单声道）
`side = 0 → coh = 1 → gain = 1`，输出**逐样本无损可复原**；
完全反相的内容（L = −R）`mid = 0 → gain = 1−strength`，被压到几乎为零。
所以「压不掉中置混音的 BGM」是它的**已知上限**，但它绝不会像谱减法那样
误伤本就正中的语音——这是相对 `denoise.py`（谱减法，压不动非稳态音乐）的主要优势。

线程与性能：纯 numpy，无模型、无显存；每帧两次 rfft + 一次 irfft，
与降噪同量级（实测 <0.2ms / 64ms 音频）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["CenterExtractor", "extract_center"]


class CenterExtractor:
    """流式中置提取：喂入左右声道，吐出「保留正中成分」的单声道。"""

    def __init__(self, frame: int = 512, strength: float = 0.8):
        self.frame = int(frame)
        self.hop = self.frame // 2
        self.strength = float(min(max(strength, 0.0), 1.0))
        n = np.arange(self.frame, dtype=np.float64)
        # sqrt-Hann 双侧加窗（见 denoise.py 的说明）：gain=1 的频点精确重建，
        # 不会有 Hann² 那种 16ms 周期的振幅调制。
        self._win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / self.frame)).astype(np.float32)
        self._bl = np.zeros(0, dtype=np.float32)
        self._br = np.zeros(0, dtype=np.float32)
        # 跨块传递「尚未完整」的输出尾巴（与 denoise.py 同一套修复）：
        # 每块开头 hop 个采样只被 1 帧覆盖，接上上一块在该位置的累计值才连续。
        self._tail = np.zeros(self.hop, dtype=np.float32)
        self.last_coh = 1.0          # 最近一帧的平均中置度，供界面/日志观察

    def process(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """喂入等长的左右声道块，返回提取后的单声道。

        跨块边界用输出尾巴累积补齐（与 `denoise.py` 同一套修复）：
        每块 `n` 个输入样本恰好对应 `n` 个输出样本，不丢不重、波形连续。
        """
        if left is None or right is None:
            return np.zeros(0, dtype=np.float32)
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        n = min(len(left), len(right))
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        left, right = left[:n], right[:n]

        l = np.concatenate([self._bl, left]) if self._bl.size else left
        r = np.concatenate([self._br, right]) if self._br.size else right
        n = min(len(l), len(r))
        if n < self.frame:
            self._bl, self._br = l, r
            return np.zeros(0, dtype=np.float32)

        nframes = (n - self.frame) // self.hop + 1
        syn = np.zeros((nframes - 1) * self.hop + self.frame, dtype=np.float32)
        for i in range(nframes):
            pos = i * self.hop
            syn[pos:pos + self.frame] += self._frame(
                l[pos:pos + self.frame], r[pos:pos + self.frame])
        last = (nframes - 1) * self.hop
        keep = last + self.hop
        self._bl = l[keep:] if keep < n else np.zeros(0, dtype=np.float32)
        self._br = r[keep:] if keep < n else np.zeros(0, dtype=np.float32)
        if len(syn) <= self.hop:
            self._tail = syn.copy() if len(syn) == self.hop else np.zeros(self.hop, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)
        syn[:len(self._tail)] += self._tail
        self._tail = syn[-self.hop:].copy()
        return syn[:-self.hop].astype(np.float32)

    def _frame(self, fl: np.ndarray, fr: np.ndarray) -> np.ndarray:
        wl = fl * self._win
        wr = fr * self._win
        L = np.fft.rfft(wl)
        R = np.fft.rfft(wr)
        mid = (L + R) * 0.5
        side = (L - R) * 0.5
        mag = np.abs(mid)
        smag = np.abs(side)
        coh = mag / (mag + smag + 1e-9)
        try:
            self.last_coh = float(coh.mean())
        except Exception:
            pass
        gain = (1.0 - self.strength) + self.strength * coh
        return (np.fft.irfft(mid * gain)).astype(np.float32) * self._win

    def reset(self) -> None:
        self._bl = np.zeros(0, dtype=np.float32)
        self._br = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(self.hop, dtype=np.float32)


def extract_center(stereo: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """离线入口：立体声 (n, 2) → 提取后的单声道（长度与输入一致，不丢样）。"""
    a = np.asarray(stereo, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 2:
        return a.reshape(-1).astype(np.float32)
    ex = CenterExtractor(strength=strength)
    out = ex.process(a[:, 0], a[:, 1])
    # 冲掉缓冲（尾巴 + 残留输入），保持总长相符
    for _ in range(3):
        tail = ex.process(np.zeros(ex.frame, dtype=np.float32),
                          np.zeros(ex.frame, dtype=np.float32))
        if tail.size == 0:
            break
        out = np.concatenate([out, tail])
    return out.astype(np.float32)


# ---------------------------------------------------------------- 自测
if __name__ == "__main__":
    import sys
    import time

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr * 2) / sr
    tok = 0
    total = 0

    def check(name: str, cond: bool, extra: str = "") -> None:
        global tok, total
        total += 1
        tok += int(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'} | {name} {extra}")

    # ① 纯正中内容（L == R）必须无损。流式开头 hop 个采样处于窗口爬升段（流起始
    #    的 16ms 淡入），从 hop 之后比对；长度必须与输入一致（不丢样）。
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    st = np.stack([x, x], axis=1)
    y = extract_center(st)
    hop = CenterExtractor().hop
    m = min(len(x), len(y))
    diff = float(np.max(np.abs(y[hop:m] - x[hop:m]))) if m > hop else 1.0
    check("正中内容无损（hop 之后逐样本一致）", diff < 1e-5 and m >= len(x),
          f"(max|Δ|={diff:.2e}, 长度 {len(x)}→{len(y)})")

    # ② 完全反相（L = −R）是纯侧向内容，必须被大幅压低
    st2 = np.stack([x, -x], axis=1)
    y2 = extract_center(st2)
    e_in, e_out = float(np.mean(x ** 2)), float(np.mean(y2 ** 2)) if len(y2) else 0.0
    check("纯侧向内容被压低 >20x", e_in / max(e_out, 1e-12) > 20,
          f"(抑制 {10*np.log10(e_in/max(e_out,1e-12)):.1f} dB)")

    # ③ BGM 抑制量（分两档，如实反映物理上限）：
    #    度量方式——把提取结果与「纯净语音（正中，增益 1 应无损）」相减，
    #    残差就是残留 BGM 能量，与「降混后的 BGM 能量」比较得到抑制量。
    #
    #    重要结论：抑制量取决于 BGM 有多「宽」。
    #      · 完全宽（左右反相）：能量全在 side，可压到 -100dB 级；
    #      · 部分去相关（左右差 60° 相位，更接近真实混音）：有 cos²(30°)≈68%
    #        的能量本来就落在 mid 里，中置提取拿不走，只能压 ~3dB。
    #    这不是实现缺陷，是「用位置信息分离」的原理上限——中置混音的 BGM 压不掉。
    speech = (0.25 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

    def bgm_suppression(phase: float) -> float:
        ml = (0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        mr = (0.12 * np.sin(2 * np.pi * 440 * t + phase)).astype(np.float32)
        out = extract_center(np.stack([(speech + ml).astype(np.float32),
                                       (speech + mr).astype(np.float32)], axis=1))
        m = min(len(speech), len(out))
        e_res = float(np.mean((out[:m] - speech[:m]) ** 2))
        e_mus = float(np.mean((((ml + mr) / 2)[:m]) ** 2))
        return 10 * np.log10(e_mus / max(e_res, 1e-12))

    s_part = bgm_suppression(1.2)                 # 部分去相关（真实混音更接近这档）
    # 完全反相（左右差 180°）这一档不在此处断言：此时 mono 降混本身就已把 BGM 抵消
    # （mid=0），比值退化；它由上面的用例 ② 直接覆盖（纯侧向信号被压 >20x）。
    check("部分去相关的 BGM 抑制 ≥3dB", s_part >= 3.0, f"({s_part:.1f} dB)")

    # ④ 耗时（每块 64ms 立体声）
    blk_l = rng.normal(0, 0.05, 1024).astype(np.float32)
    blk_r = rng.normal(0, 0.05, 1024).astype(np.float32)
    ex = CenterExtractor()
    for _ in range(50):
        ex.process(blk_l, blk_r)
    t0 = time.perf_counter()
    for _ in range(300):
        ex.process(blk_l, blk_r)
    ms = (time.perf_counter() - t0) / 300 * 1000
    check("吞吐 ≥200x 实时", ms < 64 / 200, f"({ms:.3f} ms / 64ms 音频)")

    print(f"\n{tok}/{total} 通过")
    sys.exit(0 if tok == total else 1)
