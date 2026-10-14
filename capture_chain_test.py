# -*- coding: utf-8 -*-
"""采集链路保真度测试：链路本身有没有把音频弄坏（用户怀疑「采集本身就有缺陷」）。

把同一段合成语音走三种路径送进 Whisper，看识别结果的差异：

  ① 参考   ：mp3 → 16k 单声道                      （完全不过链路）
  ② 链路·关人声提取：→ 降噪 → 48k → 立体声 → 重采样 16k
  ③ 链路·开人声提取：→ 降噪 → 48k → 立体声 → 中置提取 → 重采样 16k   （= 出厂配置）

三者的识别文本若一致，说明链路无损；若有差异，说明链路里某个环节在伤音频。
立体声按「正中」摆位（真实直播里人声就在正中）。

用法：.runtime\\venv\\Scripts\\python.exe capture_chain_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.asr import WhisperASR                 # noqa: E402
from core.audio import resample, TARGET_SR      # noqa: E402
from core.config import AppConfig               # noqa: E402
from core.denoise import Denoise                # noqa: E402
from core.vocals import CenterExtractor         # noqa: E402
from title_asr_test import cer, decode, tts     # noqa: E402

LINES = [
    "こんにちは、みなさん！今日も元気に配信していきます",
    "黒神話：悟空が今すごく話題になってますよね",
    "このゲームは本当に難しくて、何回も死んじゃいました",
    "ありがとうございます、すごく嬉しいです",
]


def chain(audio16: np.ndarray, vocal: bool, denoise: bool) -> np.ndarray:
    """模拟采集链路：16k →(降噪) → 48k → 立体声 →(中置提取) → 16k。"""
    x = audio16.astype(np.float32)
    if denoise:
        d = Denoise()
        x = d.process(x)
    up = resample(x, TARGET_SR, 48000)                 # 设备播放采样率
    stereo = np.stack([up, up], axis=1)                # 正中摆位
    if vocal:
        ex = CenterExtractor(strength=0.8)
        x = ex.process(stereo[:, 0], stereo[:, 1])
    else:
        x = stereo.mean(axis=1)
    return resample(x, 48000, TARGET_SR)


def main() -> int:
    cfg = AppConfig.load()
    asr = WhisperASR(cfg.model_size, cfg.asr_device, cfg.compute_type, cfg.source_lang)
    print("加载模型 %s …" % cfg.model_size, flush=True)
    asr.load()
    print("设备：%s\n" % asr.dev_info, flush=True)

    agg = {"① 参考": 0.0, "② 链路(无提取)": 0.0, "③ 链路(有提取)": 0.0}
    diff = {"② 链路(无提取)": 0, "③ 链路(有提取)": 0}

    def rec(audio: np.ndarray) -> str:
        # 同一句话要连测三种路径，必须先清「连续相同文本」幻觉护栏：
        # 不清的话第 3 次会被判成幻觉直接返回空串，看起来像「链路把音频搞坏了」。
        asr.reset_repeat_guard()
        return asr.transcribe(audio, roster="") or ""

    for line in LINES:
        raw = decode(tts(line))
        ref = rec(raw)
        a = rec(chain(raw, vocal=False, denoise=True))
        b = rec(chain(raw, vocal=True, denoise=True))
        agg["① 参考"] += cer(line, ref)
        agg["② 链路(无提取)"] += cer(line, a)
        agg["③ 链路(有提取)"] += cer(line, b)
        if a != ref:
            diff["② 链路(无提取)"] += 1
        if b != ref:
            diff["③ 链路(有提取)"] += 1
        print("参考　：%s" % line)
        print("① 直喂：%s   (CER %.1f%%)" % (ref, cer(line, ref) * 100))
        print("② 链路：%s   (CER %.1f%%)%s" % (a, cer(line, a) * 100, "  ← 与①不同" if a != ref else ""))
        print("③ +提取：%s   (CER %.1f%%)%s" % (b, cer(line, b) * 100, "  ← 与①不同" if b != ref else ""))
        print(flush=True)

    n = len(LINES)
    print("=" * 64)
    for k, v in agg.items():
        print("%-14s 平均 CER %.1f%%" % (k, v / n * 100))
    print("文本与「① 直喂」不一致的句数：② %d/%d　③ %d/%d"
          % (diff["② 链路(无提取)"], n, diff["③ 链路(有提取)"], n))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
