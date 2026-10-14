# -*- coding: utf-8 -*-
"""端到端延迟测试。

合成日语语音 → 播到扬声器 → WASAPI loopback 抓回来 → VAD 断句 → Whisper 识别 → 翻译，
量出「这句话说完」到「字幕出现」的真实延迟，以及快语速下有没有漏句。

用法：.runtime\\venv\\Scripts\\python.exe latency_test.py
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import time
import urllib.parse

import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import AppConfig          # noqa: E402
from core.pipeline import Pipeline         # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".runtime", "tts_cache")
os.makedirs(CACHE, exist_ok=True)

_winmm = ctypes.windll.winmm


def mci(cmd: str) -> str:
    buf = ctypes.create_unicode_buffer(256)
    _winmm.mciSendStringW(cmd, buf, 256, 0)
    return buf.value


def tts(text: str) -> str:
    """百度日语 TTS，缓存到本地避免重复下载。"""
    # 缓存名用 md5：Python 的 str hash() 每个进程都会加随机盐（PYTHONHASHSEED），
    # 用它做文件名等于缓存永远不命中，每次跑都在重新下载（实测 4/4 全部未命中）。
    name = hashlib.md5(text.encode("utf-8")).hexdigest()[:16] + ".mp3"
    path = os.path.join(CACHE, name)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        # spd=5：贴近主播的正常语速（spd=3 是朗读腔，句内停顿长到会被 VAD 切开）
        url = ("https://fanyi.baidu.com/gettts?lan=jp&text=%s&spd=5&source=web"
               % urllib.parse.quote(text))
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def open_and_length(path: str) -> float:
    mci("close s")
    mci('open "%s" type mpegvideo alias s' % path)
    mci("set s time format milliseconds")
    return int(mci("status s length") or 0) / 1000.0


def play(path: str) -> None:
    mci("seek s to start")
    mci("play s")


def decode_mp3(path: str) -> np.ndarray:
    """mp3 → 16k 单声道 float32（`faster_whisper.audio.decode_audio` 在本机 PyAV 上不可用）。"""
    import av

    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def output_for_loopback(loopback_name: str) -> int:
    """把回环设备名映射到对应的**物理输出**设备 index（供 sounddevice 播放）。

    2026-10-13 踩坑：`mciSendStringW`（winmm）播放走的是 Windows「多媒体默认设备」，
    与 WASAPI/浏览器/游戏用的「控制台默认设备」可能不是同一个端点——实测 mci 在播
    （status=playing、position 在走），三个回环却一个回调都没有，测试永远识别不到东西。
    改成用 sounddevice（WASAPI）播到与采集同一个端点，链路才对得上。
    """
    base = loopback_name.replace(" [Loopback]", "")
    head = base.split(" (")[0]
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] >= 2 and head in d["name"]:
                return i
    except Exception:
        pass
    return -1


def play_wasapi(audio: np.ndarray, sr: int, loopback_name: str) -> None:
    """用 WASAPI 播到与采集端一致的那个输出设备上。"""
    import sounddevice as sd
    idx = output_for_loopback(loopback_name)
    sd.play(audio, sr, device=(idx if idx >= 0 else None), blocking=False)


# 模拟主播连说的四句话，长短不一
LINES = [
    "こんにちは、みなさん！",
    "今日は新しいゲームをやっていきたいと思います。",
    "えっと、まず最初に設定をちょっと確認しておきますね。",
    "ありがとうございます、すごく嬉しいです！",
]
GAP = 0.35          # 句间停顿：故意压得很短，模拟快语速
TAIL = 9.0          # 播放完后再等多久收结果


def main() -> int:
    cfg = AppConfig.load()
    print("配置：sensitivity=%.2f  min_silence=%dms  max_speech=%dms  模型=%s  设备=%s"
          % (cfg.sensitivity, cfg.min_silence_ms, cfg.max_speech_ms,
             cfg.model_size, cfg.asr_device), flush=True)

    print("正在下载合成语音…", flush=True)
    files = [(t, tts(t)) for t in LINES]

    results = []
    lock = __import__("threading").Lock()
    end_times = []

    def on_partial(seq: int, src: str) -> None:
        with lock:
            results.append((time.time(), src, None, "partial"))

    def on_result(seq: int, src: str, out: str) -> None:
        with lock:
            results.append((time.time(), src, out, "final"))

    pipe = Pipeline(cfg)
    pipe.on_partial = on_partial
    pipe.on_result = on_result
    pipe.on_status = lambda s: print("  [状态] " + s, flush=True)

    # 给识别和翻译套计时器，好看出时间到底花在哪一环
    import threading as _th
    _tl = _th.Lock()
    _asr = pipe.asr.transcribe
    _tr = pipe.translator.translate

    def timed_asr(seg, word_map="", **kw):
        t = time.time()
        try:
            return _asr(seg, word_map=word_map, **kw)
        finally:
            d = time.time() - t
            with _tl:
                print("      · 识别 %.2fs（音频 %.2fs）" % (d, len(seg) / 16000), flush=True)

    def timed_tr(text):
        t = time.time()
        try:
            return _tr(text)
        finally:
            d = time.time() - t
            with _tl:
                print("      · 翻译 %.2fs" % d, flush=True)

    pipe.asr.transcribe = timed_asr
    pipe.translator.translate = timed_tr

    print("加载模型并启动采集…", flush=True)
    pipe.start()
    for _ in range(120):
        if pipe.state == "running":
            break
        if pipe.state == "error":
            print("启动失败，放弃测试", flush=True)
            return 1
        time.sleep(0.25)
    else:
        print("启动超时", flush=True)
        return 1
    time.sleep(1.0)   # 让 VAD 先摸清底噪

    print("开始播放…", flush=True)
    loop_name = (cfg.device or {}).get("name", "")
    for text, path in files:
        audio = decode_mp3(path)
        dur = len(audio) / 16000.0
        # 用 WASAPI 播到与采集同一个端点（mci/winmm 走的是另一个默认设备，实测收不到）
        play_wasapi(audio, 16000, loop_name)
        t0 = time.time()
        end_times.append(t0 + dur)
        print("  ▶ %.2fs  %s" % (dur, text), flush=True)
        time.sleep(dur + GAP)

    time.sleep(TAIL)
    pipe.stop()
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass

    # 不能按时间窗口配对：主播连着说时 VAD 会把两句并成一段，字幕时间必然错位。
    # 改成按内容重叠配对——每句话找跟它字符重合最多的那条字幕。
    def overlap(a: str, b: str) -> float:
        import difflib
        return difflib.SequenceMatcher(
            None,
            "".join(c for c in a if c not in "、。，．！？!? "),
            "".join(c for c in b if c not in "、。，．！？!? "),
        ).ratio()

    # 完整字幕（含译文）才参与配对；partial 只用来量"字幕第一次动起来"的感知延迟
    finals = [r for r in results if r[3] == "final"]
    partials = [r for r in results if r[3] == "partial"]

    assigned = {}
    taken = set()
    for i, text in enumerate(LINES):
        best, bi = 0.0, None
        for j, r in enumerate(finals):
            if j in taken:
                continue
            v = overlap(text, r[1])
            if v > best:
                best, bi = v, j
        if bi is not None and best >= 0.5:
            assigned[i] = finals[bi]
            taken.add(bi)

    print("\n结果（延迟 = 字幕到达时刻 − 这句话播放结束时刻）：", flush=True)
    hit = 0
    p_lags, f_lags = [], []
    for i, (t_end, text) in enumerate(zip(end_times, LINES)):
        if i in assigned:
            fin = assigned[i]
            f_lag = fin[0] - t_end
            f_lags.append(f_lag)
            # 找这句最早出现的 partial（原文上屏）作为感知延迟
            p_lag = None
            for p in partials:
                if overlap(text, p[1]) >= 0.5:
                    p_lag = p[0] - t_end
                    break
            if p_lag is not None:
                p_lags.append(p_lag)
            hit += 1
            print("  %d. 感知 %+.2fs / 完整 %+.2fs | %s → %s"
                  % (i + 1, p_lag if p_lag is not None else float("nan"),
                     f_lag, fin[1], fin[2] or "（无译文）"), flush=True)
        else:
            print("  %d. ❌ 漏了 | %s" % (i + 1, text), flush=True)

    print("\n出句 %d/%d，收完整字幕 %d 条、原文上屏 %d 条"
          % (hit, len(LINES), len(finals), len(partials)), flush=True)
    if p_lags:
        print("  感知延迟均值 %+.2fs（原版本没这步，只能等完整字幕）" % (sum(p_lags) / len(p_lags)), flush=True)
    if f_lags:
        print("  完整字幕延迟均值 %+.2fs" % (sum(f_lags) / len(f_lags)), flush=True)
    return 0 if hit == len(LINES) else 2


if __name__ == "__main__":
    sys.exit(main())
