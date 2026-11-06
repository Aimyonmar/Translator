# -*- coding: utf-8 -*-
"""音频转文字（离线文件转写）：把整段音频/视频转成带时间戳的文本，导出 TXT / SRT。

与实时字幕的区别：
- 输入是文件，不是系统声音回环；一次性转完，进度按「已转写时长 / 总时长」推进；
- 解码宽度用 5（离线更看重准确度，实时那套 beam=2 是为延迟服务的）；
- 打开 VAD 过滤，跳过整段静音，避免 Whisper 在长静音上循环吐幻觉句。
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from .asr import _batch_supported

SAMPLE_RATE = 16000

# 句段：(起始秒, 结束秒, 文本)
Segment = Tuple[float, float, str]

# faster-whisper 走 PyAV/ffmpeg 解码，下面这些容器/编码都能直接读
AUDIO_EXTS = (
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts", ".flv",
)
FILE_FILTER = ("音频/视频 (" + " ".join("*" + e for e in AUDIO_EXTS) + ");;"
               "所有文件 (*.*)")


def format_ts(seconds: float, sep: str = ".") -> str:
    """秒 → HH:MM:SS.mmm（SRT 用 sep=","）。"""
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000.0))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(segments: List[Segment]) -> str:
    """TXT：每行「[起 → 止] 文本」，时间戳保留到毫秒，方便对照原片。"""
    return "".join(
        f"[{format_ts(a)} → {format_ts(b)}] {t}\n" for a, b, t in segments
    )


def to_srt(segments: List[Segment]) -> str:
    """标准 SRT：序号 + 时间轴 + 文本，播放器/剪辑软件可直接载入。"""
    out = []
    for i, (a, b, t) in enumerate(segments, 1):
        out.append(f"{i}\n{format_ts(a, ',')} --> {format_ts(b, ',')}\n{t}\n")
    return "\n".join(out)


def decode_audio(path: str, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """把任意音视频文件解成单声道 16k float32 PCM。

    不复用 faster-whisper 自带的 decode_audio：它给 `av.open` 传了
    metadata_errors="ignore"，而 PyAV 19 已经删掉这个参数（实时字幕只喂内存
    数组，从不走文件解码，所以这个问题一直没暴露）。这里统一解成数组再交给
    Whisper，PyAV 新旧版本都能跑。
    """
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks: List[np.ndarray] = []

    def _take(out) -> None:
        if out is None:
            return
        if isinstance(out, (list, tuple)):
            for frame in out:
                _take(frame)
            return
        arr = out.to_ndarray()
        if arr.ndim > 1:
            arr = arr[0]
        chunks.append(arr)

    with av.open(path) as container:
        if not container.streams.audio:
            raise RuntimeError("这个文件里没有音轨")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            _take(resampler.resample(frame))
        _take(resampler.resample(None))     # 冲掉重采样器里剩下的尾巴

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _apply_fixes(text: str, word_map: str, roster: str) -> str:
    """和实时字幕用同一套确定性修正（纠错表 + 音近人名），保证两条路径出来的名字一致。"""
    from .asr import WhisperASR
    if word_map:
        text = WhisperASR.apply_word_map(text, word_map)
    if roster:
        from .namefix import apply_names
        text = apply_names(text, roster)
    return text


def transcribe_media(asr, path: str, language: Optional[str] = "ja",
                     word_map: str = "", roster: str = "",
                     on_progress: Optional[Callable[[int], None]] = None,
                     should_stop: Optional[Callable[[], bool]] = None
                     ) -> Tuple[List[Segment], float]:
    """转写一个音视频文件，返回 (句段列表, 总时长秒)。

    复用已经加载好的实时模型（asr.model），不另起一份 —— 两份模型同时驻留显存
    正是「与实时字幕互斥」要避免的情况。调用方负责保证模型已 load。
    """
    if asr.model is None:
        raise RuntimeError("模型尚未加载")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    kwargs = dict(
        language=language,
        beam_size=5,                    # 离线不吃延迟，用更宽的解码换准确度
        vad_filter=True,                # 长静音/纯 BGM 段落跳过，避免循环幻觉
        condition_on_previous_text=False,
        temperature=[0.0, 0.4, 0.6],
    )
    if _batch_supported() and getattr(asr, "batch_size", 1) > 1:
        kwargs["batch_size"] = asr.batch_size

    audio = decode_audio(path)
    if audio.size == 0:
        raise RuntimeError("没有解出音频内容，确认文件没损坏、也不是纯图片/纯文本")

    segments: List[Segment] = []
    total = 0.0
    with asr.lock:                      # 与实时识别共用同一份模型，串行使用
        seg_iter, info = asr.model.transcribe(audio, **kwargs)
        total = float(getattr(info, "duration", 0.0) or 0.0)
        for seg in seg_iter:
            if should_stop is not None and should_stop():
                break
            text = (seg.text or "").strip()
            if text:
                text = _apply_fixes(text, word_map, roster)
                segments.append((float(seg.start), float(seg.end), text))
            if on_progress is not None and total > 0:
                # 留 1% 给收尾（写文件、刷新界面），别让进度条卡在 100% 等
                on_progress(min(99, int(float(seg.end) / total * 100)))
    if on_progress is not None:
        on_progress(100)
    return segments, total
