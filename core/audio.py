# -*- coding: utf-8 -*-
"""音频采集：优先抓取电脑系统声音（WASAPI Loopback），可回退到麦克风。

对外统一接口：
    devices = list_capture_devices()          # 枚举可用采集设备
    cap = AudioCapture(device=..., blocksize=1024)
    cap.start(); data = cap.read(); cap.stop()
其中 data 为 float32 单声道 numpy 数组（已重采样到 16kHz）。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

TARGET_SR = 16000


# ---------------------------------------------------------------- 重采样
def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """把音频重采样到 target_sr。整数倍抽取时先做均值抗混叠。"""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    audio = audio.astype(np.float32, copy=False)
    ratio = orig_sr / target_sr
    if abs(ratio - round(ratio)) < 1e-6:
        r = int(round(ratio))
        n = len(audio) // r * r
        if n > 0:
            return audio[:n].reshape(-1, r).mean(axis=1).astype(np.float32)
        return np.zeros(0, dtype=np.float32)
    duration = len(audio) / float(orig_sr)
    n = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def to_mono(data: np.ndarray) -> np.ndarray:
    """(n, ch) -> (n,)，多声道取均值。"""
    if data.ndim == 2:
        if data.shape[1] == 1:
            return data[:, 0].astype(np.float32, copy=False)
        return data.mean(axis=1).astype(np.float32)
    return data.astype(np.float32, copy=False)


# ---------------------------------------------------------------- 设备描述
@dataclass
class AudioDevice:
    index: Optional[int]
    name: str
    api: str
    kind: str          # 'loopback' | 'input' | 'output'
    samplerate: int
    channels: int
    backend: str       # 'sounddevice' | 'pyaudiowpatch'
    is_default: bool = False   # 是否为系统默认播放设备对应的回环

    @property
    def label(self) -> str:
        tag = {"loopback": "[系统声音]", "input": "[麦克风]", "output": "[扬声器]"}[self.kind]
        star = "★ " if self.is_default else ""
        return f"{star}{tag} {self.name}"

    def to_dict(self) -> Dict:
        return {
            "index": self.index, "name": self.name, "api": self.api, "kind": self.kind,
            "samplerate": self.samplerate, "channels": self.channels, "backend": self.backend,
            "is_default": self.is_default,
        }

    @staticmethod
    def from_dict(d: Dict) -> Optional["AudioDevice"]:
        if not d:
            return None
        return AudioDevice(
            index=d.get("index"), name=d.get("name", ""), api=d.get("api", ""),
            kind=d.get("kind", "input"), samplerate=int(d.get("samplerate") or TARGET_SR),
            channels=int(d.get("channels") or 2), backend=d.get("backend", "sounddevice"),
            is_default=bool(d.get("is_default", False)),
        )


def _sd_devices() -> List[AudioDevice]:
    try:
        import sounddevice as sd
    except Exception:
        return []
    out: List[AudioDevice] = []
    try:
        apis = {i: a.get("name", "") for i, a in enumerate(sd.query_hostapis())}
        for i, d in enumerate(sd.query_devices()):
            api = apis.get(d.get("hostapi"), "")
            name = d.get("name", "") or ""
            max_in = int(d.get("max_input_channels") or 0)
            max_out = int(d.get("max_output_channels") or 0)
            sr = int(d.get("default_samplerate") or 48000)
            low = name.lower()
            # PortAudio 为 WASAPI 输出设备生成的回环设备，名字里通常带 Loopback
            if "loopback" in low and max_in > 0:
                out.append(AudioDevice(i, name, api, "loopback", sr, min(2, max_in), "sounddevice"))
            elif max_in > 0:
                out.append(AudioDevice(i, name, api, "input", sr, min(2, max_in), "sounddevice"))
            elif max_out > 0 and "wasapi" in api.lower():
                out.append(AudioDevice(i, name, api, "output", sr, min(2, max_out), "sounddevice"))
    except Exception:
        pass
    return out


_PW = {"mod": None, "pa": None}


def _get_pyaudio():
    """全局复用一个 PyAudio 实例：反复初始化/terminate 在部分环境下会崩溃。"""
    if _PW["pa"] is None:
        import pyaudiowpatch as pyaudio
        _PW["mod"] = pyaudio
        _PW["pa"] = pyaudio.PyAudio()
    return _PW["mod"], _PW["pa"]


def _pw_devices() -> List[AudioDevice]:
    """用 pyaudiowpatch 枚举 WASAPI 回环设备（打了 loopback patch 的 PortAudio）。"""
    out: List[AudioDevice] = []
    try:
        _pyaudio, p = _get_pyaudio()
    except Exception:
        return out
    try:
        # 系统默认播放设备，用来给回环设备打星标，避免默认选到没声音的 HDMI/DP 口
        default_name = ""
        try:
            default_name = p.get_default_output_device_info().get("name", "") or ""
        except Exception:
            pass
        for d in p.get_loopback_device_info_generator():
            name = d.get("name", "") or ""
            base = name.replace("[Loopback]", "").strip()
            is_def = bool(default_name) and (base in default_name or default_name in base)
            out.append(AudioDevice(
                index=int(d.get("index")), name=name, api="WASAPI",
                kind="loopback", samplerate=int(d.get("defaultSampleRate") or 48000),
                channels=min(2, int(d.get("maxInputChannels") or 2)),
                backend="pyaudiowpatch", is_default=is_def,
            ))
    except Exception:
        pass
    return out


def list_capture_devices() -> List[AudioDevice]:
    """返回去重后的采集设备列表，系统声音优先排在前面。"""
    devices: List[AudioDevice] = []
    seen = set()
    for dev in _sd_devices() + _pw_devices():
        key = (dev.backend, dev.kind, dev.name)
        if key in seen:
            continue
        seen.add(key)
        devices.append(dev)
    # 已经有真正能抓系统声音的回环设备时，丢掉那些只能输出、无法采集的扬声器条目
    if any(d.kind == "loopback" for d in devices):
        devices = [d for d in devices if d.kind != "output"]

    devices.sort(key=lambda d: (0 if d.kind == "loopback" else 1 if d.kind == "output" else 2,
                                0 if d.is_default else 1,
                                0 if d.api and "wasapi" in d.api.lower() else 1, d.name))
    return devices


def pick_default_device() -> Optional[AudioDevice]:
    devs = list_capture_devices()
    for dev in devs:
        if dev.kind == "loopback" and dev.is_default:
            return dev
    for dev in devs:
        if dev.kind == "loopback":
            return dev
    for dev in devs:
        if dev.kind == "output":
            return dev
    for dev in devs:
        if dev.kind == "input":
            return dev
    return None


# ---------------------------------------------------------------- 采集器
class AudioCapture:
    """以 callback/阻塞读取两种方式采集音频，输出 16k 单声道 float32 数据块。"""

    def __init__(self, device: Optional[AudioDevice] = None, blocksize: int = 1600,
                 vocal_extract: bool = False, vocal_strength: float = 0.8):
        self.device = device or pick_default_device()
        self.blocksize = blocksize
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
        self._running = False
        self._stream = None
        self._pa = None
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        # 人声提取（中置声道提取）：回环是立体声，人声多半钉在正中、BGM 常被做宽，
        # 在**降混之前**用 core/vocals.py 把宽声场成分压低再送 VAD/识别。
        # 对本来就正中的内容逐样本无损（可测），只有立体声设备才生效。
        self.vocal_extract = bool(vocal_extract)
        self.vocal_strength = float(vocal_strength)
        self._center = None
        # 采集看门狗用：最近一次收到回调数据的时刻；ever_data 区分
        # 「中途断流」（该自动重启）和「一直没声音」（该提示用户换设备）。
        self.last_data_ts = time.time()
        self.ever_data = False

    # -- 生命周期
    def start(self) -> None:
        if self._running:
            return
        if self.device is None:
            raise RuntimeError("没有找到可用的音频设备")
        self._running = True
        self.error = None
        if self.device.backend == "pyaudiowpatch":
            self._thread = threading.Thread(target=self._pw_loop, daemon=True)
            self._thread.start()
        else:
            self._thread = threading.Thread(target=self._sd_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """只置停止标志并等采集线程自己收尾。

        流的 stop/close 必须由「拥有它的采集线程」执行：PortAudio 的音频回调
        可能正在另一个系统线程里跑，从别的线程直接 close 有竞态 ——
        Windows 事件日志里实测出现过 _portaudiowpatch.pyd 0xc0000005 访问冲突
        直接把整个进程带走（pythonw 闪退）。这里先让回调自然退出
        （_running=False 后回调返回 paAbort），再 join 线程，最多 2 秒。
        """
        self._running = False
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        # 线程已把流关掉（正常路径）。异常残留（线程死在半路）才由本线程兜底关一次。
        st, self._stream = self._stream, None
        if st is not None:
            try:
                st.stop_stream()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass
        self._thread = None
        # 不调用 PyAudio.terminate()：pyaudiowpatch 在部分环境下 terminate 会直接崩进程，
        # PyAudio 实例改为全局复用，随进程退出自然释放。
        self._pa = None
        with self._q.mutex:
            self._q.queue.clear()

    @property
    def running(self) -> bool:
        return self._running

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except Exception:
            return None

    def _push(self, mono: np.ndarray, sr: int) -> None:
        self.last_data_ts = time.time()
        self.ever_data = True
        # 人声提取模块首块会返回空（等待凑满一帧），别把空数组塞进队列/重采样
        if mono is None or len(mono) == 0:
            return
        if sr != TARGET_SR:
            mono = resample(mono, sr, TARGET_SR)
        if self._q.full():
            try:
                self._q.get_nowait()   # 丢弃最旧数据，保证实时性
            except Exception:
                pass
        self._q.put_nowait(mono)

    # -- sounddevice 后端
    def _sd_loop(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = f"sounddevice 不可用: {exc}"
            self._running = False
            return
        dev = self.device
        idx = dev.index if dev.index is not None else None
        last_err = None
        try:
            for sr in (None, dev.samplerate, 48000, 44100, TARGET_SR):
                if not self._running:
                    break
                try:
                    kwargs = dict(samplerate=sr, blocksize=self.blocksize, dtype="float32",
                                  channels=dev.channels, callback=self._sd_callback, latency="low")
                    if idx is not None:
                        kwargs["device"] = idx
                    self._stream = sd.InputStream(**kwargs)
                    self._stream.start()
                    while self._running:
                        time.sleep(0.2)
                    return
                except Exception as exc:      # 换采样率重试
                    last_err = exc
                    try:
                        if self._stream is not None:
                            self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
            self.error = f"打开音频设备失败: {last_err}"
        finally:
            # 流由本线程收尾（见 stop() 里的说明：跨线程 close 有竞态会闪退）
            st, self._stream = self._stream, None
            if st is not None:
                try:
                    st.stop()
                except Exception:
                    pass
                try:
                    st.close()
                except Exception:
                    pass
            self._running = False

    def _pre_mono(self, arr: np.ndarray) -> np.ndarray:
        """把回调拿到的原始帧转成单声道（可选先做中置人声提取）。

        立体声设备且开启 vocal_extract 时，先按「中置度」压低宽声场的 BGM，
        再交给 VAD/识别；否则退化为普通降混。单声道设备直接降混。
        任何异常都退回普通降混——这个回调在音频线程里跑，绝不能抛异常。
        """
        try:
            if self.vocal_extract and arr.ndim == 2 and arr.shape[1] >= 2:
                if self._center is None:
                    from .vocals import CenterExtractor
                    self._center = CenterExtractor(strength=self.vocal_strength)
                out = self._center.process(arr[:, 0], arr[:, 1])
                if out is not None and len(out) > 0:
                    return out
                return np.zeros(0, dtype=np.float32)
        except Exception:
            self._center = None
        return to_mono(arr)

    def _sd_callback(self, indata, frames, time_info, status):
        if not self._running:
            return
        try:
            self._push(self._pre_mono(np.asarray(indata, dtype=np.float32).copy()), TARGET_SR)
        except Exception:
            pass

    # -- pyaudiowpatch 后端（回调模式）
    def _pw_loop(self) -> None:
        try:
            pyaudio, pa = _get_pyaudio()
            import time as _t
        except Exception as exc:
            self.error = f"pyaudiowpatch 不可用: {exc}"
            self._running = False
            return
        dev = self.device
        try:
            self._pa = pa
            self._pa_mod = pyaudio
            self._pa_sr = int(dev.samplerate or 48000)
            self._pa_ch = int(dev.channels or 2)
            sr = self._pa_sr
            try:
                self._stream = self._pa.open(
                    format=pyaudio.paInt16, channels=dev.channels, rate=sr,
                    input=True, frames_per_buffer=self.blocksize,
                    input_device_index=dev.index, stream_callback=self._pw_cb, start=False,
                )
            except Exception:
                sr = self._pa_sr = 48000
                self._stream = self._pa.open(
                    format=pyaudio.paInt16, channels=dev.channels, rate=sr,
                    input=True, frames_per_buffer=self.blocksize,
                    input_device_index=dev.index, stream_callback=self._pw_cb, start=False,
                )
            self._stream.start_stream()
            while self._running:
                _t.sleep(0.2)
        except Exception as exc:
            self.error = f"打开音频设备失败: {exc}"
            self._running = False
        finally:
            # 流的 stop/close 必须在「拥有它的本线程」里做：PortAudio 回调可能正在
            # 系统音频线程里执行，从别的线程 close 会踩出 0xc0000005 访问冲突
            # （事件日志实测，pythonw 闪退的直接凶手）。先 stop 等回调排空再 close。
            st, self._stream = self._stream, None
            if st is not None:
                try:
                    st.stop_stream()
                except Exception:
                    pass
                try:
                    st.close()
                except Exception:
                    pass

    def _pw_cb(self, in_data, frame_count, time_info, status):
        mod = getattr(self, "_pa_mod", None)
        if not self._running or mod is None:
            return (None, mod.paAbort if mod else 1)
        try:
            arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            ch = getattr(self, "_pa_ch", 2)
            if ch > 1 and arr.size >= ch:
                arr = arr[: arr.size // ch * ch].reshape(-1, ch)
                arr = self._pre_mono(arr)      # 立体声：可选做中置人声提取
            self._push(arr, getattr(self, "_pa_sr", 48000))
        except Exception:
            pass
        return (None, mod.paContinue)
