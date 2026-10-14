# -*- coding: utf-8 -*-
"""识别-翻译流水线：采集 → VAD 分段 → Whisper 识别 → 在线翻译 → 回调结果。

三个工作线程串联（采集 / 识别 / 翻译），任何一环卡顿都不会堵死上一环，
队列满了丢弃最旧的数据以保证字幕是「当下」的内容。
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from .audio import AudioCapture, AudioDevice, TARGET_SR, pick_default_device
from .asr import WhisperASR
from .config import AppConfig
from .translator import Translator
from .vad import SpeechSegmenter
from .denoise import Denoise


class Pipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.on_status: Callable[[str], None] = lambda s: None
        self.on_result: Callable[[int, str, str], None] = lambda seq, s, t: None
        # 识别一出结果就回调（译文还没到）：字幕先动起来，别干等网络
        self.on_partial: Callable[[int, str], None] = lambda seq, s: None
        self.on_level: Callable[[float], None] = lambda v: None

        self.state = "stopped"          # stopped | loading | running | paused | error
        self.capture: Optional[AudioCapture] = None
        self.vad = SpeechSegmenter(sample_rate=TARGET_SR, sensitivity=cfg.sensitivity,
                                   min_speech_ms=cfg.min_speech_ms, min_silence_ms=cfg.min_silence_ms,
                                   max_speech_ms=cfg.max_speech_ms)
        self.asr = WhisperASR(cfg.model_size, cfg.asr_device, cfg.compute_type, cfg.source_lang)
        self._sync_beam()
        self.translator = Translator(cfg, on_log=self.on_status)
        # 采集端降噪：在送进 VAD 之前先压一压稳态背景噪声（风扇/电流声）
        self.denoise = Denoise(enable=cfg.enable_denoise, strength=cfg.denoise_strength)
        self.translator.apply_proxy()

        # 队列放宽：快语速时不会因为一时的积压就把整句丢掉
        self._asr_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
        self._tr_q: "queue.Queue[tuple]" = queue.Queue(maxsize=32)
        # 单段识别耗时的滑动均值（秒）。_asr_q 按「还能积压几秒的识别工作量」动态限流，
        # 而不是死守 32 条 —— 慢模型按 32 条排会让字幕滞后一两分钟。
        self._asr_lat = 0.3
        # 并行翻译后按识别顺序上屏，避免字幕颠倒
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._next_seq = 0
        self._pending: dict = {}
        self._threads: list = []
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()

    # ------------------------------------------------------------ 控制
    def start(self) -> None:
        if self.state in ("running", "loading", "paused"):
            return
        self._stop.clear()
        self._pause.set()
        self.state = "loading"
        self.on_status("正在准备模型…")
        t = threading.Thread(target=self._boot, daemon=True)
        t.start()

    def _boot(self) -> None:
        try:
            # 不再用 gpu_ok 记住「上次 GPU 失败」而直接锁死 CPU：
            # 失败多半是显存被游戏临时占满，重启后显存空了就该回 GPU。
            # load() 内部按显存余量自动选档并逐级降级，GPU 真不可用时
            # 几秒内就会落到可用组合，不会像以前那样白等。
            self.asr.load(on_progress=self.on_status)
            if self.asr.dev_info.startswith("cpu") and getattr(self.cfg, "gpu_ok", True):
                self.cfg.gpu_ok = False
                try:
                    self.cfg.save()
                except Exception:
                    pass
        except Exception as exc:
            self.state = "error"
            self.on_status(f"模型加载失败：{exc}（首次需联网下载，可检查网络或换用更小模型）")
            return
        def probe_all():
            if self.cfg.auto_proxy and not self.cfg.proxy:
                found = self.translator.detect_local_proxy()
                if found:
                    self.cfg.proxy = found
                    self.cfg.save()
                    self.translator.apply_proxy()
                    self.on_status(f"发现本机代理 {found}，Google/Edge 通道已解锁")
            self.translator.probe()
        threading.Thread(target=probe_all, daemon=True).start()
        try:
            dev = AudioDevice.from_dict(self.cfg.device) if self.cfg.device else None
            # 保存的设备当时是「系统默认播放设备的回环」→ 默认设备可能已经换了
            # （插耳机/拔音箱/切蓝牙）。跟着系统默认走，避免「声音从耳机出、
            # 却还在抓扬声器」——那会表现为有声音但一个字都不出
            # （WASAPI 回环对没在播放的设备根本不产生回调）。
            if dev is None or dev.is_default:
                cur = pick_default_device()
                if cur is not None and (dev is None or (cur.name, cur.backend) != (dev.name, dev.backend)):
                    if dev is not None:
                        self.on_status(f"默认声音设备已切换为「{cur.name}」，采集已自动跟随")
                    self.cfg.device = cur.to_dict()
                    try:
                        self.cfg.save()
                    except Exception:
                        pass
                dev = cur if cur is not None else dev
            self.capture = self._new_capture(dev)
            self.capture.start()
            time.sleep(0.4)
            if self.capture.error:
                self.state = "error"
                self.on_status(f"音频采集失败：{self.capture.error}")
                return
        except Exception as exc:
            self.state = "error"
            self.on_status(f"音频采集失败：{exc}")
            return

        self.state = "running"
        self._cap_restarts = 0
        self.on_status(f"正在聆听…（系统声音 · {self.asr.model_size} {self.asr.dev_info}）")
        # 采集 1 路、识别 1 路（GPU 很快）、翻译 2 路并发。
        # 并发别开太高：免费 MyMemory 会限流，开 4 路反而把请求拖到十几秒。
        for fn, n in ((self._capture_loop, 1), (self._asr_loop, 1), (self._trans_loop, 2)):
            for _ in range(n):
                th = threading.Thread(target=fn, daemon=True)
                th.start()
                self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.state = "stopped"
        self.vad.reset()
        for q in (self._asr_q, self._tr_q):
            with q.mutex:
                q.queue.clear()
        with self._seq_lock:
            self._pending.clear()
            self._seq = 0
            self._next_seq = 0
        self._threads = []

    def set_paused(self, paused: bool) -> None:
        if self.state not in ("running", "paused"):
            return
        if paused:
            self._pause.clear()
            if self.capture:
                self.capture.stop()
            self.vad.reset()
            self.state = "paused"
            self.on_status("已暂停")
        else:
            self._pause.set()
            if self.capture:
                try:
                    self.capture.start()
                except Exception as exc:
                    self.on_status(f"恢复采集失败：{exc}")
                    return
            self.state = "running"
            self.on_status("正在聆听…（系统声音）")

    def restart_audio(self) -> None:
        """切换设备后重新采集，不影响已加载的模型。"""
        if self.state not in ("running", "paused"):
            return
        try:
            if self.capture:
                self.capture.stop()
            dev = AudioDevice.from_dict(self.cfg.device) if self.cfg.device else None
            self.capture = self._new_capture(dev)
            if self.state == "running":
                self.capture.start()
        except Exception as exc:
            self.on_status(f"切换音频设备失败：{exc}")

    def _default_device_changed(self) -> bool:
        """系统默认回环设备是否已不等于当前采集设备（用于区分「断流」与「空闲静音」）。

        枚举设备有开销（要开 PyAudio），所以内部节流到 2s 一次；
        异常一律当作「没变」（宁可不重启，也不要因枚举失败打断采集）。
        """
        now = time.time()
        if now - getattr(self, "_dev_check_ts", 0.0) < 2.0:
            return False
        self._dev_check_ts = now
        try:
            cur = pick_default_device()
        except Exception:
            return False
        if cur is None or self.capture is None:
            return False
        return cur.name != getattr(self.capture.device, "name", "")

    def _restart_capture(self, reason: str) -> None:
        """采集流静默死亡后的自动恢复：先试原设备，不行就换系统默认回环设备。"""
        self.on_status(f"{reason}，正在自动恢复采集…")
        try:
            if self.capture:
                self.capture.stop()
        except Exception:
            pass
        try:
            dev = AudioDevice.from_dict(self.cfg.device) if self.cfg.device else None
            new = self._new_capture(dev)
            new.start()
            time.sleep(0.5)
            if new.error:
                new.stop()
                d = pick_default_device()
                if d is None:
                    self.on_status("自动恢复失败：找不到可用的采集设备，请手动刷新设备列表")
                    return
                new = self._new_capture(d)
                self.cfg.device = d.to_dict()
                try:
                    self.cfg.save()
                except Exception:
                    pass
                new.start()
                time.sleep(0.5)
                if new.error:
                    self.on_status(f"自动恢复失败：{new.error}")
                    return
            self.capture = new
            self.on_status("采集已自动恢复")
        except Exception as exc:
            self.on_status(f"自动恢复采集异常：{exc}")

    def update_vad(self) -> None:
        self.vad.update_params(sensitivity=self.cfg.sensitivity, min_speech_ms=self.cfg.min_speech_ms,
                               min_silence_ms=self.cfg.min_silence_ms, max_speech_ms=self.cfg.max_speech_ms)

    def update_translator(self) -> None:
        self.translator.apply_proxy()
        # 用户改了翻译设置：清掉「已确认不通」的通道，给它们一次重新尝试的机会
        self.translator._active = None
        self.translator._dead.clear()
        self.translator.clear_cache()

    def update_asr(self) -> None:
        self.asr.model_size = self.cfg.model_size
        self.asr.device = self.cfg.asr_device
        self.asr.compute_type = self.cfg.compute_type
        self.asr.language = self.cfg.source_lang
        self._sync_beam()

    def _sync_beam(self) -> None:
        """large-v* 模型较重，默认用最快解码宽度（beam=1）保实时性；其余模型保持
        默认 2 兼顾识别率。仅在用户没在面板手动改过（beam_auto）时才自动降档。"""
        if getattr(self.cfg, "beam_auto", True) and self.cfg.model_size.startswith("large"):
            self.cfg.beam_size = 1
        try:
            self.asr.beam_size = int(self.cfg.beam_size)
        except Exception:
            pass

    def update_beam(self) -> None:
        """改解码宽度无需重载模型，直接热更新（面板拖动即时生效）。"""
        try:
            self.asr.beam_size = int(self.cfg.beam_size)
        except Exception:
            pass

    def _new_capture(self, dev) -> AudioCapture:
        """统一构造采集实例：带上采集期参数（人声提取开关/强度）。"""
        return AudioCapture(
            dev,
            vocal_extract=bool(getattr(self.cfg, "vocal_extract", True)),
            vocal_strength=float(getattr(self.cfg, "vocal_strength", 0.8)),
        )

    def update_denoise(self) -> None:
        self.denoise.enable = bool(self.cfg.enable_denoise)
        self.denoise.strength = float(self.cfg.denoise_strength)

    def update_vocal(self) -> None:
        """人声提取开关热更新（面板改动下一块音频即生效，不用重启）。

        `_pre_mono` 每个回调都读这两个属性，且 `_center` 是惰性创建的，
        所以直接赋值即可；关掉时走普通降混，已有实例留着不碍事。
        """
        c = self.capture
        if c is not None:
            c.vocal_extract = bool(getattr(self.cfg, "vocal_extract", True))
            c.vocal_strength = float(getattr(self.cfg, "vocal_strength", 0.8))

    # ------------------------------------------------------------ 工作线程
    def _capture_loop(self) -> None:
        last_level_post = 0.0
        started = time.time()
        peak = 0.0
        warned = False
        while not self._stop.is_set():
            self._pause.wait()
            if self._stop.is_set() or self.capture is None:
                break
            chunk = self.capture.read(timeout=0.3)
            # 采集看门狗：切默认设备/插拔耳机/VMware 虚拟声卡上下线时，
            # WASAPI 回环流会「静默死亡」——不报错、不再有回调，永远没声音。
            # 中途断流（曾收到过数据）就自动重启采集；带退避和次数上限防死循环。
            # 注意：**空闲静音时 WASAPI 常常干脆不派发回调**，这不算断流——
            # 若无脑重启，会在「没人说话」时白重启一次、并把状态栏刷脏。
            # 因此只有「系统默认回环设备换了」才立刻重启；设备没换就先当空闲等，
            # 超过 IDLE_GRACE 秒仍无数据，才按真断流处理。
            IDLE_GRACE = 25.0
            cap = self.capture                 # 本地取一次：另一线程的 stop() 会把它置 None
            stale = (time.time() - cap.last_data_ts) if cap is not None else 0.0
            if (chunk is None and cap is not None and cap.running and cap.ever_data
                    and stale > 8.0
                    and time.time() - started > 10.0
                    and getattr(self, "_cap_restarts", 0) < 3):
                if not self._default_device_changed():
                    if stale < IDLE_GRACE:
                        continue          # 判定为空闲静音，继续等，不重启
                    reason = "采集长时间无数据（>25s），已重启"
                else:
                    reason = "采集似乎断开（可能切换了声音设备）"
                self._cap_restarts = getattr(self, "_cap_restarts", 0) + 1
                self._restart_capture(reason)
                continue
            if chunk is None:
                continue
            if self.denoise.enable:
                chunk = self.denoise.process(chunk)
                if chunk is None or len(chunk) == 0:
                    continue
            seg, level = self.vad.feed(chunk)
            # 抓了十几秒还是死寂，多半是选错设备（比如实际用耳机却抓了扬声器）
            peak = max(peak, level)
            if not warned and time.time() - started > 15 and peak < 0.002:
                warned = True
                self.on_status("一直没抓到声音 —— 请在「采集设备」里换一个 [系统声音] 设备试试"
                               "（你现在出声的可能是耳机或外接音箱）")
            now = time.time()
            if now - last_level_post > 0.08:
                last_level_post = now
                try:
                    self.on_level(min(1.0, level * 6.0))
                except Exception:
                    pass
            if seg is not None and len(seg) > TARGET_SR * 0.2:
                self._put_asr(seg)
        tail = self.vad.flush()
        if tail is not None and len(tail) > TARGET_SR * 0.3:
            self._put_asr(tail)

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            try:
                seg = self._asr_q.get(timeout=0.5)
            except Exception:
                continue
            if self._stop.is_set():
                break
            # 碎片段合并：VAD 偶尔会把一句话剁出 <0.45s 的碎段（语气词/吸气/单拍词），
            # 碎段单独送识别必出「诶、那个」式碎片字幕，而且短段的 ASR 固定开销占比极高
            # （实测 0.72s 音频要识别 0.74s，RTF≈1）。稍等下一段拼上（20ms 静音垫
            # 防止词粘连），拼到 5s 上限为止；等不到就按原样送。
            if getattr(self.cfg, "merge_fragments", True) and 0 < len(seg) < TARGET_SR * 0.45:
                try:
                    nxt = self._asr_q.get(timeout=0.25)
                except Exception:
                    nxt = None
                if nxt is not None and len(nxt) > 0:
                    pad = np.zeros(TARGET_SR // 50, dtype=np.float32)
                    seg = np.concatenate([seg, pad, nxt])
                    while len(seg) < TARGET_SR * 5.0 and len(seg) < TARGET_SR * 0.45:
                        try:
                            nxt2 = self._asr_q.get(timeout=0.25)
                        except Exception:
                            break
                        if nxt2 is None or len(nxt2) == 0:
                            break
                        seg = np.concatenate([seg, pad, nxt2])
            t0 = time.time()
            roster = self.cfg.name_roster if getattr(self.cfg, "enable_namefix", True) else ""
            hotwords = ""
            if getattr(self.cfg, "enable_hotwords", True):
                # 热词 = 用户词表 + 内置游戏/番剧标题，**汉字/片假名原样**下发
                # （2026-10-13 实测：用假名读音反而把解码带偏，CER 7.1%→26.1%）。
                from .namefix import hotword_text
                hotwords = hotword_text(roster,
                                        builtin=bool(getattr(self.cfg, "builtin_titles", True)))
            text = self.asr.transcribe(seg, word_map=self.cfg.word_map,
                                       roster=roster, hotwords=hotwords)
            if text and hotwords:
                # prompt 回声护栏：解码器偶尔把热词表当内容吐出来，宁可少一句也不上屏词表
                from .namefix import prompt_echo
                if prompt_echo(text, hotwords):
                    self.on_status("已丢弃一段疑似「热词复述」的输出")
                    continue
            # 用实测的单段耗时动态决定 _asr_q 能积压几段（见 _put_asr）
            try:
                el = time.time() - t0
                if el > 0:
                    self._asr_lat = 0.7 * self._asr_lat + 0.3 * min(el, 30.0)
            except Exception:
                pass
            if not text:
                # 识别异常要让用户看见，不能闷声不出字
                if self.asr.last_error:
                    self.on_status(f"识别异常：{self.asr.last_error}")
                    self.asr.last_error = ""
                continue
            with self._seq_lock:
                seq = self._seq
                self._seq += 1
            # 先上屏原文：实测翻译还要 0.2~0.8 秒，画面空着等这段时间最难受
            try:
                self.on_partial(seq, text)
            except Exception:
                pass
            # 送去翻译的文本可以做「标题术语替换」（エルデンリング→艾尔登法环）：
            # 机翻不认识游戏名（实测「原神→上帝」「エルデンリング→埃尔金戒指」），
            # 先把中文名塞进源文里，机翻就会照抄。上屏的原文保持识别原样，所以带两份。
            tr_text = text
            if getattr(self.cfg, "enable_title_glossary", True):
                try:
                    from .namefix import apply_title_glossary
                    tr_text = apply_title_glossary(
                        text, self.cfg.name_roster or "",
                        builtin=bool(getattr(self.cfg, "builtin_titles", True)))
                except Exception:
                    tr_text = text
            self._put_keep_newest(self._tr_q, (seq, tr_text, text))

    @staticmethod
    def _put_keep_newest(q: "queue.Queue", item) -> None:
        """队列满时丢最旧的、保最新的：宁可跳掉积压的旧内容，也要让字幕跟得上当下。"""
        if q.full():
            try:
                q.get_nowait()
            except Exception:
                pass
        q.put_nowait(item)

    # 允许积压的识别工作量（秒）。超出就把最旧的丢掉，保证字幕跟得上当下。
    ASR_PENDING_SEC = 3.0

    def _put_asr(self, seg) -> None:
        """按「还能积压多少秒的识别工作量」限流，而不是固定条数。

        large-v3 配大 beam 时单段要 3~8 秒，若仍按 32 条排队，正在识别的永远是
        一两分钟前说的话 —— 表现就是「字幕极慢」。所以慢模型只保留 1~2 段待识别，
        宁可跳句也要把延迟压住；快模型（small/medium + beam 小）则允许十几条的突发缓冲。
        """
        try:
            # 上限 24 是给极快模型兜底：别让限流值超过队列容量 32，否则会变成丢新段
            limit = max(1, min(int(self.ASR_PENDING_SEC / max(self._asr_lat, 0.05)), 24))
        except Exception:
            limit = 1
        try:
            while self._asr_q.qsize() >= limit:
                self._asr_q.get_nowait()
        except Exception:
            pass
        try:
            self._asr_q.put_nowait(seg)
        except Exception:
            try:
                self._asr_q.get_nowait()
                self._asr_q.put_nowait(seg)
            except Exception:
                pass

    def _trans_loop(self) -> None:
        """多个翻译线程并行跑（网络请求是瓶颈），但输出仍按识别顺序排好。"""
        while not self._stop.is_set():
            try:
                item = self._tr_q.get(timeout=0.5)
            except Exception:
                continue
            if self._stop.is_set():
                break
            seq, tr_text, disp_text = item
            # 翻译必须兜住异常：这个线程一旦被异常打断就再也不会出字幕了
            try:
                out = self.translator.translate(tr_text)
            except Exception as exc:
                out = ""
                self.on_status(f"翻译异常：{type(exc).__name__}")
            with self._seq_lock:
                self._pending[seq] = (disp_text, out or "")
            self._flush_ordered()

    def _flush_ordered(self) -> None:
        """并行翻译会乱序返回，这里按识别顺序依次上屏，保证字幕不颠倒。"""
        while True:
            with self._seq_lock:
                item = self._pending.pop(self._next_seq, None)
                if item is None:
                    break
                seq = self._next_seq
                self._next_seq += 1
            try:
                # 翻译失败也要把原文顶上去，别让字幕一片空白
                self.on_result(seq, item[0], item[1])
            except Exception:
                pass
