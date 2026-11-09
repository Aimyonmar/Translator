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
from .context import ContextBuffer, fix_homophones, fix_translation
from .translator import Translator
from .vad import SpeechSegmenter
from .denoise import Denoise
from . import feedback
from . import timing


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
        # 被队列丢弃的序号：上屏时必须跳过它们，否则 _next_seq 会卡在缺口上，
        # 之后所有译文都堆在 _pending 里再也不上屏（表现为字幕彻底停住）。
        self._skipped: set = set()
        self._threads: list = []
        # 停止令牌：每次 start() 换一个新的，旧令牌立刻置位。
        # 若共用同一个 Event，stop()→start() 之间的 clear() 会让上一轮线程「复活」，
        # 出现两套采集/识别/翻译线程同时跑（重复字幕、CPU/GPU 翻倍）。
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        # 语境窗口：最近几句上屏内容，供同音词纠正与翻译上下文使用。
        # 容量固定 8（配置上限），每次按 cfg.context_window 取用。
        self.context = ContextBuffer(maxlen=8)
        # 标错用最近句快照：只存内存、只在上屏/译文回来时各更新一次，
        # 关闭 enable_feedback 后一次都不写（见 core/feedback.py）。
        self.recent = feedback.RecentSentences(feedback.RECENT_MAX)
        # 全量记录写手（回看用，不等于判错）：首次需要时才建，关闭开关时恒为 None
        self._live = None
        self.session_id = ""
        self._session_t0 = time.time()
        # 链路计数：与 bench 的 finals_n / trans_fail 同口径，都按「on_result 上屏一次」记。
        # 只做只读统计，不参与任何翻译/识别决策；关掉标错也照常记（它统计的是链路本身）。
        # 需要锁只是因为翻译线程有 2 条，自增本身是纳秒级，不进任何阻塞路径。
        self._count_lock = threading.Lock()
        self._counts = {"final": 0, "tgt": 0, "fail": 0, "hedge": 0,
                        "drop_tr": 0, "drop_asr": 0, "skip": 0}

    # ------------------------------------------------------------ 控制
    def start(self) -> None:
        if self.state in ("running", "loading", "paused"):
            return
        # 换新令牌：旧线程只认自己启动时拿到的那个，不会被这次 start 复活
        old = self._stop
        if old is not None:
            old.set()
        self._stop = threading.Event()
        stop_evt = self._stop
        self._pause.set()
        # 清掉上一轮可能残留的序号状态：老线程在停止瞬间写进来的 pending
        # 不能污染新一轮的上屏顺序（否则会在新序号走到同一数字时冒出旧字幕）
        with self._seq_lock:
            self._pending.clear()
            self._skipped.clear()
            self._seq = 0
            self._next_seq = 0
        # 新一场直播：上一场的句子不能当这一场的语境
        self.context.clear()
        # 标错会话：每次开始换新 id，旧句子不再出现在可标记窗口里
        self.session_id = feedback.new_session_id("live")
        self._session_t0 = time.time()
        self.recent.clear()
        with self._count_lock:
            self._counts = {"final": 0, "tgt": 0, "fail": 0, "hedge": 0,
                            "drop_tr": 0, "drop_asr": 0, "skip": 0}
        # 上一轮 stop 抑制过保活；新一轮允许它自启（是否起线程仍由开关决定）
        self.translator.resume_keepalive()
        self.state = "loading"
        self.on_status("正在准备模型…")
        threading.Thread(target=self._boot, args=(stop_evt,), daemon=True).start()

    def _boot(self, stop_evt: threading.Event) -> None:
        def aborted() -> bool:
            """停止请求已下达：收尾刚建好的采集，保持 stopped，不再改回 running。"""
            if not stop_evt.is_set():
                return False
            # 已有新一轮运行时（self._stop 换了），这轮只负责退出，
            # 绝不能去动新一轮的 capture/state
            if self._stop is stop_evt:
                cap, self.capture = self.capture, None
                if cap is not None:
                    try:
                        cap.stop()
                    except Exception:
                        pass
                self.state = "stopped"
            return True

        try:
            # 不再用 gpu_ok 记住「上次 GPU 失败」而直接锁死 CPU：
            # 失败多半是显存被游戏临时占满，重启后显存空了就该回 GPU。
            # load() 内部按显存余量自动选档并逐级降级，GPU 真不可用时
            # 几秒内就会落到可用组合，不会像以前那样白等。
            self.asr.load(on_progress=self.on_status)
            # 冷启动加载要十几秒，这期间用户可能已经点了停止
            if aborted():
                return
            if self.asr.dev_info.startswith("cpu") and getattr(self.cfg, "gpu_ok", True):
                self.cfg.gpu_ok = False
                try:
                    self.cfg.save()
                except Exception:
                    pass
        except Exception as exc:
            if aborted():
                return
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
            cap = self._new_capture(dev)
            cap.start()
            time.sleep(0.4)
            # 采集起来后再确认归属：若期间已经 stop() 或已被新一轮接管，
            # 这个流由本轮自己关掉，不能写进 self.capture 去顶掉新一轮的采集
            if stop_evt.is_set() or self._stop is not stop_evt:
                cap.stop()
                if self._stop is stop_evt:
                    self.state = "stopped"
                return
            self.capture = cap
            if cap.error:
                self.state = "error"
                self.on_status(f"音频采集失败：{cap.error}")
                return
        except Exception as exc:
            if aborted():
                return
            self.state = "error"
            self.on_status(f"音频采集失败：{exc}")
            return

        if stop_evt.is_set():
            return
        self.state = "running"
        self._cap_restarts = 0
        self.on_status(f"正在聆听…（系统声音 · {self.asr.model_size} {self.asr.dev_info}）")
        # 采集 1 路、识别 1 路（GPU 很快）、翻译 2 路并发。
        # 并发别开太高：免费 MyMemory 会限流，开 4 路反而把请求拖到十几秒。
        for fn, n in ((self._capture_loop, 1), (self._asr_loop, 1), (self._trans_loop, 2)):
            for _ in range(n):
                th = threading.Thread(target=fn, args=(stop_evt,), daemon=True)
                th.start()
                self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()
        # 程序停止后不再发保活 ping；线程在 Event.wait 上立即返回
        self.translator.stop_keepalive()
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
            self._skipped.clear()
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

    # ------------------------------------------------------------ 标错
    def recent_sentence(self, back: int = 0) -> Optional[dict]:
        """最近一句（back=0）或倒数第 back+1 句的快照；没有则 None。"""
        try:
            return self.recent.get(back)
        except Exception:
            return None

    def session_start_wall(self) -> float:
        """本轮开始时的墙钟时间，用来把快照里的相对秒换成时间戳。"""
        return self._session_t0

    def counts(self) -> dict:
        """本场链路计数快照：上屏 / 有译文 / 翻译失败（原文兜底）/ 丢弃来源。只读。

        丢弃相关：drop_tr = 翻译队列满丢旧；drop_asr = 识别队列限流丢弃的段；
        skip = _skip_seq 登记作废序号的次数（当前与 drop_tr 同源，用于对账）。
        """
        with self._count_lock:
            return dict(self._counts)

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

    def _restart_capture(self, reason: str, stop_evt: threading.Event) -> None:
        """采集流静默死亡后的自动恢复：先试原设备，不行就换系统默认回环设备。"""
        if stop_evt.is_set():
            return
        self.on_status(f"{reason}，正在自动恢复采集…")
        try:
            if self.capture:
                self.capture.stop()
        except Exception:
            pass
        try:
            dev = AudioDevice.from_dict(self.cfg.device) if self.cfg.device else None
            # 保存的是「跟随系统默认」的回环时，必须先重新挑当前默认设备：
            # 插耳机/切蓝牙后默认输出已经变了，旧扬声器回环虽然还能打开，
            # 但那边没有声音、永远不会有回调，一直重开它就会「有声音、没字幕」。
            # 手动选定的具体设备（is_default=False）保持不动，尊重用户选择。
            if dev is None or dev.is_default:
                cur = pick_default_device()
                if cur is not None and (dev is None or (cur.name, cur.backend) != (dev.name, dev.backend)):
                    if dev is not None:
                        self.on_status(f"默认声音设备已切换为「{cur.name}」，采集已自动跟随")
                    dev = cur
                    self.cfg.device = cur.to_dict()
                    try:
                        self.cfg.save()
                    except Exception:
                        pass
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
            if stop_evt.is_set():      # 恢复期间用户已停止：别把新采集留着空转
                new.stop()
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
    def _capture_loop(self, stop_evt: threading.Event) -> None:
        last_level_post = 0.0
        started = time.time()
        peak = 0.0
        warned = False
        while not stop_evt.is_set():
            self._pause.wait()
            if stop_evt.is_set() or self.capture is None:
                break
            chunk = self.capture.read(timeout=0.3)
            if stop_evt.is_set():
                break
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
                self._restart_capture(reason, stop_evt)
                # 换设备后按新设备重新计时：新设备仍然静音或打不开时，要能再提示一次
                started = time.time()
                peak = 0.0
                warned = False
                continue
            # 「一次回调都没有」不是静音，而是流根本没数据：典型场景是插了耳机、
            # 声音从耳机出却还绑在旧扬声器回环上（WASAPI 对没在播放的设备不派发回调），
            # 或设备被独占/禁用、采样率打不开。这种情况必须明确说出来，
            # 否则界面永远停在「正在聆听…」，用户会以为程序坏了。
            if (cap is not None and cap.running and not cap.ever_data
                    and time.time() - started > 12.0 and not warned):
                warned = True
                if cap.error:
                    self.on_status(f"音频采集不可用：{cap.error}（请在「采集设备」里刷新后改选）")
                else:
                    self.on_status("采集设备一直没有数据 —— 若刚插了耳机/切了蓝牙，"
                                   "请在「采集设备」里刷新并改选耳机对应的 [系统声音]")
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
                self._put_asr((seg, dict(self.vad.last_cut)))
        tail = self.vad.flush()
        if tail is not None and len(tail) > TARGET_SR * 0.3:
            self._put_asr((tail, dict(self.vad.last_cut)))

    def _asr_loop(self, stop_evt: threading.Event) -> None:
        while not stop_evt.is_set():
            try:
                item = self._asr_q.get(timeout=0.5)
            except Exception:
                continue
            if stop_evt.is_set():
                break
            seg, meta = self._unpack_seg(item)
            merged = 1
            # 碎片段合并：VAD 偶尔会把一句话剁出 <0.45s 的碎段（语气词/吸气/单拍词），
            # 碎段单独送识别必出「诶、那个」式碎片字幕，而且短段的 ASR 固定开销占比极高
            # （实测 0.72s 音频要识别 0.74s，RTF≈1）。稍等下一段拼上（20ms 静音垫
            # 防止词粘连），拼到 5s 上限为止；等不到就按原样送。
            if getattr(self.cfg, "merge_fragments", True) and 0 < len(seg) < TARGET_SR * 0.45:
                try:
                    nxt = self._asr_q.get(timeout=0.25)
                except Exception:
                    nxt = None
                nxt_audio, nxt_meta = self._unpack_seg(nxt)
                if nxt_audio is not None and len(nxt_audio) > 0:
                    pad = np.zeros(TARGET_SR // 50, dtype=np.float32)
                    seg = np.concatenate([seg, pad, nxt_audio])
                    # 合并后这句话真正「结束」的位置是最后一段的断句时刻，
                    # 计时锚点也随之换成最后一段的（见 core/timing.py）
                    meta = nxt_meta or meta
                    merged += 1
                    while len(seg) < TARGET_SR * 5.0 and len(seg) < TARGET_SR * 0.45:
                        try:
                            nxt2 = self._asr_q.get(timeout=0.25)
                        except Exception:
                            break
                        nxt2_audio, nxt2_meta = self._unpack_seg(nxt2)
                        if nxt2_audio is None or len(nxt2_audio) == 0:
                            break
                        seg = np.concatenate([seg, pad, nxt2_audio])
                        meta = nxt2_meta or meta
                        merged += 1
            t_asr_start = time.perf_counter()
            t0 = time.time()
            roster = ""
            if getattr(self.cfg, "enable_namefix", True):
                # 用户词表 + 内置标题（用户优先）：kotoba 不能开热词，
                # 这里是 エルジェンリング→エルデンリング 这类错名的唯一纠错通道。
                from .namefix import merge_roster
                roster = merge_roster(self.cfg.name_roster or "")
            hotwords = ""
            if getattr(self.cfg, "enable_hotwords", True):
                # 热词 = 用户词表 + 内置游戏/番剧标题，**汉字/片假名原样**下发
                # （2026-10-13 实测：用假名读音反而把解码带偏，CER 7.1%→26.1%）。
                # 这里只传用户词表，内置由 hotword_text 自己加，避免与 roster 重复。
                from .namefix import hotword_text
                hotwords = hotword_text(self.cfg.name_roster or "",
                                        builtin=bool(getattr(self.cfg, "builtin_titles", True)))
            text = self.asr.transcribe(seg, word_map=self.cfg.word_map,
                                       roster=roster, hotwords=hotwords)
            t_asr_end = time.perf_counter()
            # 识别是最长的一段耗时：停止请求若发生在识别途中，这里必须及时收手，
            # 否则识别完还会占用新序号、往已清空的队列里塞旧内容
            if stop_evt.is_set():
                break
            if text and hotwords:
                # prompt 回声护栏：解码器偶尔把热词表当内容吐出来，宁可少一句也不上屏词表
                from .namefix import prompt_echo
                if prompt_echo(text, hotwords):
                    self.on_status("已丢弃一段疑似「热词复述」的输出")
                    continue
            # 语境同音词纠正（変身→返信）：只有「含错形 + 有线索 + 无反向线索」才改，
            # 每次改动都上报，便于发现改错后关掉 enable_homophone_fix
            if text and getattr(self.cfg, "enable_homophone_fix", True):
                try:
                    text, fixes = fix_homophones(
                        text, self.context.sources(int(getattr(self.cfg, "context_window", 3))))
                    for note in fixes:
                        self.on_status(f"语境纠错：{note}")
                except Exception:
                    pass
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
            # 分段计时锚点：断句 → 排队 → 识别 → 上屏原文（后续译文再记）
            try:
                meta.update({
                    "audio_s": len(seg) / TARGET_SR,
                    "merged": merged,
                    "queue_ms": (t_asr_start - meta.get("cut_ts", t_asr_start)) * 1000.0,
                    "asr_ms": (t_asr_end - t_asr_start) * 1000.0,
                    "asr_end_ts": t_asr_end,
                })
            except Exception:
                meta = {}
            self._emit_text(text, meta)

    # ------------------------------------------------------ 上屏
    def _emit_text(self, text: str, meta: dict) -> None:
        """给一段识别结果分配序号、上屏原文、再投递给翻译线程。"""
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        # 先上屏原文：实测翻译还要 0.2~0.8 秒，画面空着等这段时间最难受
        try:
            self.on_partial(seq, text)
        except Exception:
            pass
        if getattr(self.cfg, "enable_feedback", True):
            try:
                self.recent.add_text(seq, text, time.time() - self._session_t0)
            except Exception:
                pass
        if getattr(self.cfg, "enable_live_log", False):
            try:
                # 会话内起始秒：上屏时就固定，译文回来后仍用这个起点
                meta["log_start_s"] = round(time.time() - self._session_t0, 3)
            except Exception:
                pass
        try:
            meta["partial_ts"] = time.perf_counter()
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
        self._put_keep_newest(self._tr_q, (seq, tr_text, text, meta), on_drop=self._skip_seq)

    @staticmethod
    def _unpack_seg(item):
        """(音频, 计时元数据) → 两者；兼容直接塞 ndarray 的旧调用。"""
        if isinstance(item, tuple):
            return item[0], item[1]
        return item, {}

    def _put_keep_newest(self, q: "queue.Queue", item, on_drop=None) -> None:
        """队列满时丢最旧的、保最新的：宁可跳掉积压的旧内容，也要让字幕跟得上当下。

        `on_drop` 会收到被丢掉的那一项：上屏层必须知道这个序号已经作废，
        否则有序输出会一直等它（见 _flush_ordered）。
        """
        if q.full():
            try:
                old = q.get_nowait()
                with self._count_lock:
                    self._counts["drop_tr"] += 1
                if on_drop is not None:
                    on_drop(old)
            except Exception:
                pass
        q.put_nowait(item)

    def _skip_seq(self, item) -> None:
        """登记一个不会再产出译文的序号，让 _next_seq 能跨过它继续上屏。"""
        try:
            seq = item[0]
        except Exception:
            return
        with self._count_lock:
            self._counts["skip"] += 1
        with self._seq_lock:
            if seq >= self._next_seq:
                self._skipped.add(seq)
        try:
            text = item[2] if len(item) > 2 else ""
        except Exception:
            text = ""
        self.on_status(f"丢弃 #{seq}「{str(text)[:10]}」——翻译队列已满（上限 {self._tr_q.maxsize}），丢最旧")

    def _live_log(self, seq: int, src: str, tgt: str, meta: dict) -> None:
        """全量记录：只入队，落盘在 feedback.LiveLogWriter 的后台线程。

        开关关闭时直接返回；开启后 submit 是微秒级，队列满则丢弃并计数，
        不会反压字幕线程。记录只用于回看，不代表判错。
        """
        if not getattr(self.cfg, "enable_live_log", False):
            return
        try:
            if self._live is None:
                self._live = feedback.LiveLogWriter(
                    keep_days=int(getattr(self.cfg, "live_log_keep_days",
                                          feedback.LIVE_KEEP_DAYS)),
                    max_mb=int(getattr(self.cfg, "live_log_max_mb",
                                       feedback.LIVE_MAX_MB)))
            self._live.submit(feedback.live_entry(
                cfg=self.cfg, session_id=self.session_id, seq=seq,
                src=src, tgt=tgt,
                channel=meta.get("channel") or self.translator.last_channel(),
                start_s=meta.get("log_start_s"),
                end_s=time.time() - self._session_t0,
                pipeline=self))
        except Exception:
            pass

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
        dropped = 0
        try:
            while self._asr_q.qsize() >= limit:
                self._asr_q.get_nowait()
                dropped += 1
        except Exception:
            pass
        if dropped:
            with self._count_lock:
                self._counts["drop_asr"] += dropped
            self.on_status(f"丢弃 {dropped} 段待识别音频——识别积压超过上限 {limit} 段")
        try:
            self._asr_q.put_nowait(seg)
        except Exception:
            try:
                self._asr_q.get_nowait()
                self._asr_q.put_nowait(seg)
            except Exception:
                pass

    def _trans_loop(self, stop_evt: threading.Event) -> None:
        """多个翻译线程并行跑（网络请求是瓶颈），但输出仍按识别顺序排好。"""
        while not stop_evt.is_set():
            try:
                item = self._tr_q.get(timeout=0.5)
            except Exception:
                continue
            if stop_evt.is_set():
                break
            seq, tr_text, disp_text, meta = item
            # 带上最近几句做语境：只对自定义 LLM 通道生效，免费通道仍是单句
            ctx = ()
            if getattr(self.cfg, "enable_context", True):
                try:
                    ctx = self.context.snapshot(int(getattr(self.cfg, "context_window", 3)))
                except Exception:
                    ctx = ()
            # 翻译必须兜住异常：这个线程一旦被异常打断就再也不会出字幕了
            t_tr_start = time.perf_counter()
            req0 = getattr(self.translator, "req_count", 0)
            try:
                out = self.translator.translate(tr_text, context=ctx)
            except Exception as exc:
                out = ""
                self.on_status(f"翻译异常：{type(exc).__name__}")
            finally:
                t_tr_end = time.perf_counter()
            meta["req"] = getattr(self.translator, "req_count", 0) - req0
            # 通道在 translate 返回后立刻取：last_channel() 是线程内的，
            # 两个翻译线程并发时，只有这里取才是这一句自己的通道
            meta["channel"] = self.translator.last_channel()
            # 主备赛跑结果（是否备胎救回/晚到毫秒/胜出通道）：线程内取值，只做上报
            meta["hedge"] = self.translator.last_hedge()
            if stop_evt.is_set():      # 停止后别再往新一轮的状态里写旧序号
                break
            with self._seq_lock:
                self._pending[seq] = (disp_text, out or "", meta, t_tr_start, t_tr_end)
            self._flush_ordered()

    def _flush_ordered(self) -> None:
        """并行翻译会乱序返回，这里按识别顺序依次上屏，保证字幕不颠倒。

        队列满被丢弃的序号登记在 _skipped 里，这里先跳过它们再取下一个，
        否则一个缺口就会把后面所有句子永久卡在 _pending 里。
        """
        while True:
            with self._seq_lock:
                while self._next_seq in self._skipped:
                    self._skipped.discard(self._next_seq)
                    self._next_seq += 1
                item = self._pending.pop(self._next_seq, None)
                if item is None:
                    break
                seq = self._next_seq
                self._next_seq += 1
            # 译文兜底：原文已认对（如「返信」）但译文仍出现固定错译（「变身」）时改回
            disp_text, out, meta, t_tr_start, t_tr_end = item
            t_screen = time.perf_counter()
            if out and getattr(self.cfg, "enable_homophone_fix", True):
                try:
                    out, fixes = fix_translation(disp_text, out)
                    for note in fixes:
                        self.on_status(f"译文纠错：{note}")
                except Exception:
                    pass
            # 分段计时（默认关闭，TRANSLATOR_TIMING=1 才写日志）：
            # total_ms 从「最后一个有声音的样本」算起，是用户真正感受到的延迟。
            if timing.enabled():
                try:
                    end_ts = meta.get("audio_end_ts") or t_screen
                    partial_ts = meta.get("partial_ts") or t_screen
                    asr_ts = meta.get("asr_end_ts") or t_tr_start
                    timing.record(
                        seq=seq,
                        cut=meta.get("cut", ""),
                        merged=meta.get("merged", 1),
                        audio_s=meta.get("audio_s", 0.0),
                        vad_tail_ms=meta.get("tail_ms", 0.0),
                        vad_wall_tail_ms=meta.get("wall_tail_ms", 0.0),
                        queue_ms=meta.get("queue_ms", 0.0),
                        asr_ms=meta.get("asr_ms", 0.0),
                        partial_ms=(partial_ts - end_ts) * 1000.0,
                        trans_wait_ms=(t_tr_start - asr_ts) * 1000.0,
                        trans_ms=(t_tr_end - t_tr_start) * 1000.0,
                        order_ms=(t_screen - t_tr_end) * 1000.0,
                        total_ms=(t_screen - end_ts) * 1000.0,
                        chars=len(disp_text),
                        trans_ok=1 if out else 0,
                        req=meta.get("req", 1),
                        channel=self.translator.last_channel(),
                    )
                except Exception:
                    pass
            # 计数先于回调：面板在 on_result 里就会刷新顶栏，先记账才不会读到上一句的数字。
            # 口径与 bench 一致（每次 on_result 记一条，译文为空记 fail）；hedge 只在
            # 备胎胜出且译文可用时记一次，关闭 hedge 时永远是 0。
            hedge = meta.get("hedge") or {}
            with self._count_lock:
                self._counts["final"] += 1
                self._counts["tgt" if out else "fail"] += 1
                if out and hedge.get("rescued"):
                    self._counts["hedge"] += 1
            try:
                # 翻译失败也要把原文顶上去，别让字幕一片空白
                self.on_result(seq, disp_text, out)
            except Exception:
                pass
            if getattr(self.cfg, "enable_feedback", True):
                try:
                    self.recent.set_result(seq, disp_text, out or "",
                                           meta.get("channel", ""),
                                           time.time() - self._session_t0)
                except Exception:
                    pass
            self._live_log(seq, disp_text, out or "", meta)
            # 只在真正上屏后入窗：并行翻译回包是乱序的，按上屏顺序记才不会错位
            try:
                self.context.push(disp_text, out)
            except Exception:
                pass
