# -*- coding: utf-8 -*-
"""VTuber 实时翻译器 —— 程序入口。

流程：抓取电脑系统声音 → VAD 断句 → Whisper 日语识别 → 在线翻译 → 置顶悬浮字幕。
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_HERE, "run.log")


def _setup_logging() -> None:
    """把问题留在 run.log 里，而不是让它们无声无息地消失。

    - pythonw（无控制台）下 stdout/stderr 重定向进 run.log；
    - 追加模式 + 超 2MB 自动轮转成 run.log.old（以前每次启动清空，
      闪退现场一点痕迹都不留）；
    - faulthandler 挂钩原生崩溃（0xc0000005 访问冲突等）：下次再闪退，
      run.log 里会有崩溃时刻的 Python 调用栈，一眼定位是哪行触发的；
    - threading.excepthook：工作线程抛异常也进日志
      （以前线程死了都无声无息，字幕说没就没）。
    """
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 2 * 1024 * 1024:
            os.replace(LOG_PATH, LOG_PATH + ".old")
    except Exception:
        pass
    if sys.stdout is None or sys.stderr is None or \
            "pythonw" in os.path.basename(sys.executable).lower():
        try:
            f = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
            sys.stdout = f
            sys.stderr = f
        except Exception:
            pass

    def hook(*exc):
        traceback.print_exception(*exc, file=sys.stderr)
        try:
            sys.stderr.flush()
        except Exception:
            pass
    sys.excepthook = hook

    def thread_hook(args):
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback,
                                  file=sys.stderr)
        try:
            sys.stderr.flush()
        except Exception:
            pass
    threading.excepthook = thread_hook

    try:
        _fh = open(LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(file=_fh, all_threads=True)
    except Exception:
        pass


_setup_logging()

_SERVER = None


def _single_instance() -> bool:
    """只允许一个实例：重复双击会把已运行的窗口叫出来，而不是再起一个卡住的进程。"""
    global _SERVER
    try:
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
    except Exception:
        return True
    name = "VTuberTranslator_SingleInstance"
    sock = QLocalSocket()
    sock.connectToServer(name)
    if sock.waitForConnected(300):
        try:
            sock.write(b"show")
            sock.flush()
            sock.waitForBytesWritten(500)
        except Exception:
            pass
        sock.disconnectFromServer()
        return False
    QLocalServer.removeServer(name)
    _SERVER = QLocalServer()
    if not _SERVER.listen(name):
        # 锁没抢到也要继续跑（功能不能因此不可用），但日志里必须留痕：
        # 这种情况下再双击不会提示「已经在运行」，会起第二份副本。
        print(f"单实例锁获取失败：{_SERVER.errorString()}，仍继续启动", flush=True)
    return True

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import feedback
from core.audio import list_capture_devices, pick_default_device
from core.config import AppConfig
from core.pipeline import Pipeline
from core.transcribe import resplit, transcribe_media, translate_segments
from ui.overlay import SubtitleOverlay
from ui.panel import ControlPanel, _icon


class AppController:
    def __init__(self, app: QApplication):
        self.app = app
        self._quitting = False          # 退出收尾只跑一次（面板关闭/托盘退出可能同时触发）
        self._trs_busy = False          # 音频转文字进行中：与实时字幕互斥
        self._trs_cancel = threading.Event()
        # 唯一的写盘许可入口：真实应用启动。脚本/测试用默认的只读加载。
        self.cfg = AppConfig.load(interactive=True)
        # 标错写手随程序启动：后台线程只做追加写，面板按键只是入队
        self.feedback = feedback.FeedbackWriter()
        self.pipeline = Pipeline(self.cfg)
        self.overlay = SubtitleOverlay(self.cfg)
        self.panel = ControlPanel(self.cfg, self)

        self.pipeline.on_status = self.status
        self.pipeline.on_level = lambda v: self.panel.sig_level.emit(v)
        self.pipeline.on_result = self.on_result
        self.pipeline.on_partial = self.on_partial

        self._build_tray()
        self.overlay.apply_config()
        self.overlay.show()
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()
        if _SERVER is not None:
            _SERVER.newConnection.connect(self._on_another_instance)
        self.refresh_devices()
        self.panel.set_running(False)
        self.status("未启动 —— 点「开始翻译」即可")
        QTimer.singleShot(600, self._hello)
        # 模型预加载：面板一出来就在后台把模型加载好（冷加载约 15 秒），
        # 用户点「开始翻译」时立即就能开始，不用干等。
        self._prewarm()

    def _prewarm(self) -> None:
        def work():
            try:
                t0 = time.time()
                self.pipeline.asr.load()
                self.status(f"模型已预热：{self.pipeline.asr.model_size}"
                            f"（{self.pipeline.asr.dev_info}，用时 {time.time() - t0:.0f}s）"
                            f"—— 点「开始翻译」即刻可用")
            except Exception as exc:
                self.status(f"模型预热失败（点「开始翻译」会再试）：{exc}")
        threading.Thread(target=work, daemon=True).start()

    def _hello(self) -> None:
        """右下角弹一条提示，避免用户以为程序没启动。"""
        try:
            self.tray.showMessage(
                "VTuber 实时翻译器 已启动",
                "控制面板已打开，字幕窗在屏幕底部。\n点面板上的「开始翻译」即可开始。",
                QSystemTrayIcon.Information, 5000)
        except Exception:
            pass

    def _on_another_instance(self) -> None:
        """第二次双击启动时，把窗口重新叫到前台。"""
        try:
            while _SERVER.hasPendingConnections():
                sock = _SERVER.nextPendingConnection()
                sock.readyRead.connect(self.show_panel)
        except Exception:
            pass
        self.show_panel()

    # ------------------------------------------------------------ 回调
    def status(self, text: str) -> None:
        self.panel.sig_status.emit(text)
        self.panel.sig_log.emit(text)
        # 按钮状态跟着流水线真实状态走：模型/采集失败时流水线已经是 error，
        # 但 set_running 原来只在 toggle_start 里调用，面板会一直停在
        # 「停止翻译」——看起来在运行，实际字幕早就不出了。
        st = self.pipeline.state
        self.panel.sig_running.emit(st in ("loading", "running", "paused"), st == "paused")

    def on_partial(self, seq: int, src: str) -> None:
        """识别先出结果：字幕立刻动起来，译文随后补上。"""
        self.overlay.sig_src.emit(seq, src)
        # 同一出口补发实时记录信号：原文上屏时刻就是该句的起始时间
        self.panel.sig_live_partial.emit(seq, src)
        if src:
            self.panel.sig_log.emit(f"JA  {src}")

    def on_result(self, seq: int, src: str, tgt: str) -> None:
        self.overlay.sig_tgt.emit(seq, tgt)
        self.panel.sig_live_line.emit(seq, src, tgt)
        if tgt:
            self.panel.sig_log.emit(f"ZH  {tgt}")

    # ------------------------------------------------------------ 托盘
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(_icon(), self.app)
        self.tray.setToolTip("VTuber 实时翻译器")
        menu = QMenu()
        a_panel = QAction("显示控制面板", menu)
        a_panel.triggered.connect(self.show_panel)
        a_sub = QAction("显示/隐藏字幕", menu)
        a_sub.triggered.connect(self.toggle_subtitle)
        a_pin = QAction("鼠标穿透", menu)
        a_pin.setCheckable(True)
        a_pin.setChecked(bool(self.cfg.click_through))
        a_pin.triggered.connect(lambda: self.overlay.set_click_through(not self.cfg.click_through))
        a_pause = QAction("暂停/继续", menu)
        a_pause.triggered.connect(self.toggle_pause)
        a_clear = QAction("清空字幕", menu)
        a_clear.triggered.connect(lambda: self.overlay.sig_clear.emit())
        a_quit = QAction("退出", menu)
        a_quit.triggered.connect(self.quit)
        menu.addAction(a_panel)
        menu.addAction(a_sub)
        menu.addAction(a_pin)
        menu.addAction(a_pause)
        menu.addAction(a_clear)
        menu.addSeparator()
        menu.addAction(a_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self.show_panel() if r == QSystemTrayIcon.DoubleClick else None)
        self.tray.show()

    def show_panel(self) -> None:
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    # ------------------------------------------------------------ 控制动作
    def refresh_devices(self) -> None:
        def work():
            try:
                devs = list_capture_devices()
                cur = None
                if self.cfg.device:
                    from core.audio import AudioDevice
                    cur = AudioDevice.from_dict(self.cfg.device)
                if cur is None:
                    cur = pick_default_device()
                    if cur is not None:
                        self.cfg.device = cur.to_dict()
                        self.cfg.save()
                # 控件只能由主线程操作：工作线程通过信号投递，别直接调 panel.set_devices
                self.panel.sig_devices.emit(devs, cur)
                self.status(f"发现 {len(devs)} 个音频设备，当前：{cur.label if cur else '无'}")
            except Exception as exc:
                self.status(f"枚举音频设备失败：{exc}")
        threading.Thread(target=work, daemon=True).start()

    def on_device_selected(self, dev) -> None:
        self.cfg.device = dev.to_dict()
        self.cfg.save()
        self.pipeline.restart_audio()
        self.status(f"采集设备已切换：{dev.label}")

    def on_model_changed(self) -> None:
        self.pipeline.update_asr()
        if self.pipeline.state in ("running", "paused"):
            def reload():
                try:
                    self.pipeline.asr.load(on_progress=self.status)
                    self.status("模型已切换，正在聆听…")
                except Exception as exc:
                    self.status(f"模型切换失败：{exc}")
            threading.Thread(target=reload, daemon=True).start()

    def on_trans_changed(self) -> None:
        self.pipeline.update_translator()
        threading.Thread(target=self.pipeline.translator.probe, daemon=True).start()

    def on_vad_changed(self) -> None:
        self.pipeline.update_vad()

    def on_denoise_changed(self) -> None:
        self.pipeline.update_denoise()

    def on_vocal_changed(self) -> None:
        self.pipeline.update_vocal()

    def on_channel_check(self) -> None:
        """逐个试各翻译通道并把结果送到面板（后台线程跑，别卡住界面）。"""

        def work():
            try:
                res = self.pipeline.translator.health_check()
            except Exception as exc:
                self.panel.sig_channel_result.emit(f"健康检查异常：{type(exc).__name__}: {exc}")
                return
            proxy = (self.cfg.proxy or "").strip()
            head = f"探测句：こんにちは"
            head += f"    代理：{proxy or '未设置'}"
            if proxy:
                head += "（腾讯/彩云已绕过代理直连）"
            lines = [head, ""]
            ok_cnt = 0
            for name, ok, ms, info in res:
                if ok:
                    ok_cnt += 1
                lines.append(f"{'✓' if ok else '✗'} {name:<10} {ms:>5}ms  {info}")
            lines.append("")
            lines.append(f"可用 {ok_cnt}/{len(res)} 个通道")
            if ok_cnt == 0:
                lines.append("全部不可用 —— 字幕会只剩原文。常见原因：代理不通 / 彩云 token 失效。")
            self.status(f"通道健康检查完成：可用 {ok_cnt}/{len(res)}")
            self.panel.sig_channel_result.emit("\n".join(lines))

        threading.Thread(target=work, daemon=True).start()

    def on_beam_changed(self) -> None:
        self.pipeline.update_beam()

    def on_style_changed(self) -> None:
        self.cfg.save()
        self.overlay.apply_config()

    def detect_proxy(self) -> None:
        def work():
            self.status("正在探测本机代理端口…")
            found = self.pipeline.translator.detect_local_proxy()
            if found:
                self.cfg.proxy = found
                self.cfg.save()
                # 同上：探针线程不直接碰控件，交给主线程更新输入框
                self.panel.sig_proxy_text.emit(found)
                self.pipeline.translator.apply_proxy()
                self.status(f"发现代理 {found}，已启用；Google/Edge 通道可用了")
                self.pipeline.translator.probe()
            else:
                self.status("没发现可用的本地代理（免费通道仍可尝试 MyMemory）")
        threading.Thread(target=work, daemon=True).start()

    def test_translate(self) -> None:
        def work():
            out = self.pipeline.translator.translate("こんにちは、今日も配信を見てくれてありがとう！")
            self.status(f"翻译测试：{out or '失败（请检查网络或换通道）'}")
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ 音频转文字
    def transcribe_start(self, path: str, translate: bool = False,
                         max_chars: int = 0) -> None:
        """整段音视频转文字（后台线程）。

        与实时字幕互斥：两者共用 pipeline.asr 里的那一份模型，同时跑会让解码排队、
        显存也顶不住；所以运行中拒绝开始，转写中面板会锁住「开始翻译」。

        translate=True 时转写完接着逐句翻译（复用实时那套通道/主备/缓存）；
        max_chars>0 时先按最大字数重新断句，再翻译，保证两种产物句界一致。
        """
        if getattr(self.panel, "_mdl_busy", None):
            self.panel.sig_trs_fail.emit("正在下载/卸载模型，等它完成再开始转写。")
            return
        if self.pipeline.state in ("running", "paused", "loading"):
            self.panel.sig_trs_fail.emit("实时字幕正在运行，先停止翻译再转写。")
            return
        if self._trs_busy:
            return
        self._trs_busy = True
        self._trs_cancel = threading.Event()
        cancel = self._trs_cancel

        def work():
            try:
                self._transcribe_job(path, translate, max_chars, cancel)
            except Exception as exc:
                self.panel.sig_trs_fail.emit(f"转写失败：{type(exc).__name__}: {exc}")
            finally:
                self._trs_busy = False

        threading.Thread(target=work, daemon=True).start()

    def _transcribe_job(self, path: str, translate: bool, max_chars: int,
                        cancel: threading.Event) -> None:
        """转写一份本地音视频：加载模型 → ASR → 可选翻译 → 发 sig_trs_done。

        本地文件与 URL 下载后的音频共用这一段；取消靠 cancel.is_set。
        """
        asr = self.pipeline.asr
        self.panel.sig_trs_status.emit("正在加载模型…")
        asr.load(on_progress=self.panel.sig_trs_status.emit)
        # 显式降级：kotoba 原件的 alignment 层号越界，开字词时间码会直接崩进程。
        # 有校验通过的修复副本时 words_safe 会放开；副本缺失/损坏则回到降级。
        from core.asr import words_safe
        if not words_safe(asr.model_size, getattr(asr, "loaded_ref", "")):
            msg = (f"已降级：{asr.model_size} 不支持字词时间码"
                   "（alignment 层号超出解码器层数，ctranslate2 会直接崩进程），"
                   "本次只出句级时间轴；文本与导出不受影响。")
            self.panel.sig_trs_status.emit(msg)
            # 再进日志：完成时状态标签会被「转写完成」覆盖，日志里要留这条降级说明
            self.status(msg)
        self.status(f"开始转写：{os.path.basename(path)}")
        namefix = bool(self.cfg.enable_namefix)
        t0 = time.perf_counter()
        segments, total = transcribe_media(
            asr, path,
            # "" / auto 表示让 Whisper 自己判语种；kotoba 等日语专用档由
            # asr.resolve_language() 强制 ja（不能在这里直接传 None 绕开）
            language=asr.resolve_language(self.cfg.source_lang),
            word_map=self.cfg.word_map if namefix else "",
            roster=self.cfg.name_roster if namefix else "",
            # 离线多花一点时间换字词时间码：工程 JSON 与重新断句都要用
            word_timestamps=True,
            on_progress=self.panel.sig_trs_progress.emit,
            should_stop=cancel.is_set,
        )
        t_asr = time.perf_counter() - t0
        if max_chars and segments:
            n0 = len(segments)
            segments = resplit(segments, max_chars)
            self.panel.sig_trs_status.emit(
                f"已重新断句：{n0} → {len(segments)} 句（≤{max_chars} 字）")
        if translate and segments and not cancel.is_set():
            def on_tr(i: int, n: int) -> None:
                self.panel.sig_trs_progress.emit(int(i / max(1, n) * 100))
                if i % 5 == 0 or i == n:
                    self.panel.sig_trs_status.emit(f"正在翻译 {i}/{n} 句…")

            t1 = time.perf_counter()
            ok = translate_segments(
                self.pipeline.translator, segments,
                roster=self.cfg.name_roster if namefix else "",
                builtin_titles=bool(getattr(self.cfg, "builtin_titles", True)),
                on_progress=on_tr,
                should_stop=cancel.is_set,
            )
            self.panel.sig_trs_status.emit(
                f"翻译完成：{ok}/{len(segments)} 句，用时 {time.perf_counter() - t1:.1f}s")
        self.panel.sig_trs_done.emit((segments, total, cancel.is_set()))
        if cancel.is_set():
            self.status(f"已取消转写（保留 {len(segments)} 句）")
        else:
            self.status(f"转写完成：{len(segments)} 句，识别耗时 {t_asr:.1f}s")

    # ------------------------------------------------------------ 单视频 URL
    def url_probe(self, url: str) -> None:
        """只读探测：标题、时长、字幕语种；结果用非阻塞信息框呈现。"""
        url = (url or "").strip()
        if not url:
            self.panel.sig_trs_fail.emit("先粘贴一个视频链接。")
            return

        def work():
            from core import url_source as us
            try:
                info = us.probe(url, proxy=self.cfg.proxy)
                page = info["webpage_url"] or url
                lines = [info["title"] or "(无标题)",
                         "时长 %d 秒" % int(info["duration"]),
                         "站点：%s" % (page.split("//")[-1].split("/")[0])]
                if info.get("uploader"):
                    lines.append("UP 主：%s" % info["uploader"])
                subs = info["subtitles"]
                autos = info["automatic"]
                if subs:
                    lines.append("人工字幕：" + "、".join(
                        "%s(%s)" % (s["lang"], "/".join(s["formats"]) or "-") for s in subs))
                if autos:
                    lines.append("自动字幕：" + "、".join(
                        "%s(%s)" % (s["lang"], "/".join(s["formats"]) or "-") for s in autos))
                if not subs and not autos:
                    lines.append("未读到可用字幕：可能需要登录，或该视频没有字幕。")
                self.panel.sig_url_info.emit("\n".join(lines))
            except us.UrlError as exc:
                text = exc.message + (("\n\n" + exc.hint) if exc.hint else "")
                self.panel.sig_url_info.emit(text)
            except Exception as exc:
                self.panel.sig_url_info.emit("探测失败：%s: %s" % (type(exc).__name__, exc))

        threading.Thread(target=work, daemon=True).start()

    def url_transcribe_start(self, url: str, translate: bool = False,
                             max_chars: int = 0) -> None:
        """下载单个视频的音频到临时目录，转写完自动删除。"""
        url = (url or "").strip()
        if not url:
            self.panel.sig_trs_fail.emit("先粘贴一个视频链接。")
            return
        if self._trs_busy:
            return
        if self.pipeline.state in ("running", "paused", "loading"):
            self.panel.sig_trs_fail.emit("实时字幕正在运行，先停止翻译再转写。")
            return
        self._trs_busy = True
        self._trs_cancel = threading.Event()
        cancel = self._trs_cancel
        self.panel.sig_trs_status.emit("正在解析链接…")

        def work():
            import shutil
            import tempfile
            from core import url_source as us
            rt = os.path.join(os.path.dirname(__file__), ".runtime")
            os.makedirs(rt, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix="vttr_url_", dir=rt)
            job_started = False
            try:
                if not us.have_ytdlp():
                    raise us.UrlError("missing", "缺少可选依赖 yt-dlp", us.INSTALL_HINT)

                last = {"p": -10}

                def prog(pct: int, text: str) -> None:
                    if cancel.is_set():
                        raise us.Cancelled()
                    self.panel.sig_trs_progress.emit(pct)
                    if pct - last["p"] >= 10 or pct >= 100:
                        last["p"] = pct
                        self.panel.sig_trs_status.emit("下载音频 %d%%" % pct)

                path, info = us.download_audio(
                    url, tmp, proxy=self.cfg.proxy,
                    on_progress=prog, should_stop=cancel.is_set)
                title = (info.get("title") or info.get("id") or "video").strip()
                # 导出目录固定到「文档」，音频稍后删除也能继续导出结果
                docs = os.path.join(os.path.expanduser("~"), "Documents")
                safe = "".join(c for c in title if c not in '\\/:*?"<>|').strip() or "video"
                self.panel.sig_trs_source.emit(os.path.join(docs, safe + ".mp3"))
                self.panel.sig_trs_status.emit(
                    "音频下载完成（%d 秒），开始转写…" % int(info.get("duration") or 0))
                job_started = True
                self._transcribe_job(path, translate, max_chars, cancel)
            except us.UrlError as exc:
                if exc.kind == "cancel":
                    self.panel.sig_trs_done.emit(([], 0.0, True))
                else:
                    self.panel.sig_trs_fail.emit(
                        exc.message + (("\n" + exc.hint) if exc.hint else ""))
            except us.Cancelled:
                self.panel.sig_trs_done.emit(([], 0.0, True))
            except Exception as exc:
                self.panel.sig_trs_fail.emit("转写失败：%s: %s" % (type(exc).__name__, exc))
            finally:
                try:
                    shutil.rmtree(tmp, ignore_errors=True)
                except Exception:
                    pass
                self._trs_busy = False
                if cancel.is_set() and not job_started:
                    self.panel.sig_trs_status.emit("已取消，临时文件已清理。")

        threading.Thread(target=work, daemon=True).start()

    def url_subs_start(self, url: str) -> None:
        """直接取平台字幕生成文本/SRT（不做 ASR），只处理单视频。"""
        url = (url or "").strip()
        if not url:
            self.panel.sig_trs_fail.emit("先粘贴一个视频链接。")
            return
        if self._trs_busy:
            return
        if self.pipeline.state in ("running", "paused", "loading"):
            self.panel.sig_trs_fail.emit("实时字幕正在运行，先停止翻译再读字幕。")
            return
        self._trs_busy = True
        self.panel.sig_trs_status.emit("正在读取字幕…")

        def work():
            from core.transcribe import Cue
            from core import url_source as us
            try:
                if not us.have_ytdlp():
                    raise us.UrlError("missing", "缺少可选依赖 yt-dlp", us.INSTALL_HINT)
                folder = os.path.join(os.path.dirname(__file__), ".runtime", "url_subs")
                files = us.download_subtitles(url, folder, proxy=self.cfg.proxy)
                path = files[0]
                raw = us.read_srt(path)
                if not raw:
                    raise us.UrlError("subs", "字幕文件为空或格式无法解析",
                                      "可改用「下载并转写」走语音识别。")
                # 面板与导出统一按 Cue 取值（_on_trs_done 读 .start/.end/.text）
                segs = [Cue(a, b, t) for a, b, t in raw]
                total = segs[-1].end
                docs = os.path.join(os.path.expanduser("~"), "Documents")
                stem = os.path.splitext(os.path.basename(path))[0]
                self.panel.sig_trs_source.emit(os.path.join(docs, stem + ".srt"))
                self.panel.sig_trs_status.emit(
                    "已读取平台字幕 %d 条（非 ASR，可能有断句差异）" % len(segs))
                self.panel.sig_trs_done.emit((segs, total, False))
            except us.UrlError as exc:
                self.panel.sig_trs_fail.emit(
                    exc.message + (("\n" + exc.hint) if exc.hint else ""))
            except Exception as exc:
                self.panel.sig_trs_fail.emit("读取字幕失败：%s: %s"
                                             % (type(exc).__name__, exc))
            finally:
                self._trs_busy = False

        threading.Thread(target=work, daemon=True).start()

    def transcribe_cancel(self) -> None:
        if self._trs_busy:
            self._trs_cancel.set()
            self.panel.sig_trs_status.emit("正在取消…（当前这句转完就停）")

    def toggle_start(self) -> None:
        if getattr(self.panel, "_mdl_busy", None):
            self.status("正在下载/卸载模型，等它完成再开始实时翻译")
            return
        if self._trs_busy:
            self.status("正在转写文件，先点「转写」页的「取消」再开始实时翻译")
            return
        if self.pipeline.state in ("running", "paused", "loading"):
            self.pipeline.stop()
            self.panel.set_running(False)
            self.status("已停止")
        else:
            self.pipeline.start()
            self.panel.set_running(True)
            self.overlay.show()

    def toggle_pause(self) -> None:
        if self.pipeline.state == "running":
            self.pipeline.set_paused(True)
            self.panel.set_running(True, True)
        elif self.pipeline.state == "paused":
            self.pipeline.set_paused(False)
            self.panel.set_running(True, False)

    def toggle_subtitle(self) -> None:
        vis = not self.overlay.isVisible()
        self.overlay.setVisible(vis)
        self.panel.set_subtitle_visible(vis)

    def cleanup(self) -> None:
        """退出收尾（可重复调用）：先隐藏窗口，再停流水线、放锁。

        顺序很重要——pipeline.stop() 最坏要等音频线程 join 2 秒，
        若先停再藏，字幕窗会在这段时间里继续留在桌面上。
        """
        try:
            self.overlay.hide()
            self.panel.hide()
            self.tray.hide()
        except Exception:
            pass
        self.pipeline.stop()
        # 退出前把标错队列写完（含 fsync），别让最后几次标记停在内存里
        try:
            self.feedback.close()
        except Exception:
            pass
        # 全局热键是系统级资源，退出必须显式注销（closeEvent 之外再兜一层）
        try:
            getattr(self, "panel", None).shutdown_hotkey()
        except Exception:
            pass
        # 主动释放单实例名：别等进程彻底结束才放手，
        # 否则收尾期间再双击会误判「已经在运行」
        if _SERVER is not None:
            _SERVER.close()

    def quit(self) -> None:
        """真正退出：保存布局 → 隐藏全部窗口 → 停采集/流水线 → 放锁 → 退出事件循环。"""
        if self._quitting:
            return
        self._quitting = True
        self.cfg.geometry = {"x": self.overlay.x(), "y": self.overlay.y(),
                             "w": self.overlay.width(), "h": self.overlay.height()}
        self.cfg.window_width = self.overlay.width()
        self.cfg.save()
        self.cleanup()
        print("==== VTuber 实时翻译器退出 ====", flush=True)
        self.app.quit()


def main() -> int:
    print(f"==== VTuber 实时翻译器启动 {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"python {sys.version.split()[0]} ====", flush=True)
    # Qt6 默认已启用高 DPI 缩放，不需要再设 AA_EnableHighDpiScaling（该属性已废弃）
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.warning(None, "提示", "系统托盘不可用：关掉控制面板就等于退出程序。")

    if not _single_instance():
        print("==== 已有实例在运行，本次启动已取消 ====", flush=True)
        QMessageBox.information(
            None, "VTuber 实时翻译器",
            "程序已经在运行了。\n\n"
            "· 已经帮你把控制面板叫到前台\n"
            "· 字幕窗在屏幕底部（可能被别的窗口挡住，看一眼）\n"
            "· 或在屏幕右下角托盘里找它的图标\n\n"
            "如果主界面确实找不到了，用任务管理器结束 pythonw.exe 后再启动。")
        return 0

    try:
        ctrl = AppController(app)
    except Exception:
        traceback.print_exc()
        msg = "启动失败，详情见 run.log：\n\n" + traceback.format_exc(limit=3)
        QMessageBox.critical(None, "VTuber 实时翻译器 启动失败", msg)
        return 1
    # 兜底收尾：无论从面板关闭、托盘退出还是系统关机触发，
    # 都保证窗口先隐藏、单实例名被释放，不把字幕窗留在桌面上
    app.aboutToQuit.connect(ctrl.cleanup)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
