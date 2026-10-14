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
    _SERVER.listen(name)
    return True

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QMessageBox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.audio import list_capture_devices, pick_default_device
from core.config import AppConfig
from core.pipeline import Pipeline
from ui.overlay import SubtitleOverlay
from ui.panel import ControlPanel, _icon


class AppController:
    def __init__(self, app: QApplication):
        self.app = app
        self.cfg = AppConfig.load()
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

    def on_partial(self, seq: int, src: str) -> None:
        """识别先出结果：字幕立刻动起来，译文随后补上。"""
        self.overlay.sig_src.emit(seq, src)
        if src:
            self.panel.sig_log.emit(f"JA  {src}")

    def on_result(self, seq: int, src: str, tgt: str) -> None:
        self.overlay.sig_tgt.emit(seq, tgt)
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

    def on_panel_hidden(self) -> None:
        self.tray.showMessage("VTuber 实时翻译器", "已最小化到托盘，双击图标可重新打开",
                              QSystemTrayIcon.Information, 2500)

    def show_panel(self) -> None:
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    # ------------------------------------------------------------ 控制动作
    def refresh_devices(self) -> None:
        def work():
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
            self.panel.set_devices(devs, cur)
            self.status(f"发现 {len(devs)} 个音频设备，当前：{cur.label if cur else '无'}")
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
                self.panel.txt_proxy.setText(found)
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

    def toggle_start(self) -> None:
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

    def quit(self) -> None:
        self.pipeline.stop()
        self.cfg.geometry = {"x": self.overlay.x(), "y": self.overlay.y(),
                             "w": self.overlay.width(), "h": self.overlay.height()}
        self.cfg.window_width = self.overlay.width()
        self.cfg.save()
        self.tray.hide()
        self.app.quit()


def main() -> int:
    print(f"==== VTuber 实时翻译器启动 {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"python {sys.version.split()[0]} ====", flush=True)
    # Qt6 默认已启用高 DPI 缩放，不需要再设 AA_EnableHighDpiScaling（该属性已废弃）
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.warning(None, "提示", "系统托盘不可用，程序将在关闭窗口后继续在后台运行。")

    if not _single_instance():
        QMessageBox.information(
            None, "VTuber 实时翻译器",
            "程序已经在运行了。\n\n"
            "· 字幕窗在屏幕底部（可能被别的窗口挡住，看一眼）\n"
            "· 或在屏幕右下角托盘里找它的图标\n\n"
            "如果完全没反应，关掉它再用「调试模式启动.bat」看报错。")
        return 0

    try:
        ctrl = AppController(app)
    except Exception:
        traceback.print_exc()
        msg = "启动失败，详情见 run.log：\n\n" + traceback.format_exc(limit=3)
        QMessageBox.critical(None, "VTuber 实时翻译器 启动失败", msg)
        return 1
    app.aboutToQuit.connect(lambda: ctrl.pipeline.stop())
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
