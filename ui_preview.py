# -*- coding: utf-8 -*-
"""离屏渲染界面预览图：不启动音频采集与模型，只把面板/字幕窗画出来存成 PNG。

用法：python ui_preview.py [输出目录]
默认输出到 docs/images/，用于「视觉改动前后」的可视证据。

注意：这里只用 QWidget.grab() 渲染，窗口全程不 show()，不会弹到屏幕上；
但要用系统平台插件（offscreen 插件没有字体库，中文会全部画成方块）。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from core.config import AppConfig
from ui.overlay import SubtitleOverlay
from ui.panel import ControlPanel


class _Vad:
    """给面板的诊断行喂一组固定数值，避免预览时去碰真实管线。"""
    last_level = 0.183
    threshold = 0.062


class _Pipeline:
    vad = _Vad()


class _FakeController:
    """只为构造面板：所有回调都空转，不启动任何后台线程。"""
    pipeline = _Pipeline()

    def __getattr__(self, name):
        return lambda *a, **k: None


def _grab(widget, w: int, h: int, scroll_bottom: bool = False) -> QPixmap:
    # 不 show()：给定尺寸做一次布局，然后直接渲染进 QPixmap
    widget.ensurePolished()
    widget.resize(w, h)
    widget.layout().activate()
    QApplication.processEvents()
    if scroll_bottom:
        sb = widget.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
        QApplication.processEvents()
    return widget.grab()


def _fill(panel: "ControlPanel", running: bool) -> None:
    if running:
        panel.set_running(True)
        panel.lbl_state.setText("正在聆听…")
        panel.sig_level.emit(0.62)
        panel._log("发现 3 个音频设备，当前：扬声器 (Realtek) 回环")
        panel._log("模型已预热：large-v3（cuda / float16，用时 12s）—— 点「开始翻译」即刻可用")
        panel._log("JA  今日は新しいゲームをやっていきます！")
        panel._log("ZH  今天我们要来玩一款新游戏！")
        panel._log("JA  このボスの第二形態、攻撃が全然読めなくて本当に難しいんだよね…")
        panel._log("ZH  这个 BOSS 的第二形态，攻击完全看不出来，真的很难啊……")
        panel._log("通道健康检查完成：可用 3/4")
        panel._show_channel_result(
            "探测句：こんにちは    代理：http://127.0.0.1:7890\n\n"
            "✓ tencent       212ms  正常\n"
            "✓ caiyun        348ms  正常\n"
            "✓ google        690ms  正常\n"
            "✗ deepl         超时   连接失败（代理不通？）\n\n"
            "可用 3/4 个通道")
    else:
        panel.set_running(False)
        panel.lbl_state.setText("未启动")
        panel._log("设备枚举完成：扬声器 (Realtek) 回环")
        panel._log("点「开始翻译」即可开始，字幕会出现在屏幕底部")


def _panel(cfg, running: bool) -> ControlPanel:
    panel = ControlPanel(cfg, _FakeController())
    _fill(panel, running)
    return panel


def _overlay(cfg) -> QPixmap:
    ov = SubtitleOverlay(cfg)
    ov.resize(1000, 200)
    ov.sig_src.emit(1, "今日は新しいゲームをやっていきます！")
    ov.sig_tgt.emit(1, "今天我们要来玩一款新游戏！")
    ov.sig_src.emit(2, "このボスの第二形態、攻撃が全然読めなくて本当に難しいんだよね…")
    ov.sig_tgt.emit(2, "这个 BOSS 的第二形态，攻击完全看不出来，真的很难啊……")
    ov.sig_src.emit(3, "それに加えて第三形態では雑魚を召喚してくるから、"
                       "みんなは僕のHPバーをちゃんと見ていてね、絶対に巻き込まれないで！")
    ov.sig_tgt.emit(3, "而且第三形态还会召唤小怪，大家记得盯紧我的血条，千万别被一起卷进去！")
    ov._ensure_layout()
    QApplication.processEvents()
    shot = ov.grab()
    ov.hide()

    # 贴一张模拟直播画面做背景，看字幕在复杂底图上的可读性
    canvas = QPixmap(1120, 460)
    canvas.fill(QColor("#0B0D12"))
    p = QPainter(canvas)
    g = QLinearGradient(0, 0, 1120, 460)
    g.setColorAt(0.0, QColor("#1B2440"))
    g.setColorAt(0.5, QColor("#2A1F3D"))
    g.setColorAt(1.0, QColor("#0E1622"))
    p.fillRect(canvas.rect(), g)
    p.setPen(QColor(255, 255, 255, 18))
    for i in range(0, 1120, 56):
        p.drawLine(i, 0, i, 460)
    for j in range(0, 460, 56):
        p.drawLine(0, j, 1120, j)
    p.drawPixmap(60, 60, shot)
    p.end()
    return canvas


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "docs", "images")
    os.makedirs(out_dir, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)

    base = AppConfig.load()
    # 预览只改内存里的副本，不落盘
    base.font_size = 26
    base.show_source = True
    base.opacity = 0.82
    base.max_lines = 3
    base.geometry = None

    idle = _panel(AppConfig.load(), running=False)
    _grab(idle, 760, 820).save(os.path.join(out_dir, "ui_panel_idle.png"), "PNG")

    running = _panel(AppConfig.load(), running=True)
    _grab(running, 760, 820).save(os.path.join(out_dir, "ui_panel_running.png"), "PNG")

    # 1080p 屏幕（1920×1080）下的窗口：内容超出可视区，靠滚动查看
    big = _panel(AppConfig.load(), running=True)
    _grab(big, 1200, 860).save(os.path.join(out_dir, "ui_panel_1080p.png"), "PNG")

    # 1366×768 小屏：窗口压到 1000×560，底栏固定可见
    small = _panel(AppConfig.load(), running=True)
    _grab(small, 1000, 560).save(os.path.join(out_dir, "ui_panel_1366x768.png"), "PNG")
    _grab(small, 1000, 560, scroll_bottom=True).save(
        os.path.join(out_dir, "ui_panel_1366x768_bottom.png"), "PNG")

    _overlay(base).save(os.path.join(out_dir, "ui_overlay_live.png"), "PNG")

    print("已输出：")
    for name in ("ui_panel_idle.png", "ui_panel_running.png", "ui_panel_1080p.png",
                 "ui_panel_1366x768.png", "ui_panel_1366x768_bottom.png",
                 "ui_overlay_live.png"):
        print(" ", os.path.join(out_dir, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
