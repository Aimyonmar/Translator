# -*- coding: utf-8 -*-
"""置顶悬浮字幕窗：半透明、可拖动、可缩放、带描边抗背景干扰。"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QPoint, QRect, QTimer, QSize
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPainterPath,
                           QPen, QBrush, QCursor, QAction)
from PySide6.QtWidgets import QWidget, QMenu, QApplication

FONT_FAMILY = "Microsoft YaHei UI"


def wrap_text(fm: QFontMetrics, text: str, max_w: int) -> List[str]:
    """按字符断行，CJK 友好；英文尽量在空格处断。"""
    lines: List[str] = []
    cur = ""
    last_space = -1
    for ch in text:
        if ch == " ":
            last_space = len(cur)
        test = cur + ch
        if fm.horizontalAdvance(test) > max_w and cur:
            cut = len(cur)
            if last_space > 0 and last_space < len(cur):
                cut = last_space + 1
            lines.append(cur[:cut].rstrip())
            cur = cur[cut:].lstrip() + ch
            last_space = -1
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


class SubtitleOverlay(QWidget):
    # 分两次上屏：识别一出来先顶原文（约 0.3s），译文到了再原地补上（约 1s）。
    # 用户看到字幕动起来的时间提前了将近一秒，最终形态跟原来一样。
    sig_src = Signal(int, str)      # 序号, 日语原文
    sig_tgt = Signal(int, str)      # 序号, 中译文（空串表示这句没翻出来）
    sig_clear = Signal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lines: List[Tuple[int, str, str, float]] = []
        self._drag_pos: Optional[QPoint] = None
        self._resizing = False
        self._pinned = True

        self.setWindowTitle("实时字幕")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setMinimumWidth(320)

        self.sig_src.connect(self._on_src)
        self.sig_tgt.connect(self._on_tgt)
        self.sig_clear.connect(self._on_clear)

        self._apply_fonts()
        self._restore_geometry()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

    # ---------------------------------------------------------- 配置
    def apply_config(self) -> None:
        self._apply_fonts()
        self.set_click_through(self.cfg.click_through)
        self.update()

    def _apply_fonts(self) -> None:
        self.font_trans = QFont(FONT_FAMILY, int(self.cfg.font_size), QFont.Bold)
        self.font_src = QFont(FONT_FAMILY, int(self.cfg.source_font_size))
        self.font_src.setItalic(False)

    def _restore_geometry(self) -> None:
        screens = QApplication.screens() or []
        primary = QApplication.primaryScreen()
        avail = primary.availableGeometry() if primary else QRect(0, 0, 1280, 720)
        w = int(getattr(self.cfg, "window_width", 0) or min(1100, avail.width() - 120))
        g = self.cfg.geometry or {}
        saved_ok = False
        if all(k in g for k in ("x", "y")):
            saved = QRect(int(g["x"]), int(g["y"]), int(g.get("w") or w), 160)
            # 位置必须落在某个屏幕上（换显示器/拔外接屏后旧坐标可能在屏幕外）
            saved_ok = any(s.availableGeometry().intersects(saved) for s in screens) if screens else True
        if saved_ok:
            self.resize(int(g.get("w") or w), 160)
            self.move(int(g["x"]), int(g["y"]))
        else:
            self.resize(w, 160)
            self.move(avail.center().x() - w // 2, avail.bottom() - 240)

    # ---------------------------------------------------------- 内容
    def _on_src(self, seq: int, src: str) -> None:
        """识别刚出结果：先把原文顶上屏，别让画面空着等翻译。"""
        if not src:
            return
        self.lines.append((seq, src, "", time.time()))
        self._trim()
        self._relayout()
        self.update()

    def _on_tgt(self, seq: int, tgt: str) -> None:
        """译文到了：填回对应那一行（可能已经被挤出屏幕，那就不管了）。"""
        for i, item in enumerate(self.lines):
            if item[0] == seq:
                self.lines[i] = (seq, item[1], tgt or "", item[3])
                self._relayout()
                self.update()
                return

    def _trim(self) -> None:
        keep = max(1, int(getattr(self.cfg, "max_lines", 3)))
        if len(self.lines) > keep:
            self.lines = self.lines[-keep:]

    def _on_clear(self) -> None:
        self.lines = []
        self._relayout()
        self.update()

    def _tick(self) -> None:
        sec = float(getattr(self.cfg, "auto_clear_sec", 0) or 0)
        if sec > 0 and self.lines:
            if time.time() - self.lines[-1][3] > sec:
                self.lines = []
                self._relayout()
                self.update()

    # ---------------------------------------------------------- 绘制
    def _layout(self, width: int):
        pad = int(self.cfg.font_size * 0.5) + 8
        inner = max(60, width - pad * 2)
        fm_t = QFontMetrics(self.font_trans)
        fm_s = QFontMetrics(self.font_src)
        items = []
        y = pad
        show_src = bool(self.cfg.show_source)
        for seq, src, tgt, _t in self.lines:
            if tgt:
                for line in wrap_text(fm_t, tgt, inner):
                    items.append(("t", line, y))
                    y += fm_t.height() + 6
                if show_src and src:
                    y += 2
                    for line in wrap_text(fm_s, src, inner):
                        items.append(("s", line, y))
                        y += fm_s.height() + 2
                    y += fm_s.height() * 0.35
            else:
                # 译文还没到：先用原文占住这一行，等译文到了原地替换
                for line in wrap_text(fm_t, src, inner):
                    items.append(("p", line, y))
                    y += fm_t.height() + 6
            y += 10
        return items, max(70, int(y + pad - 10)), pad, fm_t, fm_s

    def _relayout(self) -> None:
        items, h, pad, fm_t, fm_s = self._layout(self.width())
        self._items = items
        self._pad = pad
        if abs(self.height() - h) > 2:
            self.setFixedHeight(h)

    def resizeEvent(self, event):
        self.cfg.window_width = self.width()
        self._relayout()
        super().resizeEvent(event)

    def paintEvent(self, event):
        items, h, pad, fm_t, fm_s = self._layout(self.width())
        self._items = items
        self._pad = pad
        if abs(self.height() - h) > 2:
            self.setFixedHeight(h)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        alpha = int(max(0.0, min(1.0, float(self.cfg.opacity))) * 255)
        rect = self.rect().adjusted(2, 2, -2, -2)
        path = QPainterPath()
        path.addRoundedRect(rect.x(), rect.y(), rect.width(), rect.height(), 14, 14)
        p.fillPath(path, QBrush(QColor(8, 10, 14, alpha)))

        if not items:
            p.setPen(QColor(190, 198, 210, 190))
            p.setFont(self.font_src)
            p.drawText(rect, Qt.AlignCenter, "等待语音输入…")
            p.end()
            return

        for kind, text, y in items:
            if kind == "t":
                font, color = self.font_trans, QColor(255, 255, 255, 255)
                baseline = y + fm_t.ascent()
            elif kind == "p":
                # 译文还在路上：原文先用译文字号显示，颜色压暗一档以示「未翻译」
                font, color = self.font_trans, QColor(206, 214, 226, 235)
                baseline = y + fm_t.ascent()
            else:
                font, color = self.font_src, QColor(176, 196, 222, 235)
                baseline = y + fm_s.ascent()
            tp = QPainterPath()
            tp.addText(pad, baseline, font, text)
            if kind == "t":
                p.strokePath(tp, QPen(QColor(0, 0, 0, 235), max(3, self.cfg.font_size // 7),
                                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.fillPath(tp, QBrush(color))

        # 右下角缩放提示
        p.setPen(QPen(QColor(255, 255, 255, 70), 2))
        p.drawLine(self.width() - 16, self.height() - 6, self.width() - 6, self.height() - 16)
        p.end()

    # ---------------------------------------------------------- 交互
    def _in_resize_zone(self, pos: QPoint) -> bool:
        return pos.x() > self.width() - 28 and pos.y() > self.height() - 28

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._in_resize_zone(event.position().toPoint()):
                self._resizing = True
                self._drag_pos = event.globalPosition().toPoint()
            else:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        gp = event.globalPosition().toPoint()
        if self._resizing and self._drag_pos is not None:
            w = max(self.minimumWidth(), gp.x() - self.frameGeometry().left())
            self.resize(w, self.height())
            self._drag_pos = gp
            self.cfg.window_width = self.width()
            event.accept()
        elif self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(gp - self._drag_pos)
            event.accept()
        else:
            if self._in_resize_zone(event.position().toPoint()):
                self.setCursor(QCursor(Qt.SizeFDiagCursor))
            else:
                self.setCursor(QCursor(Qt.SizeAllCursor))

    def mouseReleaseEvent(self, event):
        if self._drag_pos is not None or self._resizing:
            self.cfg.geometry = {"x": self.x(), "y": self.y(), "w": self.width(), "h": self.height()}
            self.cfg.save()
        self._drag_pos = None
        self._resizing = False
        event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act_src = QAction("显示日语原文", self)
        act_src.setCheckable(True)
        act_src.setChecked(bool(self.cfg.show_source))
        act_src.triggered.connect(lambda: self._toggle_source())
        menu.addAction(act_src)

        act_pin = QAction("鼠标穿透（不挡操作）", self)
        act_pin.setCheckable(True)
        act_pin.setChecked(bool(self.cfg.click_through))
        act_pin.triggered.connect(lambda: self.set_click_through(not self.cfg.click_through))
        menu.addAction(act_pin)
        menu.addSeparator()

        menu.addAction("字号 +", lambda: self._bump_font(2))
        menu.addAction("字号 -", lambda: self._bump_font(-2))
        menu.addAction("背景更深", lambda: self._bump_bg(0.08))
        menu.addAction("背景更浅", lambda: self._bump_bg(-0.08))
        menu.addSeparator()
        menu.addAction("清空字幕", lambda: self.sig_clear.emit())
        menu.exec(event.globalPos())

    def _toggle_source(self) -> None:
        self.cfg.show_source = not self.cfg.show_source
        self.cfg.save()
        self._relayout()
        self.update()

    def _bump_font(self, d: int) -> None:
        self.cfg.font_size = max(12, min(72, int(self.cfg.font_size) + d))
        self.cfg.source_font_size = max(9, min(48, int(self.cfg.source_font_size) + d))
        self._apply_fonts()
        self.cfg.save()
        self._relayout()
        self.update()

    def _bump_bg(self, d: float) -> None:
        self.cfg.opacity = round(max(0.15, min(1.0, float(self.cfg.opacity) + d)), 2)
        self.cfg.save()
        self.update()

    def set_click_through(self, on: bool) -> None:
        """开启后鼠标直接穿透字幕（打游戏不挡操作），用托盘菜单切回来。"""
        self.cfg.click_through = bool(on)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, bool(on))
        if on:
            self.setCursor(QCursor(Qt.ArrowCursor))
        self.cfg.save()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.width(), self.height())
