# -*- coding: utf-8 -*-
"""置顶悬浮字幕窗：半透明、可拖动、可缩放、带描边抗背景干扰。"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QPoint, QRect, QTimer, QSize
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPainterPath,
                           QPen, QBrush, QCursor, QAction, QLinearGradient)
from PySide6.QtWidgets import QWidget, QMenu, QApplication

from ui import theme

FONT_FAMILY = "Microsoft YaHei UI"

# 中文禁则：行首不能是收尾标点，行尾不能是开括号（否则读起来像断错气）
_NO_LINE_START = "、。，,．.!?！？：；:;）)］]｝}」』】〉》”’…—ー~～%°"
_NO_LINE_END = "（(［[｛{「『【〈《“‘"


def wrap_text(fm: QFontMetrics, text: str, max_w: int) -> List[str]:
    """按字符断行，CJK 友好；英文尽量在空格处断，并做简单的中文禁则处理。"""
    lines: List[str] = []
    cur = ""
    for ch in text:
        over = cur and fm.horizontalAdvance(cur + ch) > max_w
        # 收尾标点宁可让上一行略微超宽，也不要它孤零零出现在行首
        if over and ch not in _NO_LINE_START:
            cut = len(cur)
            sp = cur.rfind(" ")
            if sp > 0:
                cut = sp + 1
            head = cur[:cut].rstrip()
            tail = cur[cut:]
            # 开括号不能留在行尾：挪到下一行去和后面的内容配对
            if head and head[-1] in _NO_LINE_END:
                tail = head[-1] + tail
                head = head[:-1].rstrip()
            if head:
                lines.append(head)
            cur = tail.lstrip() + ch
        else:
            cur += ch
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
        # 排版缓存：内容/尺寸/样式没变就不重新断行（以前每次 paintEvent 都重算一遍）
        self._items: List[Tuple[str, str, int]] = []
        self._pad = 0
        self._dirty = True
        self.font_trans = QFont(FONT_FAMILY)
        self.font_src = QFont(FONT_FAMILY)
        self._fm_t = QFontMetrics(self.font_trans)
        self._fm_s = QFontMetrics(self.font_src)

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
        self._dirty = True
        self.update()

    def _apply_fonts(self) -> None:
        main = theme.main_font_size(self.cfg.font_size)
        self.font_trans = QFont(FONT_FAMILY, main, QFont.Bold)
        # 副行字号只跟主字号走，不再单独读 cfg.source_font_size（避免双入口互相覆盖）
        self.font_src = QFont(FONT_FAMILY, theme.source_font_size(main))
        self.font_src.setItalic(False)
        self._fm_t = QFontMetrics(self.font_trans)
        self._fm_s = QFontMetrics(self.font_src)
        self._dirty = True
        # 同步回配置，保证落盘的数值与画面一致
        self.cfg.font_size = main
        self.cfg.source_font_size = theme.source_font_size(main)

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
        self._dirty = True
        self.update()

    def _on_tgt(self, seq: int, tgt: str) -> None:
        """译文到了：填回对应那一行（可能已经被挤出屏幕，那就不管了）。"""
        for i, item in enumerate(self.lines):
            if item[0] == seq:
                self.lines[i] = (seq, item[1], tgt or "", item[3])
                self._dirty = True
                self.update()
                return

    def _trim(self) -> None:
        keep = max(1, int(getattr(self.cfg, "max_lines", 3)))
        if len(self.lines) > keep:
            self.lines = self.lines[-keep:]

    def _on_clear(self) -> None:
        self.lines = []
        self._dirty = True
        self.update()

    def _tick(self) -> None:
        sec = float(getattr(self.cfg, "auto_clear_sec", 0) or 0)
        if sec > 0 and self.lines:
            if time.time() - self.lines[-1][3] > sec:
                self.lines = []
                self._dirty = True
                self.update()

    # ---------------------------------------------------------- 绘制
    def _layout(self, width: int):
        fs = int(self.cfg.font_size)
        # 内边距跟字号走：字号调大时留白同步变宽，观感不会突然变挤
        pad = max(18, int(fs * 0.62) + 10)
        inner = max(60, width - pad * 2)
        fm_t = self._fm_t
        fm_s = self._fm_s
        # 行距 ≈ 1.2 倍行高：译文行间留一点呼吸，长段也不糊成一块
        line_gap = max(4, int(fs * 0.26))
        src_gap = max(2, int(fs * 0.10))
        block_gap = max(10, int(fs * 0.55))
        items = []
        y = pad
        show_src = bool(self.cfg.show_source)
        for seq, src, tgt, _t in self.lines:
            if tgt:
                for line in wrap_text(fm_t, tgt, inner):
                    items.append(("t", line, y))
                    y += fm_t.height() + line_gap
                if show_src and src:
                    y += src_gap
                    for line in wrap_text(fm_s, src, inner):
                        items.append(("s", line, y))
                        y += fm_s.height() + src_gap
                    y += fm_s.height() * 0.3
            else:
                # 译文还没到：先用原文占住这一行，等译文到了原地替换
                for line in wrap_text(fm_t, src, inner):
                    items.append(("p", line, y))
                    y += fm_t.height() + line_gap
            y += block_gap
        return items, max(72, int(y + pad - block_gap)), pad, fm_t, fm_s

    def _ensure_layout(self) -> None:
        """排版只在「内容/尺寸/样式」变过时重算，普通重绘直接复用上次结果。"""
        if not self._dirty:
            return
        items, h, pad, fm_t, fm_s = self._layout(self.width())
        self._items = items
        self._pad = pad
        self._fm_t = fm_t
        self._fm_s = fm_s
        self._dirty = False
        if abs(self.height() - h) > 2:
            self.setFixedHeight(h)

    def _relayout(self) -> None:
        self._dirty = True
        self._ensure_layout()

    def resizeEvent(self, event):
        self.cfg.window_width = self.width()
        self._dirty = True
        self._ensure_layout()
        super().resizeEvent(event)

    def paintEvent(self, event):
        self._ensure_layout()

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        alpha = int(max(0.0, min(1.0, float(self.cfg.opacity))) * 255)
        rect = self.rect().adjusted(2, 2, -2, -2)
        path = QPainterPath()
        path.addRoundedRect(rect.x(), rect.y(), rect.width(), rect.height(), 16, 16)
        # 上浅下深的极轻渐变 + 1px 冷色描边：贴在任何画面上都不会糊成一块黑板
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QColor(16, 20, 28, alpha))
        grad.setColorAt(1.0, QColor(7, 9, 13, alpha))
        p.fillPath(path, QBrush(grad))
        p.strokePath(path, QPen(QColor(255, 255, 255, 22), 1))

        fs = int(self.cfg.font_size)
        if not self._items:
            p.setPen(QColor(154, 167, 184, 210))
            p.setFont(self.font_src)
            p.drawText(rect, Qt.AlignCenter, "等待语音输入…")
            p.end()
            return

        for kind, text, y in self._items:
            if kind == "t":
                font, color, fm = self.font_trans, QColor(255, 255, 255, 255), self._fm_t
            elif kind == "p":
                # 译文还在路上：原文先用译文字号显示，颜色压暗一档以示「未翻译」
                font, color, fm = self.font_trans, QColor(212, 223, 237, 232), self._fm_t
            else:
                font, color, fm = self.font_src, QColor(150, 178, 205, 230), self._fm_s
            baseline = y + fm.ascent()
            tp = QPainterPath()
            tp.addText(self._pad, baseline, font, text)
            if kind in ("t", "p"):
                # 描边只给大字：小字加描边会糊成毛边
                p.strokePath(tp, QPen(QColor(0, 0, 0, 215), max(3, fs // 8),
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
        menu.setStyleSheet(theme.menu_stylesheet())
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
        # 只调主字号，原文副行由 _apply_fonts 统一派生（不再两边各加各的）
        self.cfg.font_size = theme.main_font_size(int(self.cfg.font_size) + d)
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
