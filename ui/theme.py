# -*- coding: utf-8 -*-
"""统一视觉主题：配色常量、字号层级与全局 QSS。

只做「长什么样」，不含任何业务逻辑：
- 面板与字幕窗共用同一套颜色/字号，改一处两头同步；
- 纯 QSS + 代码内生成的小图标（下拉箭头/勾选标记），不引入额外依赖；
- 图标在运行时画好后缓存到系统临时目录，避免把二进制资源塞进仓库。
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable

# ---------------------------------------------------------------- 配色
# 深色、低饱和：底色越暗、面板越浅，靠「层级差」而不是描边堆叠来分区
BG = "#0E1117"          # 窗口底色
SURFACE = "#151A22"     # 分组/卡片底
SURFACE_HI = "#1B222C"  # 按钮、下拉框底
SUNKEN = "#0F141B"      # 输入框、凹槽底
BORDER = "#242C38"      # 常规描边
BORDER_HI = "#334052"   # 悬停/强调描边
TEXT = "#E4E9F2"        # 主文字
TEXT_DIM = "#9AA7B8"    # 次级文字（分组标题）
TEXT_MUTED = "#6C7A8C"  # 说明/占位
ACCENT = "#45C2FF"      # 主色：青蓝
ACCENT_HOVER = "#6FCEFF"
DANGER = "#E05A5A"      # 警示：错误/停止
WARN = "#E8B44A"
OK = "#4FD08A"

FONT_UI = '"Microsoft YaHei UI"'

# 圆角与间距节奏：大块 10 / 控件 8 / 小件 4，内边距按 4 的倍数走
RADIUS = 10
RADIUS_SM = 8


def menu_stylesheet() -> str:
    """字幕窗右键菜单单独一套：字幕窗不套全局 QSS（要保持透明自绘）。"""
    return f"""
QMenu {{
    background: #161C25;
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: {RADIUS_SM}px;
    padding: 6px;
}}
QMenu::item {{ padding: 6px 20px 6px 16px; border-radius: 6px; }}
QMenu::item:selected {{ background: #1E4A5E; }}
QMenu::item:disabled {{ color: {TEXT_MUTED}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
"""


def source_font_size(main_size: int) -> int:
    """原文（副行）字号由主字号派生——字号只保留面板一个入口。

    以前面板用「主字号 - 10」、字幕窗右键又各自 ±2，两边互相覆盖，
    最后显示多大完全看操作顺序。现在统一成一个换算规则。
    """
    return max(10, min(48, int(round(int(main_size) * 0.62))))


def main_font_size(main_size: int) -> int:
    """夹取主字号合法范围，供字幕窗右键调字号使用。"""
    return max(12, min(72, int(main_size)))


# ---------------------------------------------------------------- 图标
def _asset(name: str, size: int, paint: Callable) -> str:
    """在临时目录画一张透明 PNG 并返回 QSS 可用的路径（QSS 只认文件路径）。"""
    path = os.path.join(tempfile.gettempdir(), f"vtuber_ui_{name}_{size}.png")
    if not os.path.exists(path):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPainter, QPixmap

        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            paint(p, size)
        finally:
            p.end()
        pm.save(path, "PNG")
    return path.replace("\\", "/")


def _chevron(up: bool = False, color: str = "#9AA7B8") -> Callable:
    def draw(p, size: int) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QPen

        pen = QPen(QColor(color), max(1.4, size / 7.0))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        if up:
            p.drawPolyline([QPointF(size * 0.22, size * 0.62),
                            QPointF(size * 0.5, size * 0.36),
                            QPointF(size * 0.78, size * 0.62)])
        else:
            p.drawPolyline([QPointF(size * 0.22, size * 0.38),
                            QPointF(size * 0.5, size * 0.64),
                            QPointF(size * 0.78, size * 0.38)])
    return draw


def _check(color: str = "#04141C") -> Callable:
    def draw(p, size: int) -> None:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QPen

        pen = QPen(QColor(color), max(1.8, size / 6.5))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline([QPointF(size * 0.22, size * 0.52),
                        QPointF(size * 0.43, size * 0.72),
                        QPointF(size * 0.79, size * 0.30)])
    return draw


# ---------------------------------------------------------------- QSS
def stylesheet() -> str:
    """全局样式表（在 QApplication 起来之后再调用，里面要生成图标）。"""
    chevron_down = _asset("chevron_down", 12, _chevron(False))
    chevron_up = _asset("chevron_up", 12, _chevron(True))
    check = _asset("check", 16, _check())

    return f"""
/* ---------- 基础 ---------- */
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {FONT_UI};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{
    background: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 6px 8px;
}}
QLabel {{ background: transparent; color: {TEXT}; }}

/* 标题层级：应用名 15 / 分组名 12 / 正文 13 / 说明 11 */
QLabel#appTitle {{
    font-size: 15px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#appSubtitle {{
    font-size: 11px;
    color: {TEXT_MUTED};
}}
QLabel#stateValue {{
    font-size: 14px;
    font-weight: 700;
    color: {ACCENT};
}}
QLabel#hint {{ font-size: 11px; color: {TEXT_MUTED}; }}
/* 写盘失败提示：行内警示条，不遮挡内容、不打断字幕 */
QLabel#saveError {{
    font-size: 12px;
    color: {DANGER};
    background: rgba(224, 90, 90, 0.12);
    border: 1px solid rgba(224, 90, 90, 0.45);
    border-radius: 6px;
    padding: 6px 10px;
}}

/* ---------- 分组卡片 ---------- */
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    margin-top: 14px;
    padding: 10px 12px 10px 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {TEXT_DIM};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: {RADIUS_SM}px;
    padding: 6px 14px;
    min-height: 20px;
}}
QPushButton:hover {{ background: #212A37; border-color: #41516A; }}
QPushButton:pressed {{ background: #182029; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; background: #141A22; border-color: #1E252F; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton#primary {{
    background: {ACCENT};
    color: #04141C;
    border: 1px solid {ACCENT};
    font-weight: 700;
    font-size: 14px;
    border-radius: 10px;
    padding: 8px 18px;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: #35A9E2; }}
QPushButton#primary[running="true"] {{
    background: #3B1D1F;
    border-color: #7A3438;
    color: #FF9B9B;
}}
QPushButton#primary[running="true"]:hover {{ background: #4A2326; }}

/* ---------- 输入控件 ---------- */
QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {SUNKEN};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}px;
    padding: 6px 10px;
    selection-background-color: #1F5C77;
    selection-color: #EAF7FF;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QPlainTextEdit:disabled {{ color: {TEXT_MUTED}; }}
QPlainTextEdit#console {{
    background: #0A0E14;
    color: #B9C6D6;
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}px;
    padding: 8px;
}}
QPlainTextEdit#console QScrollBar:vertical {{ background: transparent; }}

QComboBox {{
    background: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: {RADIUS_SM}px;
    padding: 5px 26px 5px 10px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: #41516A; }}
QComboBox:focus, QComboBox:on {{ border-color: {ACCENT}; }}
QComboBox:disabled {{ color: {TEXT_MUTED}; border-color: #1E252F; background: #141A22; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
    background: transparent;
}}
QComboBox::down-arrow {{ image: url({chevron_down}); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: #171E27;
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: {RADIUS_SM}px;
    padding: 4px;
    outline: none;
    selection-background-color: #1E4A5E;
    selection-color: #EAF7FF;
}}
QComboBox QAbstractItemView::item {{ min-height: 26px; padding: 2px 8px; }}

QSpinBox, QDoubleSpinBox {{
    background: {SUNKEN};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}px;
    padding: 5px 22px 5px 8px;
    min-height: 20px;
    selection-background-color: #1F5C77;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: {RADIUS_SM}px;
    background: #171F29;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: 1px solid {BORDER};
    border-bottom-right-radius: {RADIUS_SM}px;
    background: #171F29;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: #22303f; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({chevron_up}); width: 8px; height: 8px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({chevron_down}); width: 8px; height: 8px; }}

/* ---------- 勾选框 ---------- */
QCheckBox {{ background: transparent; color: {TEXT}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_HI};
    border-radius: 4px;
    background: {SUNKEN};
}}
QCheckBox::indicator:hover {{ border-color: #4A5C74; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url({check});
}}
QCheckBox:disabled {{ color: {TEXT_MUTED}; }}

/* ---------- 滑块 ---------- */
QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: #222A35;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
    background: #EDF3FB;
    border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {SUNKEN};
    border: 1px solid {BORDER};
    border-radius: 5px;
    height: 10px;
}}
QProgressBar::chunk {{
    border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #2C87B8, stop:1 {ACCENT});
}}

/* ---------- 分页骨架：顶部传输栏 / 左侧导航 / 滚动内容 ---------- */
/* 传输栏：比底色浅一档，和内容区之间用 1px 线分隔，滚动时层次不混 */
QWidget#transport {{
    background: #121722;
    border-bottom: 1px solid {BORDER};
}}
QWidget#nav {{
    background: #0A0D13;
    border-right: 1px solid {BORDER};
}}
QPushButton#nav {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM}px;
    padding: 6px 12px;
    text-align: left;
    color: {TEXT_DIM};
    font-weight: 600;
}}
QPushButton#nav:hover {{ background: #161C25; color: {TEXT}; }}
QPushButton#nav:checked {{
    background: #16283A;
    border-color: #23445C;
    color: {ACCENT};
}}

/* ---------- 滚动容器 ---------- */
QScrollArea {{
    background: {BG};
    border: none;
}}
QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QWidget#page {{ background: {BG}; }}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #2A3441;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #3B4A5C; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #2A3441;
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
""" + menu_stylesheet()


def apply(widget) -> None:
    """给面板套上统一主题，并统一表单的间距节奏（4 的倍数）。"""
    from PySide6.QtWidgets import QFormLayout, QGroupBox

    widget.setStyleSheet(stylesheet())
    for box in widget.findChildren(QGroupBox):
        lay = box.layout()
        if isinstance(lay, QFormLayout):
            lay.setContentsMargins(2, 2, 2, 2)
            lay.setHorizontalSpacing(12)
            lay.setVerticalSpacing(8)
            lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
