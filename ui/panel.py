# -*- coding: utf-8 -*-
"""控制面板：设备 / 模型 / 翻译 / 字幕 参数调节与运行状态。"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QStandardPaths
from PySide6.QtGui import (QIcon, QPixmap, QPainter, QColor, QFont, QShortcut,
                          QKeySequence, QTextCursor)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QMessageBox, QLineEdit, QScrollArea, QFrame,
    QApplication, QAbstractSpinBox, QButtonGroup, QStackedWidget, QFileDialog,
    QDialog, QKeySequenceEdit,
)

from core import feedback, model_store
from core.asr import MODEL_SIZES
from ui.global_hotkey import DEFAULT_HOTKEY, GlobalHotkey, parse_hotkey
from core.translator import CHANNELS, CHANNEL_NAMES, LANGS
from ui import theme


def _icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setBrush(QColor(20, 22, 30))
    p.setPen(Qt.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    p.setPen(QColor(120, 220, 255))
    p.setFont(QFont("Microsoft YaHei UI", 22, QFont.Bold))
    p.drawText(pm.rect(), Qt.AlignCenter, "訳")
    p.end()
    return QIcon(pm)


class ControlPanel(QWidget):
    sig_status = Signal(str)
    sig_level = Signal(float)
    sig_log = Signal(str)
    sig_channel_result = Signal(str)
    # 工作线程（设备枚举/代理探测）不能直接碰控件，统一用信号投递回主线程
    sig_devices = Signal(list, object)
    sig_proxy_text = Signal(str)
    # 运行状态也走信号：流水线在后台线程里把 state 改成 error/stopped 时，
    # 按钮文字必须跟着变回「开始翻译」，不能只靠 toggle_start 那一次同步
    sig_running = Signal(bool, bool)
    # 模型缓存：下载/卸载在后台线程做，进度与结果一律回主线程更新控件
    sig_mdl_log = Signal(str)
    sig_mdl_done = Signal()
    sig_mdl_fail = Signal(str)
    # 音频转文字：转写在后台线程读整段文件，进度/结果同样用信号回主线程
    sig_trs_progress = Signal(int)
    sig_trs_status = Signal(str)
    sig_trs_done = Signal(object)       # (句段列表, 总时长秒, 是否被取消)
    sig_trs_fail = Signal(str)
    # 实时记录：原文上屏与译文上屏各来一次，按上屏顺序在面板里累积成转写稿
    sig_live_partial = Signal(int, str)
    sig_live_line = Signal(int, str, str)
    # 写盘结果：save() 可能在后台线程触发，统一用信号投递回主线程再改控件
    sig_save_state = Signal(object)

    def __init__(self, cfg, controller):
        super().__init__()
        self.cfg = cfg
        self.ctrl = controller
        self._running = False
        self._mdl_busy: Optional[str] = None
        # 音频转文字：与实时字幕互斥，同一时间只让一份模型干活
        self._transcribing = False
        self._trs_path = ""
        self._trs_segments: List[tuple] = []
        # 音频真实总时长（解码得到的 info.duration），工程 JSON 的 duration_s 用它，
        # 不能用「最后一句结束时间」——尾部静音会让时长偏短
        self._trs_total = 0.0
        # 实时记录：(起始秒, 结束秒, 原文, 译文)，按上屏顺序追加，只在内存里累积。
        # 用流水线的 seq 对齐：原文先到（记起始时间），译文后到（补该行并收尾）。
        self._live_records: List[Tuple[float, float, str, str]] = []
        self._live_open: dict = {}       # seq -> 记录下标：已上屏原文、还没等到译文
        self._live_index: dict = {}      # seq -> 记录下标：停止收尾后译文晚到也能回填
        self._live_t0: Optional[float] = None   # 本轮 monotonic 起点；None = 未运行
        self._live_origin = 0.0          # 本轮时间轴起点（接着上一轮结尾）
        self._live_frozen = 0.0          # 停止瞬间的秒数，用来给最后一句收尾
        # 标错：live = 流水线最近句快照；file = 已完成转写的句段
        self._fb_mode = "live"
        self._fb_count = 0
        self._trs_session = ""
        # 点击选中的句子（优先于「倒数第 N 句」）；block 区间用于把鼠标位置映射回句号
        self._fb_selected: Optional[Tuple[str, int]] = None
        self._live_blocks: List[Tuple[int, int, int]] = []
        self._trs_blocks: List[Tuple[int, int, int]] = []
        self._fb_syncing = False
        self._hotkey_ok = False
        self.setWindowTitle("VTuber 实时翻译器")
        self.setWindowIcon(_icon())
        # 横版：左导航 132 + 内容区，低于这个尺寸表单会挤成两行
        self.setMinimumSize(900, 580)
        self.sig_status.connect(self._set_status)
        self.sig_level.connect(self._set_level)
        self.sig_log.connect(self._log)
        self.sig_channel_result.connect(self._show_channel_result)
        self._build()
        # 必须放在 _build() 之后：txt_proxy 等控件是在 _build() 里创建的
        self.sig_devices.connect(self.set_devices)
        self.sig_proxy_text.connect(self.txt_proxy.setText)
        self.sig_running.connect(self.set_running)
        self.sig_mdl_log.connect(self._log)
        self.sig_mdl_done.connect(self._on_mdl_done)
        self.sig_mdl_fail.connect(self._on_mdl_fail)
        self.sig_trs_progress.connect(self._on_trs_progress)
        self.sig_trs_status.connect(self._on_trs_status)
        self.sig_trs_done.connect(self._on_trs_done)
        self.sig_trs_fail.connect(self._on_trs_fail)
        self.sig_live_partial.connect(self._on_live_partial)
        self.sig_live_line.connect(self._on_live_line)
        # 写盘失败提示：只在真实用户操作时出现，非交互闸门拦截只进日志
        self.sig_save_state.connect(self._on_save_state, Qt.QueuedConnection)
        hook = getattr(self.cfg, "set_save_hook", None)
        if hook is not None:
            hook(self._on_save_event)
        # 应用内快捷键作为全局热键失败时的后备（两者不会同时生效，避免一次按键记两条）
        self._fb_shortcut = QShortcut(QKeySequence(self._hotkey_text()), self)
        self._fb_shortcut.setContext(Qt.ApplicationShortcut)
        self._fb_shortcut.activated.connect(self.mark_feedback)
        # Windows 全局热键：面板在后台、焦点在游戏/播放器时也能触发。
        # 原生事件过滤器在 GUI 线程收 WM_HOTKEY，再排队回主线程执行 mark_feedback。
        self._hotkey = GlobalHotkey(self)
        self._hotkey.sig_activated.connect(self.mark_feedback, Qt.QueuedConnection)
        self._hotkey.sig_status.connect(self._on_hotkey_status)
        self._apply_hotkey()
        self._apply_default_size()
        # 滚轮防误触：整窗统一过滤，滑过下拉框/数值框时先滚页面（见 eventFilter）
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ------------------------------------------------------------ UI
    # 页面定义收敛在这一处：新增一页 = 这里加一行 + 写一个 _build_page_xxx()，
    # 顶部传输栏 / 左侧分页 / 右侧内容区 的骨架不用改。
    PAGES: Tuple[Tuple[str, str, str], ...] = (
        ("device", "设备", "_build_page_device"),
        ("asr", "识别", "_build_page_asr"),
        ("trans", "翻译", "_build_page_trans"),
        ("context", "语境", "_build_page_context"),
        ("transcribe", "转写", "_build_page_transcribe"),
        ("advanced", "高级", "_build_page_advanced"),
    )

    @staticmethod
    def _page_shell():
        """一页的容器：统一内边距与间距，页面内容从上往下堆。"""
        page = QWidget()
        page.setObjectName("page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        return page, lay

    def _build(self) -> None:
        # 三段式：顶部传输栏（固定）→ 左侧分页 → 右侧内容区（当前页 + 日志条）
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_transport())

        body = QWidget()
        bh = QHBoxLayout(body)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(0)
        bh.addWidget(self._build_nav())

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        self.stack = QStackedWidget()
        self._pages = {}
        self._page_scrolls = {}
        for key, _title, builder in self.PAGES:
            page = getattr(self, builder)()
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sc.setWidget(page)
            self.stack.addWidget(sc)
            self._pages[key] = page
            self._page_scrolls[key] = sc
        rv.addWidget(self.stack, 1)
        rv.addWidget(self._build_console())
        bh.addWidget(right, 1)
        outer.addWidget(body, 1)

        self.show_page(self.PAGES[0][0])
        # 统一样式在最后一次性套上：控件都建好了，QSS 才能一次覆盖全部
        theme.apply(self)

    # ------------------------------------------------------------ 骨架
    def _build_transport(self) -> QWidget:
        """顶部传输栏：主操作 + 运行状态 + 实时指标，任何分页下都固定可见。"""
        bar = QWidget()
        bar.setObjectName("transport")
        v = QVBoxLayout(bar)
        v.setContentsMargins(18, 14, 18, 12)
        v.setSpacing(10)

        # 标题区：一眼看清「这是什么 + 链路怎么走」
        head = QHBoxLayout()
        head.setSpacing(10)
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        lbl_title = QLabel("VTuber 实时翻译器")
        lbl_title.setObjectName("appTitle")
        lbl_sub = QLabel("系统声音 → 语音识别 → 翻译 → 悬浮字幕")
        lbl_sub.setObjectName("appSubtitle")
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_sub)
        head.addLayout(title_col)
        head.addStretch(1)
        # 本场链路计数：只读展示，与 eval 的 finals/with_tgt 同口径。
        # 固定最小宽度，数字变长时不推挤右侧的音量条（避免可见抖动）。
        self.lbl_counts = QLabel("")
        self.lbl_counts.setObjectName("hint")
        self.lbl_counts.setMinimumWidth(310)
        self.lbl_counts.setToolTip(
            "本场实时统计（只读，不参与翻译决策）：\n"
            "上屏 = 已显示的句子数；有译文 = 其中译文非空；\n"
            "失败 = 译文为空（原文兜底）；已标错 = 写入 feedback.jsonl 的条数；\n"
            "与 eval 报告的 finals / with_tgt 同口径，关闭标错后前四项照常统计。")
        self._refresh_counts()
        self.lbl_state = QLabel("未启动")
        self.lbl_state.setObjectName("stateValue")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedWidth(150)
        self.bar.setTextVisible(False)
        lbl_vol = QLabel("音量")
        lbl_vol.setObjectName("hint")
        head.addWidget(self.lbl_counts)
        head.addWidget(self.lbl_state)
        head.addWidget(lbl_vol)
        head.addWidget(self.bar)
        v.addLayout(head)

        # 主操作固定在顶栏：切到哪一页、内容滚到哪，开始/暂停/隐藏字幕都点得到
        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_start = QPushButton("开始翻译")
        # 主操作只在视觉上突出：运行中换成警示色，停下来的动作一眼可见
        self.btn_start.setObjectName("primary")
        self.btn_start.setFixedHeight(38)
        self.btn_start.clicked.connect(self.ctrl.toggle_start)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setFixedHeight(38)
        self.btn_pause.clicked.connect(self.ctrl.toggle_pause)
        self.btn_sub = QPushButton("隐藏字幕")
        self.btn_sub.setFixedHeight(38)
        self.btn_sub.clicked.connect(self.ctrl.toggle_subtitle)
        row.addWidget(self.btn_start, 2)
        row.addWidget(self.btn_pause, 1)
        row.addWidget(self.btn_sub, 1)
        v.addLayout(row)

        # 写盘失败提示条：不弹窗、不打断字幕，只在顶栏占一行；保存成功后自动收起。
        # 文案同时说明「文件没改」与「界面显示值未保存」，避免用户误以为改动已生效。
        self.save_bar = QWidget()
        sb = QHBoxLayout(self.save_bar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(8)
        self.lbl_save = QLabel("")
        self.lbl_save.setObjectName("saveError")
        self.lbl_save.setWordWrap(True)
        self.btn_save_retry = QPushButton("重试保存")
        self.btn_save_retry.setFixedHeight(26)
        self.btn_save_retry.clicked.connect(lambda: self._save_cfg("重试保存"))
        sb.addWidget(self.lbl_save, 1)
        sb.addWidget(self.btn_save_retry)
        self.save_bar.setVisible(False)
        self._save_failed = False
        v.addWidget(self.save_bar)
        return bar

    def _build_nav(self) -> QWidget:
        """左侧分页导航：按钮由 PAGES 生成，加页不用改这里。"""
        nav = QWidget()
        nav.setObjectName("nav")
        nav.setFixedWidth(132)
        v = QVBoxLayout(nav)
        v.setContentsMargins(10, 14, 10, 12)
        v.setSpacing(4)
        self._nav_buttons = {}
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for key, title, _builder in self.PAGES:
            btn = QPushButton(title)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.setFixedHeight(34)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self._nav_group.addButton(btn)
            v.addWidget(btn)
            self._nav_buttons[key] = btn
        v.addStretch(1)
        hint = QLabel("配置改动即时生效\n并自动保存")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return nav

    def _build_console(self) -> QWidget:
        """底部日志条：不随分页切换，识别/翻译发生了什么始终看得见。"""
        wrap = QWidget()
        wrap.setObjectName("consoleWrap")
        v = QVBoxLayout(wrap)
        v.setContentsMargins(18, 10, 18, 14)
        v.setSpacing(0)
        self.log = QPlainTextEdit()
        self.log.setObjectName("console")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setFixedHeight(126)
        # 显式给文档设字体：只靠 QSS 定字号时，文档行高与绘制字体对不上，多行会挤在一起
        self.log.setFont(QFont("Microsoft YaHei UI", 10))
        self.log.document().setDefaultFont(self.log.font())
        v.addWidget(self.log)
        return wrap

    def show_page(self, key: str) -> None:
        """切到指定分页；未知 key 忽略。预览脚本也走这个入口。"""
        sc = self._page_scrolls.get(key)
        if sc is None:
            return
        self.stack.setCurrentWidget(sc)
        # 滚轮过滤器与预览工具都按「当前页的滚动区」工作
        self.scroll = sc
        btn = self._nav_buttons.get(key)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    # ------------------------------------------------------------ 分页
    def _build_page_device(self) -> QWidget:
        page, root = self._page_shell()

        # 音频
        box_audio = QGroupBox("音频输入（默认抓取电脑系统声音）")
        f = QFormLayout(box_audio)
        self.cmb_device = QComboBox()
        self.cmb_device.setMinimumWidth(300)
        self.btn_refresh = QPushButton("刷新设备")
        self.btn_refresh.clicked.connect(self.ctrl.refresh_devices)
        row = QHBoxLayout()
        row.addWidget(self.cmb_device, 1)
        row.addWidget(self.btn_refresh)
        f.addRow("采集设备", row)
        self.cmb_device.currentIndexChanged.connect(self._on_device_changed)
        root.addWidget(box_audio)

        # 识别模型：用哪份权重、跑在哪个运算设备上
        box_model = QGroupBox("识别模型（Whisper）")
        f = QFormLayout(box_model)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODEL_SIZES)
        # 填初值时先屏蔽信号：否则一构建就触发保存，把 gpu_ok / last_channel 冲掉
        self.cmb_model.blockSignals(True)
        self.cmb_model.setCurrentText(self.cfg.model_size)
        self.cmb_model.blockSignals(False)
        self.cmb_model.setToolTip(
            "small：最快（本机约 0.21s/句），直播口语识别一般。\n"
            "medium：口语/专有名词识别更准，但比 small 慢约一倍；同理，\n"
            "口语多的直播可以把 small 换成它。\n"
            "large-v3-turbo：精度优于 small/medium、略低于 large-v3；短句比 small\n"
            "慢约四成，但 13 秒以上的长段反而更快；显存约 1.2GB。\n"
            "kotoba-whisper-v2.0（日语专用）：短句接近 small，13 秒以上长段最快；\n"
            "但游戏标题/人名准确率不如 small（本机 CER 7.3% vs 3.3%），且不支持热词，\n"
            "只建议纯日语、对速度要求高的直播使用。\n"
            "large-v3：识别最准（尤其口音/外来语/嘈杂），但单句耗时约为 small 的 3 倍，\n"
            "最吃显存（约 2GB）；6~8GB 显卡会自动改用 int8 跑。\n"
            "显存被游戏占用放不下时会自动临时降级到 medium/small 保速度\n"
            "（状态栏会提示），关掉游戏重启即可回到大模型。")
        self.cmb_model.currentTextChanged.connect(self._on_model_changed)
        self.cmb_dev = QComboBox()
        self.cmb_dev.addItems(["auto", "cpu", "cuda"])
        self.cmb_dev.blockSignals(True)
        self.cmb_dev.setCurrentText(self.cfg.asr_device)
        self.cmb_dev.blockSignals(False)
        self.cmb_dev.currentTextChanged.connect(self._on_model_changed)
        row2 = QHBoxLayout()
        row2.addWidget(self.cmb_model, 1)
        row2.addWidget(QLabel("运算"))
        row2.addWidget(self.cmb_dev)
        f.addRow("模型 / 设备", row2)
        # 本机实测（RTX 4050 6GB）：相同参数下 large-v3 约 0.63s/句、small 约 0.21s/句。
        # 只写一句克制的说明，避免用户为了「更准」误选成 3 倍延迟而不自知。
        self.lbl_model_hint = QLabel("large-v3 单句耗时约为 small 的 3 倍；追求实时性保持 small。")
        self.lbl_model_hint.setObjectName("hint")
        self.lbl_model_hint.setWordWrap(True)
        f.addRow("", self.lbl_model_hint)
        root.addWidget(box_model)

        # 模型缓存：状态直接读磁盘上的真实文件；已下载的只给「卸载」，
        # 没有下载入口，从 UI 上就堵死重复下载。
        box_mdl = QGroupBox("模型缓存（已下载的可离线使用，卸载即释放磁盘）")
        grid = QGridLayout(box_mdl)
        grid.setContentsMargins(2, 2, 2, 6)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self._mdl_rows = {}
        for r, size in enumerate(MODEL_SIZES):
            lbl_name = QLabel(size)
            lbl_name.setMinimumWidth(72)
            lbl_state = QLabel("…")
            lbl_state.setObjectName("hint")
            btn = QPushButton("下载")
            btn.setFixedWidth(88)
            btn.clicked.connect(lambda _=False, s=size: self._on_mdl_click(s))
            grid.addWidget(lbl_name, r, 0)
            grid.addWidget(lbl_state, r, 1)
            grid.addWidget(btn, r, 2)
            self._mdl_rows[size] = (lbl_state, btn)
        self.lbl_mdl_total = QLabel("")
        self.lbl_mdl_total.setObjectName("hint")
        grid.addWidget(self.lbl_mdl_total, len(MODEL_SIZES), 0, 1, 3)
        root.addWidget(box_mdl)
        self._refresh_models()

        root.addStretch(1)
        return page

    def _build_page_asr(self) -> QWidget:
        page, root = self._page_shell()

        # 识别增强：专有名词 / 热词 / 降噪 / 人声 / 解码宽度
        box_asr = QGroupBox("识别增强（专有名词 / 噪声 / 解码）")
        f = QFormLayout(box_asr)
        self.txt_wordmap = QLineEdit(self.cfg.word_map or "")
        self.txt_wordmap.setPlaceholderText("听错的人名/梗：误=正，逗号分隔。例：うさだぺこら=兎田ぺこら,ほろらいぶ=ホロライブ")
        self.txt_wordmap.editingFinished.connect(self._on_wordmap)
        f.addRow("纠错替换", self.txt_wordmap)
        # 专有名词音近修正：只录正确人名，程序按读音自动覆盖各种听错变体
        self.txt_names = QLineEdit(self.cfg.name_roster or "")
        self.txt_names.setPlaceholderText("正确人名，逗号分隔。含汉字给读音。例：兎田ぺこら=うさだぺこら,星街すいせい")
        self.txt_names.setToolTip(
            "与「纠错替换」的分工：\n"
            "· 纠错替换 = 精确替换，要写死「这次听错的字样」（うさだぺこら=兎田ぺこら）。\n"
            "· 本表 = 只录正确名字，程序按读音自动改回（假名/汉字/长音/促音/近音行都能覆盖），\n"
            "  维护量只随人名数增长，不用每出现一种新听法就加一条。")
        self.txt_names.editingFinished.connect(self._on_names)
        f.addRow("专有名词", self.txt_names)
        self.chk_hotwords = QCheckBox("同时作为识别热词 + 内置游戏/番剧标题")
        self.chk_hotwords.setChecked(bool(self.cfg.enable_hotwords))
        self.chk_hotwords.setToolTip(
            "把「专有名词」表和内置的常用游戏/番剧标题（40 个）一起下发给解码器，\n"
            "从识别源头提高专有名词命中率——识别错了，翻译层再准也救不回来。\n"
            "\n"
            "实测（6 个游戏标题 × 同一段音频，large-v3）：裸识别错误率 7.1% → 带热词 4.7%，\n"
            "典型修正：「黒髪話、悟空」→「黒神話、悟空」。\n"
            "误伤扫描：6 句普通直播句 × 热词表 → 凭空插入 0 处，错误率反而 3.1% → 1.7%。\n"
            "\n"
            "注意两条：\n"
            "· 热词用**汉字/片假名原样**最有效；早先用假名读音下发反而把解码带偏（26.1%），已改。\n"
            "· 万一解码器把词表当内容吐出来，程序有「复述护栏」直接丢弃，不会上屏词表。\n"
            "\n"
            "另外：识别对了还要翻对——同一个开关还会在送翻译前把游戏/番剧标题换成中文名\n"
            "（エルデンリング→艾尔登法环、「モンハンワイルズ」→怪物猎人：荒野），\n"
            "否则机翻会给出「埃尔金戒指」「蒙汉威尔斯」这种结果。")
        self.chk_hotwords.stateChanged.connect(self._on_hotwords)
        f.addRow("", self.chk_hotwords)
        # 降噪预处理
        self.chk_denoise = QCheckBox("降噪预处理")
        self.chk_denoise.setChecked(bool(self.cfg.enable_denoise))
        self.chk_denoise.setToolTip(
            "轻量谱减法，压低稳态背景噪声（风扇/电流声/低频隆隆）。\n"
            "对 BGM 这类非稳态音乐帮助有限（那需要人声分离模型），但能改善整体信噪比。")
        self.sp_denoise = QDoubleSpinBox()
        self.sp_denoise.setRange(0.0, 1.0)
        self.sp_denoise.setSingleStep(0.1)
        self.sp_denoise.setValue(float(self.cfg.denoise_strength))
        self.sp_denoise.setToolTip("0=几乎不消，1=最强；默认 0.5 较温和")
        self.sp_denoise.valueChanged.connect(self._on_denoise)
        self.chk_denoise.stateChanged.connect(self._on_denoise)
        row_dn = QHBoxLayout()
        row_dn.addWidget(self.chk_denoise)
        row_dn.addWidget(QLabel("强度"))
        row_dn.addWidget(self.sp_denoise)
        row_dn.addStretch(1)
        f.addRow("", row_dn)
        # 中置人声提取（压 BGM）：靠左右声道相关性取正中内容
        self.chk_vocal = QCheckBox("中置人声提取（压 BGM）")
        self.chk_vocal.setChecked(bool(self.cfg.vocal_extract))
        self.chk_vocal.setToolTip(
            "把立体声的正中成分（人声）留下、两侧成分（多数 BGM/伴奏）压下去。\n"
            "实测：BGM 很响导致一句都听不出时，识别准确率 100% 错 → 4.6% 错，等于把漏掉的句子救回来。\n"
            "代价有两处：① 人声本身偏左/偏右时会被一起削（罕见）；\n"
            "② BGM 也是居中混的（或整轨单声道）压不掉——那只能靠人声分离模型。\n"
            "纯语音直播（无 BGM）可以关掉，零改动更保险。")
        self.sp_vocal = QDoubleSpinBox()
        self.sp_vocal.setRange(0.0, 1.0)
        self.sp_vocal.setSingleStep(0.1)
        self.sp_vocal.setValue(float(self.cfg.vocal_strength))
        self.sp_vocal.setToolTip(
            "0=等同关闭（原样降混），1=只留正中成分。默认 0.8：压得动 BGM，人声几乎无损。")
        self.sp_vocal.valueChanged.connect(self._on_vocal)
        self.chk_vocal.stateChanged.connect(self._on_vocal)
        row_vc = QHBoxLayout()
        row_vc.addWidget(self.chk_vocal)
        row_vc.addWidget(QLabel("强度"))
        row_vc.addWidget(self.sp_vocal)
        row_vc.addStretch(1)
        f.addRow("", row_vc)
        # 解码宽度（速度 / 准确度权衡）：large-v3 想再快就拖到 1
        self.sp_beam = QSpinBox()
        self.sp_beam.setRange(1, 5)
        self.sp_beam.setValue(int(self.cfg.beam_size))
        self.sp_beam.setToolTip(
            "解码宽度：1=最快（贪心），2=快且稳（默认），3~5=更准但明显更慢。\n"
            "用 large-v3 还想再快就拖到 1；想更准往上拖。改完即时生效，不用重启。")
        self.sp_beam.valueChanged.connect(self._on_beam)
        self.sp_beam.setFixedWidth(88)   # 横版内容区很宽，不让数值框跟着铺满整行
        row_beam = QHBoxLayout()
        row_beam.addWidget(self.sp_beam)
        row_beam.addStretch(1)
        f.addRow("解码宽度", row_beam)
        root.addWidget(box_asr)
        root.addStretch(1)
        return page

    def _build_page_trans(self) -> QWidget:
        page, root = self._page_shell()

        # 翻译
        box_tr = QGroupBox("翻译（免费通道无需 Key；填 Key 或代理可显著提升质量）")
        f = QFormLayout(box_tr)
        self.cmb_channel = QComboBox()
        self.cmb_channel.addItems([CHANNEL_NAMES[c] for c in CHANNELS])
        names = list(LANGS.keys())
        self.cmb_channel.setCurrentIndex(CHANNELS.index(self.cfg.channel) if self.cfg.channel in CHANNELS else 0)
        self.cmb_channel.currentIndexChanged.connect(self._on_trans_changed)
        self.btn_test = QPushButton("测试")
        self.btn_test.clicked.connect(self.ctrl.test_translate)
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(names)
        cur = [k for k, v in LANGS.items() if v == self.cfg.target_lang]
        self.cmb_lang.setCurrentText(cur[0] if cur else names[0])
        self.cmb_lang.currentTextChanged.connect(self._on_trans_changed)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("通道"))
        row3.addWidget(self.cmb_channel, 1)
        row3.addWidget(QLabel("目标"))
        row3.addWidget(self.cmb_lang)
        row3.addWidget(self.btn_test)
        f.addRow("", row3)

        self.cmb_api = QComboBox()
        self.cmb_api.addItems(["不填（只用免费通道）", "OpenAI 兼容接口", "DeepL 官方 API"])
        idx = 0 if not self.cfg.custom_key else (1 if self.cfg.custom_type == "openai" else 2)
        self.cmb_api.setCurrentIndex(idx)
        self.cmb_api.currentIndexChanged.connect(self._on_api_changed)
        self.txt_key = QLineEdit(self.cfg.custom_key or "")
        self.txt_key.setEchoMode(QLineEdit.Password)
        self.txt_key.setPlaceholderText("填了就优先用它；留空则用免费通道")
        self.txt_key.editingFinished.connect(self._on_api_changed)
        row4 = QHBoxLayout()
        row4.addWidget(self.cmb_api)
        row4.addWidget(self.txt_key, 1)
        f.addRow("API", row4)

        self.txt_base = QLineEdit(self.cfg.custom_base or "")
        self.txt_base.setPlaceholderText("https://api.deepseek.com/v1")
        self.txt_base.editingFinished.connect(self._on_api_changed)
        self.txt_model = QLineEdit(self.cfg.custom_model or "")
        self.txt_model.setPlaceholderText("deepseek-chat")
        self.txt_model.editingFinished.connect(self._on_api_changed)
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("地址"))
        row5.addWidget(self.txt_base, 2)
        row5.addWidget(QLabel("模型"))
        row5.addWidget(self.txt_model, 1)
        f.addRow("", row5)

        self.txt_proxy = QLineEdit(self.cfg.proxy or "")
        self.txt_proxy.setPlaceholderText("http://127.0.0.1:7890（留空则自动探测）")
        self.txt_proxy.editingFinished.connect(self._on_proxy)
        self.btn_proxy = QPushButton("探测本地代理")
        self.btn_proxy.clicked.connect(self.ctrl.detect_proxy)
        row6 = QHBoxLayout()
        row6.addWidget(QLabel("代理"))
        row6.addWidget(self.txt_proxy, 1)
        row6.addWidget(self.btn_proxy)
        f.addRow("", row6)

        # 通道健康检查：逐个试通所有通道并显示耗时/失败原因。
        # 「译文消失」时最有用 —— 能分清是代理不通、彩云 token 失效，还是接口本身挂了。
        self.btn_health = QPushButton("通道健康检查")
        self.btn_health.setToolTip(
            "逐个试所有翻译通道，显示各自耗时与失败原因。\n"
            "译文消失时先看这个：一眼分清是代理不通、彩云 token 失效，还是接口挂了。")
        self.btn_health.clicked.connect(self._on_health_click)
        self.lbl_health = QLabel("未检查")
        row_hl = QHBoxLayout()
        row_hl.addWidget(self.btn_health)
        row_hl.addWidget(self.lbl_health, 1)
        f.addRow("", row_hl)
        self.txt_health = QPlainTextEdit()
        self.txt_health.setReadOnly(True)
        self.txt_health.setPlainText("点「通道健康检查」逐个试通各通道")
        self.txt_health.setFixedHeight(96)
        f.addRow("", self.txt_health)
        root.addWidget(box_tr)
        root.addStretch(1)
        return page

    def _build_page_context(self) -> QWidget:
        page, root = self._page_shell()

        # 语境辅助（设计见 docs/语境翻译设计.md）：三个开关 + 前文条数 + 术语表。
        # 全部改动即存，运行时每句都重新读 cfg，不需要重启，也不需要通知控制器。
        box_ctx = QGroupBox("语境辅助（减少指代/同音词误译）")
        f = QFormLayout(box_ctx)
        self.chk_context = QCheckBox("翻译参考前文")
        self.chk_context.setChecked(bool(self.cfg.enable_context))
        self.chk_context.setToolTip(
            "把最近几条「原文→译文」交给自定义 LLM 通道，帮助消解省略主语、指代和一词多义。\n"
            "只对填了 API 的自定义通道生效：免费通道没有上下文参数，单句翻译。\n"
            "关闭后翻译层完全不注入上下文，行为与加这个功能之前一致。\n"
            "前文只来自本次运行已上屏的句子，重新「开始翻译」会清空。")
        self.sp_ctx = QSpinBox()
        self.sp_ctx.setRange(0, 8)
        self.sp_ctx.setValue(int(self.cfg.context_window))
        self.sp_ctx.setToolTip(
            "带几条前文（0~8，默认 3）。0 等同关闭。\n"
            "越多越能消歧，但提示词更长，LLM 通道会稍慢一点（每条约几十字，延迟增量很小）。")
        self.chk_homophone = QCheckBox("识别同音词纠正")
        self.chk_homophone.setChecked(bool(self.cfg.enable_homophone_fix))
        self.chk_homophone.setToolTip(
            "识别层按上下文线索改掉日语同音误听：変身→返信、対照→対象。\n"
            "只在有把握时改（整句只剩该词、无上下文线索都不动），每次改动会在日志里注明。\n"
            "发现改错就关掉，识别结果原样送出。")
        self.chk_free = QCheckBox("免费通道也带语境（试验）")
        self.chk_free.setChecked(bool(self.cfg.enable_free_context))
        self.chk_free.setToolTip(
            "默认关闭。开启后把上一句原文（最多 60 字）用换行拼在当前句前面一起送翻，\n"
            "再按换行把当前句那段切出来。\n"
            "代价与风险：每次请求多带几十字（延迟略增）；个别通道会把两句合成一段，\n"
            "程序会检测到并改用单句重发一次，仍拿不到干净译文就只显示原文。\n"
            "收益有限：字幕本身做不到跨句一致，只对这一句的主语/指代有帮助，可自行对比。")
        row_ctx = QHBoxLayout()
        row_ctx.addWidget(self.chk_context)
        row_ctx.addSpacing(12)
        row_ctx.addWidget(QLabel("前文条数"))
        row_ctx.addWidget(self.sp_ctx)
        row_ctx.addStretch(1)
        f.addRow("", row_ctx)
        f.addRow("", self.chk_homophone)
        f.addRow("", self.chk_free)
        self.txt_glossary = QPlainTextEdit()
        self.txt_glossary.setPlainText(self.cfg.context_glossary or "")
        self.txt_glossary.setPlaceholderText(
            "一行一条「原词=译法」\n例：返信=回复\nホロライブ=Hololive\n配信=直播")
        self.txt_glossary.setFixedHeight(64)
        self.txt_glossary.setToolTip(
            "术语表：当前句出现「原词」时，按这里的「译法」翻。\n"
            "只对自定义 LLM 通道生效（免费通道无法强制术语）；写错不会报错，只是不生效。\n"
            "输入停顿 0.6 秒自动保存。")
        f.addRow("术语表", self.txt_glossary)
        lbl_ctx_hint = QLabel("以上只影响翻译与识别，不改动音频链路；术语表仅自定义 LLM 通道生效。")
        lbl_ctx_hint.setObjectName("hint")
        lbl_ctx_hint.setWordWrap(True)
        f.addRow("", lbl_ctx_hint)
        # 术语表逐字写盘太浪费：停顿 600ms 后一次性保存（定时器在 GUI 线程，单次触发）
        self._glossary_timer = QTimer(self)
        self._glossary_timer.setSingleShot(True)
        self._glossary_timer.setInterval(600)
        self._glossary_timer.timeout.connect(self._on_glossary)
        self.chk_context.stateChanged.connect(self._on_context_toggle)
        self.chk_homophone.stateChanged.connect(self._on_context_toggle)
        self.chk_free.stateChanged.connect(self._on_context_toggle)
        self.sp_ctx.valueChanged.connect(self._on_context_toggle)
        self.txt_glossary.textChanged.connect(self._on_glossary_changed)
        # 前文条数/免费语境都依赖「翻译参考前文」：关掉时没有任何前文可拼
        self.sp_ctx.setEnabled(self.chk_context.isChecked())
        self.chk_free.setEnabled(self.chk_context.isChecked())
        root.addWidget(box_ctx)
        root.addStretch(1)
        return page

    def _build_page_transcribe(self) -> QWidget:
        page, root = self._page_shell()

        # 实时记录：只做转写页顶部一块只读文本 + 三个操作，不加页、不动滚动结构。
        # 数据来自流水线上屏顺序（见 main.py 的 on_partial / on_result），
        # 与「文件转写」互不干扰；文件转写期间实时信号会被忽略。
        box_live = QGroupBox("实时记录（按字幕上屏顺序累积）")
        v_live = QVBoxLayout(box_live)
        self.txt_live = QPlainTextEdit()
        self.txt_live.setReadOnly(True)
        self.txt_live.setMinimumHeight(110)
        self.txt_live.setFont(QFont("Microsoft YaHei UI", 10))
        self.txt_live.document().setDefaultFont(self.txt_live.font())
        self.txt_live.setPlaceholderText(
            "开始翻译后，这里按上屏顺序累积原文与译文；停止时会保留只有原文的句子。")
        # 点击任意一句即选中，标错优先标选中句（见 _select_live_at）
        self.txt_live.viewport().installEventFilter(self)
        v_live.addWidget(self.txt_live, 1)
        row_live = QHBoxLayout()
        self.lbl_live_out = QLabel("")
        self.lbl_live_out.setObjectName("hint")
        self.lbl_live_out.setWordWrap(True)
        row_live.addWidget(self.lbl_live_out, 1)
        self.btn_live_clear = QPushButton("清空")
        self.btn_live_txt = QPushButton("导出 TXT")
        self.btn_live_srt = QPushButton("导出 SRT")
        self.btn_live_ass = QPushButton("导出 ASS")
        self.btn_live_clear.clicked.connect(self._on_live_clear)
        self.btn_live_txt.clicked.connect(lambda: self._export_live("txt"))
        self.btn_live_srt.clicked.connect(lambda: self._export_live("srt"))
        self.btn_live_ass.clicked.connect(lambda: self._export_live("ass"))
        for b in (self.btn_live_clear, self.btn_live_txt, self.btn_live_srt,
                  self.btn_live_ass):
            b.setEnabled(False)
            row_live.addWidget(b)
        v_live.addLayout(row_live)

        # 一键标错：一行操作 + 一行错误类型，数据只写 .runtime/feedback.jsonl。
        # 找不到目标时按钮置灰，避免「按了没反应」。
        row_fb = QHBoxLayout()
        self.chk_fb = QCheckBox("启用标错")
        self.chk_fb.setChecked(bool(getattr(self.cfg, "enable_feedback", True)))
        self.chk_fb.setToolTip(
            "开启：实时流保留最近 %d 句、转写保留全部结果，可按快捷键标记。\n"
            "关闭：不保留快照、不写 feedback.jsonl，翻译与识别路径不受影响。" % feedback.RECENT_MAX)
        self.chk_fb.toggled.connect(self._on_feedback_toggle)
        self.btn_fb = QPushButton("标错")
        self.btn_fb.setToolTip(
            "标记选中的句子；没有选中时标记「倒数第 N 句」。\n"
            "写 .runtime/feedback.jsonl 走后台线程，不卡字幕。")
        self.btn_fb.clicked.connect(self.mark_feedback)
        self.spn_fb_back = QSpinBox()
        self.spn_fb_back.setRange(0, feedback.RECENT_MAX - 1)
        self.spn_fb_back.setSpecialValueText("最近一句")
        self.spn_fb_back.setPrefix("倒数第 ")
        self.spn_fb_back.setSuffix(" 句")
        self.spn_fb_back.setFixedWidth(120)
        self.spn_fb_back.setToolTip(
            "实时流可回溯最近 %d 句（约 60~80 秒）；转写完成后可回溯全部句段。" % feedback.RECENT_MAX)
        self.spn_fb_back.valueChanged.connect(self._on_fb_back_changed)
        lbl_hk = QLabel("热键")
        lbl_hk.setObjectName("hint")
        self.key_fb = QKeySequenceEdit(QKeySequence(self._hotkey_text()))
        self.key_fb.setMaximumSequenceLength(1)
        self.key_fb.setFixedWidth(120)
        self.key_fb.setToolTip("Windows 全局热键：面板在后台时也能标错；至少含 Ctrl/Alt/Win")
        self.key_fb.editingFinished.connect(self._on_hotkey_edit)
        self.lbl_fb = QLabel("全局热键未注册")
        self.lbl_fb.setObjectName("hint")
        self.lbl_fb.setMaximumWidth(240)
        row_fb.addWidget(self.chk_fb)
        row_fb.addWidget(self.btn_fb)
        row_fb.addWidget(self.spn_fb_back)
        row_fb.addWidget(lbl_hk)
        row_fb.addWidget(self.key_fb)
        row_fb.addWidget(self.lbl_fb, 1)
        v_live.addLayout(row_fb)

        row_tags = QHBoxLayout()
        lbl_tags = QLabel("错误类型")
        lbl_tags.setObjectName("hint")
        row_tags.addWidget(lbl_tags)
        self._fb_tags = {}
        for key, label in feedback.TAGS:
            cb = QCheckBox(label)
            cb.toggled.connect(self._on_fb_tag_changed)
            self._fb_tags[key] = cb
            row_tags.addWidget(cb)
        row_tags.addStretch(1)
        v_live.addLayout(row_tags)
        self._refresh_feedback_ui()
        root.addWidget(box_live)
        self._live_update_out_label()

        box_src = QGroupBox("源文件（本地音频 / 视频，一次转完）")
        f = QFormLayout(box_src)
        self.lbl_trs_file = QLabel("未选择")
        self.lbl_trs_file.setObjectName("hint")
        self.lbl_trs_file.setWordWrap(True)
        self.btn_trs_pick = QPushButton("选择文件…")
        self.btn_trs_pick.clicked.connect(self._on_trs_pick)
        row = QHBoxLayout()
        row.addWidget(self.lbl_trs_file, 1)
        row.addWidget(self.btn_trs_pick)
        f.addRow("文件", row)
        self.lbl_trs_out = QLabel("导出目录：与源文件相同")
        self.lbl_trs_out.setObjectName("hint")
        f.addRow("", self.lbl_trs_out)
        root.addWidget(box_src)

        box_run = QGroupBox("转写")
        v = QVBoxLayout(box_run)
        row2 = QHBoxLayout()
        self.btn_trs_start = QPushButton("开始转写")
        self.btn_trs_start.setObjectName("primary")
        self.btn_trs_start.setFixedHeight(36)
        self.btn_trs_start.clicked.connect(self._on_trs_start)
        self.btn_trs_cancel = QPushButton("取消")
        self.btn_trs_cancel.setFixedHeight(36)
        self.btn_trs_cancel.setEnabled(False)
        self.btn_trs_cancel.clicked.connect(self._on_trs_cancel)
        row2.addWidget(self.btn_trs_start, 2)
        row2.addWidget(self.btn_trs_cancel, 1)
        v.addLayout(row2)
        # 两个可选项都只影响离线转写：识别完再处理，不进实时链路
        row_opt = QHBoxLayout()
        self.chk_trs_translate = QCheckBox("同时翻译（复用实时通道）")
        self.chk_trs_translate.setToolTip(
            "转写完接着逐句翻译，复用实时字幕用的通道、主备补发与译文缓存。\n"
            "开启后可导出双语 SRT 与单独译文 SRT。")
        self.chk_trs_translate.toggled.connect(self._on_trs_opts_changed)
        self.spn_trs_chars = QSpinBox()
        self.spn_trs_chars.setRange(0, 200)
        self.spn_trs_chars.setSpecialValueText("不重新断句")
        self.spn_trs_chars.setValue(0)
        self.spn_trs_chars.setFixedWidth(110)
        self.spn_trs_chars.setToolTip(
            "把过长的字幕按最大字数切开：优先在标点处下刀，时间码按字词时间码\n"
            "（有就用）或字数比例分配，再送去翻译。0 表示保留原句。")
        self.spn_trs_chars.valueChanged.connect(self._on_trs_opts_changed)
        self.chk_trs_translate.setChecked(bool(getattr(self.cfg, "trs_translate", False)))
        self.spn_trs_chars.setValue(int(getattr(self.cfg, "trs_max_chars", 0) or 0))
        row_opt.addWidget(self.chk_trs_translate, 1)
        row_opt.addWidget(QLabel("单句最多"))
        row_opt.addWidget(self.spn_trs_chars)
        row_opt.addWidget(QLabel("字"))
        row_opt.addStretch(1)
        v.addLayout(row_opt)
        self.bar_trs = QProgressBar()
        self.bar_trs.setRange(0, 100)
        self.bar_trs.setValue(0)
        v.addWidget(self.bar_trs)
        self.lbl_trs_state = QLabel("选好文件后点「开始转写」；与实时字幕互斥，不能同时跑。")
        self.lbl_trs_state.setObjectName("hint")
        self.lbl_trs_state.setWordWrap(True)
        v.addWidget(self.lbl_trs_state)
        # 导出按钮单独一行：格式变多以后和状态行挤在一起会互相压缩
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("导出"))
        self.btn_trs_txt = QPushButton("TXT")
        self.btn_trs_srt = QPushButton("SRT")
        self.btn_trs_ass = QPushButton("ASS")
        self.btn_trs_bi = QPushButton("双语 SRT")
        self.btn_trs_tsrt = QPushButton("译文 SRT")
        self.btn_trs_json = QPushButton("工程 JSON")
        self._trs_export_btns = (self.btn_trs_txt, self.btn_trs_srt, self.btn_trs_ass,
                                 self.btn_trs_bi, self.btn_trs_tsrt, self.btn_trs_json)
        for b in self._trs_export_btns:
            b.setEnabled(False)
        self.btn_trs_txt.clicked.connect(lambda: self._export_trs("txt"))
        self.btn_trs_srt.clicked.connect(lambda: self._export_trs("srt"))
        self.btn_trs_ass.clicked.connect(lambda: self._export_trs("ass"))
        self.btn_trs_bi.clicked.connect(lambda: self._export_trs("srt2"))
        self.btn_trs_tsrt.clicked.connect(lambda: self._export_trs("tsrt"))
        self.btn_trs_json.clicked.connect(lambda: self._export_trs("json"))
        for b in self._trs_export_btns:
            row3.addWidget(b)
        row3.addStretch(1)
        v.addLayout(row3)
        root.addWidget(box_run)

        box_res = QGroupBox("结果（带时间戳）")
        v2 = QVBoxLayout(box_res)
        self.txt_trs_result = QPlainTextEdit()
        self.txt_trs_result.setReadOnly(True)
        # 只定「最小」高度：默认 760 高时整页刚好放下「导出」按钮，窗口拉高再自动长高
        self.txt_trs_result.setMinimumHeight(90)
        self.txt_trs_result.setFont(QFont("Microsoft YaHei UI", 10))
        self.txt_trs_result.document().setDefaultFont(self.txt_trs_result.font())
        self.txt_trs_result.setPlaceholderText("转写完成后这里显示带时间戳的文本")
        self.txt_trs_result.viewport().installEventFilter(self)
        v2.addWidget(self.txt_trs_result, 1)
        root.addWidget(box_res, 1)
        return page

    def _build_page_advanced(self) -> QWidget:
        page, root = self._page_shell()

        # 断句
        box_vad = QGroupBox("断句灵敏度")
        f = QFormLayout(box_vad)
        self.sld_sens = QSlider(Qt.Horizontal)
        self.sld_sens.setRange(0, 100)
        self.sld_sens.setValue(int(float(self.cfg.sensitivity) * 100))
        self.lbl_sens = QLabel(f"{float(self.cfg.sensitivity):.2f}")
        row4 = QHBoxLayout()
        row4.addWidget(self.sld_sens, 1)
        row4.addWidget(self.lbl_sens)
        self.sld_sens.valueChanged.connect(self._on_sens)
        self.sld_sens.setToolTip(
            "不是越高越好。调到接近 1.0 时，背景音乐也会被当成语音，"
            "句子就切不开了，只能等「最长片段」超时才出字幕 —— 延迟会变成好几秒。\n"
            "建议 0.6 ~ 0.8；漏字时优先调高这个，但别拉满。")
        f.addRow("灵敏度（越大越容易触发）", row4)
        # 实时诊断：直接显示「当前音量 vs 触发线」，不出字幕时一眼能看出卡在哪
        self.lbl_diag = QLabel("点「开始翻译」后，这里显示实时音量与触发线")
        self.lbl_diag.setWordWrap(True)
        self.lbl_diag.setObjectName("hint")
        f.addRow("", self.lbl_diag)
        self.sp_silence = QSpinBox()
        self.sp_silence.setRange(150, 2000)
        self.sp_silence.setSingleStep(50)
        self.sp_silence.setValue(int(self.cfg.min_silence_ms))
        self.sp_silence.setSuffix(" ms")
        self.sp_silence.valueChanged.connect(self._on_vad)
        self.sp_max = QSpinBox()
        self.sp_max.setRange(3000, 30000)
        self.sp_max.setSingleStep(1000)
        self.sp_max.setValue(int(self.cfg.max_speech_ms))
        self.sp_max.setSuffix(" ms")
        self.sp_max.valueChanged.connect(self._on_vad)
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("停顿多久算一句"))
        row5.addWidget(self.sp_silence)
        row5.addWidget(QLabel("最长片段"))
        row5.addWidget(self.sp_max)
        f.addRow("", row5)
        root.addWidget(box_vad)

        # 字幕
        box_sub = QGroupBox("字幕样式")
        f = QFormLayout(box_sub)
        self.sp_font = QSpinBox()
        self.sp_font.setRange(12, 72)
        self.sp_font.setValue(int(self.cfg.font_size))
        self.sp_font.valueChanged.connect(self._on_style)
        self.chk_src = QCheckBox("显示日语原文")
        self.chk_src.setChecked(bool(self.cfg.show_source))
        self.chk_src.stateChanged.connect(self._on_style)
        row6 = QHBoxLayout()
        row6.addWidget(QLabel("译文字号"))
        row6.addWidget(self.sp_font)
        row6.addSpacing(12)
        row6.addWidget(self.chk_src)
        row6.addStretch(1)
        f.addRow("", row6)
        self.sld_bg = QSlider(Qt.Horizontal)
        self.sld_bg.setRange(10, 100)
        self.sld_bg.setValue(int(float(self.cfg.opacity) * 100))
        self.sld_bg.valueChanged.connect(self._on_bg)
        self.sp_lines = QSpinBox()
        self.sp_lines.setRange(1, 8)
        self.sp_lines.setValue(int(self.cfg.max_lines))
        self.sp_lines.valueChanged.connect(self._on_style)
        self.sp_clear = QDoubleSpinBox()
        self.sp_clear.setRange(0, 60)
        self.sp_clear.setSingleStep(1)
        self.sp_clear.setSuffix(" 秒后清空(0=否)")
        self.sp_clear.setValue(float(self.cfg.auto_clear_sec))
        self.sp_clear.valueChanged.connect(self._on_style)
        row7 = QHBoxLayout()
        row7.addWidget(QLabel("背景"))
        row7.addWidget(self.sld_bg, 1)
        row7.addWidget(QLabel("保留行数"))
        row7.addWidget(self.sp_lines)
        row7.addWidget(self.sp_clear)
        f.addRow("", row7)
        root.addWidget(box_sub)

        tip = QLabel("提示：字幕窗可用鼠标拖动，右下角拖动改宽度，右键可调字号/背景/穿透。")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        root.addWidget(tip)

        # 规则改写追溯：默认全开后，命中会进日志条，这里可回看最近 20 条对照
        box_rule = QGroupBox("规则化译后处理（默认开启）")
        f = QFormLayout(box_rule)
        lbl_rule = QLabel("译后规则命中时会以「规则[id/v版本] 改前 → 改后」写进底部日志；"
                          "最近 20 条同时保存在 .runtime/rule_changes.json，重启后仍可查。"
                          "配置里可用 id 只开或排除单条，如 zh_rules=-zh-tone。")
        lbl_rule.setObjectName("hint")
        lbl_rule.setWordWrap(True)
        self.btn_rule_trace = QPushButton("查看最近 20 条")
        self.btn_rule_trace.clicked.connect(self._show_rule_trace)
        row_rule = QHBoxLayout()
        row_rule.addWidget(lbl_rule, 1)
        row_rule.addWidget(self.btn_rule_trace)
        f.addRow("", row_rule)
        root.addWidget(box_rule)

        root.addStretch(1)
        return page

    def _show_rule_trace(self) -> None:
        """展示最近 20 条规则改写：内存优先，重启后回落到落盘记录。"""
        from core import zh_rules
        rows = zh_rules.recent(20)
        if not rows:
            try:
                with open(zh_rules.TRACE_PATH, "r", encoding="utf-8") as f:
                    rows = (json.load(f) or [])[-20:]
            except Exception:
                rows = []
        if not rows:
            QMessageBox.information(self, "规则改写记录", "还没有命中记录。")
            return
        lines = []
        for i, r in enumerate(rows, 1):
            rid = r.get("rule_id") or r.get("rule", "")
            ver = r.get("rule_ver")
            tag = "%s/v%s" % (rid, ver) if ver else rid
            lines.append(f"{i}. [{tag}] {r.get('rule', '')}：{r.get('before', '')} → {r.get('after', '')}")
            lines.append(f"   原文：{r.get('src', '')}")
        dlg = QDialog(self)
        dlg.setWindowTitle("最近 20 条规则改写")
        dlg.resize(640, 460)
        v = QVBoxLayout(dlg)
        view = QPlainTextEdit("\n".join(lines))
        view.setReadOnly(True)
        view.setFont(QFont("Microsoft YaHei UI", 10))
        v.addWidget(view, 1)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        v.addWidget(btn, 0, Qt.AlignRight)
        dlg.exec()

    def _apply_default_size(self) -> None:
        """默认横版尺寸：按分页布局取一个固定比例，再夹到屏幕可用区域内。

        分页后每页内容单独滚动，窗口尺寸不再由某一个页面的 sizeHint 决定；
        默认 1120×760，小屏按可用区域收窄，最小尺寸由 setMinimumSize 兜底。
        """
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        max_w = (avail.width() - 80) if avail else 1280
        max_h = (avail.height() - 80) if avail else 800
        w = max(900, min(1120, max_w))
        h = max(580, min(760, max_h))
        self.resize(w, h)

    # ------------------------------------------------------------ 回调
    def _on_device_changed(self, idx: int) -> None:
        dev = self.cmb_device.itemData(idx, Qt.UserRole)
        if dev is not None:
            self.ctrl.on_device_selected(dev)

    def _on_model_changed(self, *_a) -> None:
        self.cfg.model_size = self.cmb_model.currentText()
        self.cfg.asr_device = self.cmb_dev.currentText()
        # 用户手动改了设备，就重新给 GPU 一次机会（比如刚装好 CUDA 运行库）
        self.cfg.gpu_ok = True
        self._save_cfg()
        self.ctrl.on_model_changed()

    # ------------------------------------------------------------ 模型缓存
    def _refresh_models(self) -> None:
        """按磁盘实际文件刷新每行状态；不看 refs、不看内存变量。"""
        for size, (lbl, btn) in self._mdl_rows.items():
            if model_store.is_downloaded(size):
                used = model_store.size_bytes(size) / (1024 * 1024)
                lbl.setText(f"已下载 · 占用 {used:.0f} MB")
                lbl.setStyleSheet(f"color:{theme.OK}; font-size:11px;")
                btn.setText("卸载")
                btn.setToolTip("删除该模型的本地缓存，释放磁盘空间；之后需重新下载才能使用")
            else:
                if model_store.is_partial(size):
                    left = model_store.size_bytes(size) / (1024 * 1024)
                    lbl.setText(f"未下载（{left:.0f} MB 残留，重新下载会续传）")
                    lbl.setStyleSheet(f"color:{theme.WARN}; font-size:11px;")
                else:
                    est = model_store.EST_SIZE_MB.get(size, 0)
                    lbl.setText(f"未下载（约 {est} MB）")
                    lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
                btn.setText("下载")
                btn.setToolTip("下载到本地缓存（走 hf-mirror 镜像）")
            btn.setEnabled(not self._running and not self._transcribing
                           and self._mdl_busy != size)
        if self._mdl_busy:
            _lbl, btn = self._mdl_rows[self._mdl_busy]
            btn.setText("处理中…")
            btn.setEnabled(False)
        total = model_store.total_bytes() / (1024 * 1024)
        self.lbl_mdl_total.setText(
            f"当前缓存共占用 {total:.0f} MB；已下载的模型不会重复下载。")

    def _on_mdl_click(self, size: str) -> None:
        if self._mdl_busy:
            self._log("已有模型下载/卸载任务在进行，等它结束再操作")
            return
        # 实时/转写正在用这份模型：此时删文件会把正在跑的识别器弄坏
        if self._running or self._transcribing:
            self._log("实时字幕或转写正在运行，先停止再下载/卸载模型")
            return
        # 点下去时再判一次：状态可能刚被别处改过，已下载绝不触发下载
        if model_store.is_downloaded(size):
            used = model_store.size_bytes(size) / (1024 * 1024)
            ans = QMessageBox.question(
                self, "卸载模型",
                f"确定删除本地缓存的 {size} 模型吗？\n\n"
                f"将释放约 {used:.0f} MB 磁盘空间；删除后需要重新下载才能使用该模型。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
            # 先让识别器松手，否则 Windows 上文件被占用会删不掉
            self._release_loaded(size)
            action = "remove"
        else:
            action = "download"
        self._mdl_busy = size
        self._refresh_models()
        # 下载/卸载期间同样不许开始：同一份模型目录可能一边被删/写、一边被加载
        self.btn_start.setEnabled(False)
        self.btn_trs_start.setEnabled(False)
        self.cmb_model.setEnabled(False)
        self.cmb_dev.setEnabled(False)
        threading.Thread(target=self._mdl_task, args=(size, action), daemon=True).start()

    def _mdl_task(self, size: str, action: str) -> None:
        """后台线程：只做磁盘/网络操作，结果用信号回主线程。"""
        try:
            if action == "download":
                model_store.download(size, progress=self.sig_mdl_log.emit)
                self.sig_mdl_log.emit(f"模型 {size} 已就绪，可离线使用")
            else:
                freed = model_store.remove(size)
                self.sig_mdl_log.emit(
                    f"已卸载模型 {size}，释放 {freed / (1024 * 1024):.0f} MB 磁盘空间")
        except Exception as exc:
            name = "下载" if action == "download" else "卸载"
            self.sig_mdl_fail.emit(f"{name}模型 {size} 失败：{exc}")
        finally:
            self.sig_mdl_done.emit()

    def _on_mdl_done(self) -> None:
        self._mdl_busy = None
        self._restore_start_enabled()
        self._refresh_models()

    def _on_mdl_fail(self, msg: str) -> None:
        self._log(msg)
        self._mdl_busy = None
        self._restore_start_enabled()
        self._refresh_models()
        QMessageBox.warning(self, "模型缓存", msg)

    def _restore_start_enabled(self) -> None:
        """模型缓存任务结束后恢复按钮：按当前「运行中 / 转写中」决定谁能点。"""
        idle = not self._running and not self._transcribing
        self.btn_start.setEnabled(not self._transcribing)
        self.btn_trs_start.setEnabled(idle)
        self.cmb_model.setEnabled(idle)
        self.cmb_dev.setEnabled(idle)

    def _release_loaded(self, size: str) -> None:
        """卸载前释放已加载的同名模型，避免 Windows 上缓存文件被占用。

        必须看 `_loaded_size`（真正加载进内存的那一档）：`model_size` 会被
        `Pipeline.update_asr()` 改成用户刚在面板选的新值，用它判断会漏掉仍在内存里的
        旧模型，rmtree 直接报 WinError 32（缓存被删一半变成「未下载（残留）」）。
        """
        try:
            asr = getattr(getattr(self.ctrl, "pipeline", None), "asr", None)
            loaded = getattr(asr, "_loaded_size", None)
            if asr is not None and loaded == size:
                asr.unload()
        except Exception:
            pass

    def _on_wordmap(self) -> None:
        self.cfg.word_map = self.txt_wordmap.text().strip()
        self._save_cfg()

    def _on_names(self) -> None:
        self.cfg.name_roster = self.txt_names.text().strip()
        self.cfg.enable_namefix = True
        self._save_cfg()

    def _on_hotwords(self, *_a) -> None:
        self.cfg.enable_hotwords = self.chk_hotwords.isChecked()
        self._save_cfg()

    def _on_denoise(self, *_a) -> None:
        self.cfg.enable_denoise = self.chk_denoise.isChecked()
        self.cfg.denoise_strength = float(self.sp_denoise.value())
        self._save_cfg()
        self.ctrl.on_denoise_changed()

    def _on_vocal(self, *_a) -> None:
        self.cfg.vocal_extract = self.chk_vocal.isChecked()
        self.cfg.vocal_strength = float(self.sp_vocal.value())
        self._save_cfg()
        self.ctrl.on_vocal_changed()

    def _on_context_toggle(self, *_a) -> None:
        # 四个字段一起写：它们都只被「每句重读 cfg」的翻译/识别层使用，
        # 不需要通知控制器，也不跨线程碰控件
        self.cfg.enable_context = self.chk_context.isChecked()
        self.cfg.enable_homophone_fix = self.chk_homophone.isChecked()
        self.cfg.enable_free_context = self.chk_free.isChecked()
        self.cfg.context_window = int(self.sp_ctx.value())
        self.sp_ctx.setEnabled(self.chk_context.isChecked())
        self.chk_free.setEnabled(self.chk_context.isChecked())
        self._save_cfg()

    def _on_glossary_changed(self) -> None:
        # 逐字保存会很频繁：停手 600ms 再落盘
        self._glossary_timer.start()

    def _on_glossary(self) -> None:
        self.cfg.context_glossary = self.txt_glossary.toPlainText().strip()
        self._save_cfg()

    def _on_health_click(self) -> None:
        self.btn_health.setEnabled(False)
        self.btn_health.setText("检查中…")
        self.lbl_health.setText("正在逐个试通…")
        self.ctrl.on_channel_check()

    def _show_channel_result(self, text: str) -> None:
        self.txt_health.setPlainText(text)
        self.btn_health.setEnabled(True)
        self.btn_health.setText("通道健康检查")
        self.lbl_health.setText("完成")

    def _on_beam(self, *_a) -> None:
        self.cfg.beam_size = int(self.sp_beam.value())
        self.cfg.beam_auto = False   # 用户手动改过解码宽度，不再自动降档
        self._save_cfg()
        self.ctrl.on_beam_changed()

    def _on_trans_changed(self, *_a) -> None:
        idx = self.cmb_channel.currentIndex()
        self.cfg.channel = CHANNELS[idx] if 0 <= idx < len(CHANNELS) else "auto"
        self.cfg.target_lang = LANGS.get(self.cmb_lang.currentText(), "zh-Hans")
        self._save_cfg()
        self.ctrl.on_trans_changed()

    def _on_api_changed(self, *_a) -> None:
        idx = self.cmb_api.currentIndex()
        self.cfg.custom_key = self.txt_key.text().strip()
        if idx == 1:
            self.cfg.custom_type = "openai"
        elif idx == 2:
            self.cfg.custom_type = "deepl"
        self.cfg.custom_base = self.txt_base.text().strip()
        self.cfg.custom_model = self.txt_model.text().strip()
        self._save_cfg()
        self.ctrl.on_trans_changed()

    def _on_proxy(self) -> None:
        self.cfg.proxy = self.txt_proxy.text().strip()
        self._save_cfg()
        self.ctrl.on_trans_changed()

    def _on_sens(self, v: int) -> None:
        self.cfg.sensitivity = round(v / 100.0, 2)
        self.lbl_sens.setText(f"{self.cfg.sensitivity:.2f}")
        self.ctrl.on_vad_changed()
        self._save_cfg()

    def _on_vad(self, *_a) -> None:
        self.cfg.min_silence_ms = int(self.sp_silence.value())
        self.cfg.max_speech_ms = int(self.sp_max.value())
        self.ctrl.on_vad_changed()
        self._save_cfg()

    def _on_bg(self, v: int) -> None:
        self.cfg.opacity = round(v / 100.0, 2)
        self.ctrl.on_style_changed()

    def _on_style(self, *_a) -> None:
        self.cfg.font_size = int(self.sp_font.value())
        # 原文（副行）字号只由主字号派生，面板这个入口是唯一来源
        self.cfg.source_font_size = theme.source_font_size(self.cfg.font_size)
        self.cfg.show_source = self.chk_src.isChecked()
        self.cfg.max_lines = int(self.sp_lines.value())
        self.cfg.auto_clear_sec = float(self.sp_clear.value())
        self.ctrl.on_style_changed()

    # ------------------------------------------------------------ 外部更新
    def set_devices(self, devices: List, current=None) -> None:
        self.cmb_device.blockSignals(True)
        self.cmb_device.clear()
        sel = 0
        for i, d in enumerate(devices):
            self.cmb_device.addItem(d.label, d)
            if current is not None and d.name == current.name and d.backend == current.backend:
                sel = i
        self.cmb_device.setCurrentIndex(sel)
        self.cmb_device.blockSignals(False)

    # ------------------------------------------------------------ 实时记录
    def _live_time(self) -> float:
        """本轮时间轴读数：运行中 = 起点 + 已过秒数；停止后 = 冻结时刻。"""
        if self._live_t0 is None:
            return self._live_frozen
        return self._live_origin + (time.monotonic() - self._live_t0)

    def _live_seal_prev(self, end: float) -> None:
        """给还没收尾的记录补结束时间。

        时间轴是近似值：起始 = 该句原文上屏时刻，结束 = 下一句起始时刻，
        末句结束 = 停止时刻（见 _on_live_partial / set_running）。
        """
        for i, (a, b, src, tgt) in enumerate(self._live_records):
            if b <= a:
                self._live_records[i] = (a, max(a, end), src, tgt)

    def _on_live_partial(self, seq: int, src: str) -> None:
        """原文上屏：立刻落一条记录，译文先留空，保证丢帧跳号也不会错位。"""
        if self._transcribing or not src or self._live_t0 is None:
            return
        if seq in self._live_index:
            return
        now = self._live_time()
        self._live_seal_prev(now)
        idx = len(self._live_records)
        self._live_records.append((now, 0.0, src, ""))
        self._live_open[seq] = idx
        self._live_index[seq] = idx
        if self._fb_mode != "live":
            # 上一轮转写完成后切到了 file；新一轮实时字幕一来就切回最近句模式
            self._set_fb_mode("live")
        else:
            self._refresh_feedback_ui()
        self._live_render()
        self._refresh_counts()

    def _on_live_line(self, seq: int, src: str, tgt: str) -> None:
        """译文回来：按 seq 回填同一行；停止后才到的译文也能回填。"""
        if self._transcribing:
            return
        idx = self._live_index.get(seq)
        if idx is None:
            # 正常不会发生（译文一定跟在原文之后）。只在运行中补一条，
            # 停止后不再凭空新增记录。
            if self._live_t0 is None:
                return
            now = self._live_time()
            self._live_seal_prev(now)
            idx = len(self._live_records)
            self._live_records.append((now, 0.0, src, tgt))
            self._live_index[seq] = idx
        else:
            a, b, old_src, _old_tgt = self._live_records[idx]
            self._live_records[idx] = (a, b, src or old_src, tgt)
        self._live_open.pop(seq, None)
        self._live_render()
        self._refresh_counts()

    def _live_render(self) -> None:
        from core.transcribe import format_ts
        lines = []
        for a, _b, src, tgt in self._live_records:
            line = f"[{format_ts(a)}] {src}"
            if tgt:
                line += "\n    " + tgt
            lines.append(line)
        sb = self.txt_live.verticalScrollBar()
        saved_pos = sb.value()
        selected = self._fb_selected
        self.txt_live.setPlainText("\n".join(lines))
        self._rebuild_live_blocks()
        if (selected is not None and selected[0] == "live"
                and self._paint_selection("live", selected[1])):
            # 有选中句时暂停自动跟随，否则下一秒新字幕就把选中行顶走；
            # 改倒序号或清空选择后恢复跟到底部。
            sb.setValue(min(saved_pos, sb.maximum()))
        else:
            if selected is not None and selected[0] == "live":
                self._fb_selected = None
            sb.setValue(sb.maximum())
        has = bool(self._live_records)
        self.btn_live_clear.setEnabled(has)
        self.btn_live_txt.setEnabled(has)
        self.btn_live_srt.setEnabled(has)
        self.btn_live_ass.setEnabled(has)

    def _live_update_out_label(self) -> None:
        folder = (self.cfg.export_dir or "").strip()
        self.lbl_live_out.setText(
            f"导出目录：{folder}" if folder else "导出目录：文档（导出后自动记住）")
        self.lbl_live_out.setToolTip(folder)

    def _on_live_clear(self) -> None:
        if not self._live_records:
            return
        ans = QMessageBox.question(
            self, "清空实时记录",
            f"清空当前 {len(self._live_records)} 条实时记录？\n已导出的文件不受影响。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._live_records.clear()
        self._live_open.clear()
        self._live_index.clear()
        self._live_blocks = []
        if self._fb_selected is not None and self._fb_selected[0] == "live":
            self._fb_selected = None
        self.txt_live.clear()
        self.btn_live_clear.setEnabled(False)
        self.btn_live_txt.setEnabled(False)
        self.btn_live_srt.setEnabled(False)
        self.btn_live_ass.setEnabled(False)

    # ------------------------------------------------------------ 一键标错
    def _fb_pipeline(self):
        """拿到可用的流水线；离屏冒烟/未启动时不假设存在。"""
        pipe = getattr(self.ctrl, "pipeline", None)
        return pipe if hasattr(pipe, "recent_sentence") else None

    def _set_fb_mode(self, mode: str) -> None:
        """live：最近 20 句；file：本次转写的全部句段（按倒数序号选）。"""
        new_mode = "file" if mode == "file" else "live"
        if new_mode != self._fb_mode:
            # 换模式时旧选中句不再对应屏幕上的内容
            self._clear_selection()
        self._fb_mode = new_mode
        if self._fb_mode == "file":
            self.spn_fb_back.setRange(0, max(0, len(self._trs_segments) - 1))
        else:
            self.spn_fb_back.setRange(0, feedback.RECENT_MAX - 1)
        self._refresh_feedback_ui()

    # ------------------------------------------------------------ 热键
    def _hotkey_text(self) -> str:
        text = (getattr(self.cfg, "feedback_hotkey", "") or "").strip()
        return text or DEFAULT_HOTKEY

    def _apply_hotkey(self) -> None:
        """按当前开关/热键注册或注销；注册失败时保留应用内快捷键兜底。"""
        if not bool(getattr(self.cfg, "enable_feedback", True)):
            self._hotkey.unregister()
            self._hotkey_ok = False
            self._fb_shortcut.setKey(QKeySequence(self._hotkey_text()))
            self._fb_shortcut.setEnabled(True)
            self._set_hotkey_status(False, "标错已关闭，未注册全局热键")
            return
        text = self._hotkey_text()
        self._fb_shortcut.setKey(QKeySequence(text))
        ok, _msg = self._hotkey.register(text)
        self._hotkey_ok = ok
        # 全局注册成功就关掉应用内快捷键，否则一次按键会记两条
        self._fb_shortcut.setEnabled(not ok)

    def _on_hotkey_status(self, ok: bool, msg: str) -> None:
        self._set_hotkey_status(ok, msg)
        if not ok and bool(getattr(self.cfg, "enable_feedback", True)):
            self._log("标错热键：" + msg)

    def _set_hotkey_status(self, ok: bool, msg: str) -> None:
        self.lbl_fb.setText(msg)
        self.lbl_fb.setToolTip(msg)
        red = (not ok) and bool(getattr(self.cfg, "enable_feedback", True))
        self.lbl_fb.setStyleSheet("color:#d9534f" if red else "")

    def _on_hotkey_edit(self) -> None:
        text = self.key_fb.keySequence().toString(QKeySequence.PortableText)
        if not text:
            self.key_fb.setKey(QKeySequence(self._hotkey_text()))
            return
        try:
            parse_hotkey(text)
        except ValueError as exc:
            # 格式无效就回退到上一个键，别把可用热键注销掉
            self._set_hotkey_status(False, "%s；仍用 %s" % (exc, self._hotkey_text()))
            self.key_fb.setKey(QKeySequence(self._hotkey_text()))
            return
        if text != self._hotkey_text():
            self.cfg.feedback_hotkey = text
            try:
                self._save_cfg()
            except Exception:
                pass
            self._log("标错热键改为：%s" % text)
        self._apply_hotkey()

    def shutdown_hotkey(self) -> None:
        """退出路径统一注销，可重复调用。"""
        try:
            self._hotkey.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------ 选中句
    @staticmethod
    def _block_ranges(two_line_flags: List[bool]) -> List[Tuple[int, int, int]]:
        """把「每句 1~2 行」的纯文本映射成 (起始块, 结束块, 记录下标)。"""
        ranges = []
        block = 0
        for i, two in enumerate(two_line_flags):
            end = block + (1 if two else 0)
            ranges.append((block, end, i))
            block = end + 1
        return ranges

    def _rebuild_live_blocks(self) -> None:
        self._live_blocks = self._block_ranges([bool(tgt) for _a, _b, _s, tgt in self._live_records])

    def _rebuild_trs_blocks(self, shown_count: int) -> None:
        flags = [bool(getattr(self._trs_segments[i], "tgt", "")) for i in range(shown_count)]
        self._trs_blocks = self._block_ranges(flags)

    def _paint_selection(self, mode: str, idx: int, ensure: bool = False) -> bool:
        widget = self.txt_live if mode == "live" else self.txt_trs_result
        ranges = self._live_blocks if mode == "live" else self._trs_blocks
        match = next((r for r in ranges if r[2] == idx), None)
        if match is None:
            return False
        doc = widget.document()
        first = doc.findBlockByNumber(match[0])
        last = doc.findBlockByNumber(match[1])
        if not first.isValid() or not last.isValid():
            return False
        cur = QTextCursor(first)
        cur.setPosition(last.position() + len(last.text()))
        widget.setTextCursor(cur)
        if ensure:
            widget.ensureCursorVisible()
        return True

    def _select_at(self, widget, pos) -> None:
        if widget is self.txt_live.viewport():
            mode, ranges, view = "live", self._live_blocks, self.txt_live
        elif widget is self.txt_trs_result.viewport():
            mode, ranges, view = "file", self._trs_blocks, self.txt_trs_result
        else:
            return
        block = view.cursorForPosition(pos).blockNumber()
        for b0, b1, idx in ranges:
            if b0 <= block <= b1:
                self._fb_selected = (mode, idx)
                self._paint_selection(mode, idx, ensure=True)
                if mode == "file":
                    # 同步倒序号，方便用户看到「这是第几句」
                    self._fb_syncing = True
                    try:
                        self.spn_fb_back.setValue(len(self._trs_segments) - 1 - idx)
                    finally:
                        self._fb_syncing = False
                self._refresh_feedback_ui()
                self._log("已选中 %s，点「标错」或按热键记录" % self._fb_where(mode, idx))
                return

    def _fb_where(self, mode: str, idx: int) -> str:
        if mode == "file":
            return "转写第 %d/%d 句" % (idx + 1, len(self._trs_segments))
        return "倒数第 %d 句" % (len(self._live_records) - idx)

    def _clear_selection(self) -> None:
        if self._fb_selected is None:
            return
        self._fb_selected = None
        for w in (self.txt_live, self.txt_trs_result):
            cur = w.textCursor()
            cur.clearSelection()
            w.setTextCursor(cur)

    def _live_seq_for(self, idx: int):
        for seq, i in self._live_index.items():
            if i == idx:
                return seq
        return None

    def _channel_for_seq(self, seq) -> str:
        pipe = self._fb_pipeline()
        if pipe is None or seq is None:
            return ""
        for k in range(feedback.RECENT_MAX):
            snap = pipe.recent_sentence(k)
            if snap is None:
                break
            if snap.get("seq") == seq:
                return snap.get("channel", "")
        return ""

    def _fb_target(self):
        """选中句优先；否则回落到「倒数第 N 句」。"""
        if self._fb_selected is not None:
            mode, idx = self._fb_selected
            if mode == "live" and 0 <= idx < len(self._live_records):
                return ("live_sel", idx)
            if mode == "file" and 0 <= idx < len(self._trs_segments):
                return ("file", idx)
            self._fb_selected = None
        back = int(self.spn_fb_back.value())
        if self._fb_mode == "file" and self._trs_segments:
            idx = len(self._trs_segments) - 1 - back
            return ("file", idx) if 0 <= idx < len(self._trs_segments) else None
        pipe = self._fb_pipeline()
        if pipe is not None and pipe.recent_sentence(back) is not None:
            return ("live", back)
        return None

    def _refresh_counts(self) -> None:
        """刷新顶栏本场计数。数据源是 pipeline.counts()（只读快照），不做任何计算。"""
        c = {}
        pipe = self._fb_pipeline()
        if pipe is not None and hasattr(pipe, "counts"):
            try:
                c = pipe.counts()
            except Exception:
                c = {}
        text = "上屏 %d · 有译文 %d · 失败 %d · 标错 %d" % (
            int(c.get("final", 0)), int(c.get("tgt", 0)),
            int(c.get("fail", 0)), int(self._fb_count))
        if self.lbl_counts.text() != text:
            self.lbl_counts.setText(text)

    def _refresh_feedback_ui(self) -> None:
        on = bool(getattr(self.cfg, "enable_feedback", True))
        for cb in self._fb_tags.values():
            cb.setEnabled(on)
        self.chk_fb.setChecked(on)
        self.spn_fb_back.setEnabled(on)
        self.key_fb.setEnabled(on)
        self.btn_fb.setEnabled(on and self._fb_target() is not None)

    def _on_feedback_toggle(self, checked: bool) -> None:
        self.cfg.enable_feedback = bool(checked)
        try:
            self._save_cfg()
        except Exception:
            pass
        self._apply_hotkey()
        self._refresh_feedback_ui()

    def _on_fb_back_changed(self, _v: int) -> None:
        if self._fb_syncing:
            return
        # 用户主动改倒序号 = 放弃选中句，回到「最近 N 句」方式
        self._clear_selection()
        self._refresh_feedback_ui()

    def _on_fb_tag_changed(self, _checked: bool) -> None:
        # 类型只是写进记录的标签，不需要动配置，也不需要刷新可用状态
        pass

    def mark_feedback(self) -> None:
        """把当前目标句 + 勾选的错误类型写进 jsonl（后台线程，按键路径无 IO）。"""
        if not bool(getattr(self.cfg, "enable_feedback", True)):
            self._log("标错已关闭：先勾选「启用标错」")
            return
        tags = [k for k, cb in self._fb_tags.items() if cb.isChecked()]
        if not tags:
            # 快路径：来不及选类型时也要能记下来，归入「其他」
            tags = ["other"]
        target = self._fb_target()
        if target is None:
            self._log("没有可标记的句子：先开始翻译，或先完成一次转写")
            self._refresh_feedback_ui()
            return
        if target[0] == "file":
            idx = target[1]
            seg = self._trs_segments[idx]
            session = self._trs_session or feedback.new_session_id("file")
            tgt_text = getattr(seg, "tgt", "")
            entry = feedback.make_entry(
                cfg=self.cfg, session_id=session, source="file", seq=idx,
                back=len(self._trs_segments) - 1 - idx, src=seg.text,
                tgt=tgt_text, channel=getattr(seg, "channel", ""),
                tags=tags, start_s=seg.start, end_s=seg.end,
                has_audio=bool(self._trs_path), media=self._trs_path,
                pipeline=self._fb_pipeline())
            where = "转写第 %d/%d 句" % (idx + 1, len(self._trs_segments))
        elif target[0] == "live_sel":
            # 点击选中的实时句：原文/译文取面板记录，通道按 seq 回查流水线快照
            idx = target[1]
            a, b, src, tgt = self._live_records[idx]
            pipe = self._fb_pipeline()
            seq = self._live_seq_for(idx)
            entry = feedback.make_entry(
                cfg=self.cfg, session_id=getattr(pipe, "session_id", "") or "live-unknown",
                source="live", seq=seq if seq is not None else -1,
                back=len(self._live_records) - 1 - idx, src=src, tgt=tgt,
                channel=self._channel_for_seq(seq), tags=tags,
                start_s=a, end_s=b, has_audio=False, media="", pipeline=pipe)
            where = "选中第 %d 句" % (idx + 1)
        else:
            pipe = self._fb_pipeline()
            snap = pipe.recent_sentence(int(self.spn_fb_back.value()))
            if snap is None:
                # 极端情况：刚检查完目标就被新句子挤出窗口
                self._log("这句已经滚出最近 %d 句窗口，未记录" % feedback.RECENT_MAX)
                return
            session = getattr(pipe, "session_id", "") or feedback.new_session_id("live")
            entry = feedback.make_entry(
                cfg=self.cfg, session_id=session, source="live", seq=snap["seq"],
                back=int(self.spn_fb_back.value()), src=snap.get("src", ""),
                tgt=snap.get("tgt", ""), channel=snap.get("channel", ""),
                tags=tags, start_s=snap.get("ts_src"), end_s=snap.get("ts_tgt"),
                has_audio=False, media="", pipeline=pipe)
            where = "倒数第 %d 句" % (int(self.spn_fb_back.value()) + 1)
        store = getattr(self.ctrl, "feedback", None)
        if store is None or not hasattr(store, "submit"):
            self._log("标错写手未初始化，本条未记录")
            return
        ok = bool(store.submit(entry))
        if ok:
            self._fb_count += 1
            self._refresh_counts()
            names = "/".join(feedback.TAG_NAMES.get(t, t) for t in entry["tags"])
            self._log("已标错 #%d：%s · %s" % (self._fb_count, where, names))
            self.btn_fb.setText("已记录")
            QTimer.singleShot(900, lambda: self.btn_fb.setText("标错"))
        else:
            self._log("标错写入队列失败（队列满或序列化异常），本条未记录")

    def _export_live(self, kind: str) -> None:
        """导出实时记录：复用 core/transcribe 的 TXT / SRT / ASS 格式。

        文件名固定为 实时记录_YYYYmmdd_HHMMSS.txt/.srt/.ass，目录取配置里的导出目录
        （未设置或已失效时落到「文档」，成功导出后写回配置）。
        """
        if not self._live_records:
            return
        from core.transcribe import to_ass, to_srt, to_txt
        segments = []
        for a, b, src, tgt in self._live_records:
            # SRT/TXT/ASS 都复用同一个 Segment 结构；有译文时原文与译文分两行显示
            text = f"{src}\n{tgt}" if tgt else src
            segments.append((a, max(a, b), text))
        data = {"txt": to_txt, "ass": to_ass}.get(kind, to_srt)(segments)
        folder = (self.cfg.export_dir or "").strip()
        if not folder or not os.path.isdir(folder):
            folder = (QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
                      or os.path.expanduser("~"))
        name = f"实时记录_{datetime.now():%Y%m%d_%H%M%S}.{kind}"
        path = os.path.join(folder, name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        self.cfg.export_dir = os.path.dirname(os.path.abspath(path))
        self._save_cfg()
        self.lbl_live_out.setText(f"已导出：{path}")
        self.lbl_live_out.setToolTip(path)
        self._log(f"已导出实时记录：{path}")

    # ------------------------------------------------------------ 音频转文字
    def _on_trs_pick(self) -> None:
        from core.transcribe import FILE_FILTER
        path, _ = QFileDialog.getOpenFileName(
            self, "选择要转写的音频 / 视频", self.cfg.transcribe_dir or "", FILE_FILTER)
        if not path:
            return
        self.cfg.transcribe_dir = os.path.dirname(path)
        self._save_cfg()
        self._set_trs_source(path)

    def _set_trs_source(self, path: str) -> None:
        """换源文件时清掉上一次结果：别让旧文本被导出成新文件的字幕。"""
        self._trs_path = path
        self._trs_segments = []
        self._trs_total = 0.0
        self._trs_session = ""
        self._set_fb_mode("live")
        self.lbl_trs_file.setText(os.path.basename(path))
        self.lbl_trs_file.setToolTip(path)
        self.lbl_trs_out.setText(f"导出目录：{os.path.dirname(path)}")
        self.txt_trs_result.clear()
        self.bar_trs.setValue(0)
        for b in self._trs_export_btns:
            b.setEnabled(False)
        self.lbl_trs_state.setText("已选择文件，点「开始转写」。")

    def _on_trs_opts_changed(self, *_args) -> None:
        """转写选项写回配置：下次启动仍按上次勾选的状态跑。"""
        self.cfg.trs_translate = bool(self.chk_trs_translate.isChecked())
        self.cfg.trs_max_chars = int(self.spn_trs_chars.value())
        self._save_cfg()

    def _on_trs_start(self) -> None:
        if not self._trs_path:
            QMessageBox.information(self, "音频转文字", "先点「选择文件…」选一个音频或视频文件。")
            return
        if self._running:
            QMessageBox.information(
                self, "音频转文字",
                "实时字幕正在运行。两者共用同一份识别模型，"
                "先点顶部「停止翻译」，再回来转写。")
            return
        self.txt_trs_result.clear()
        self._trs_segments = []
        self._trs_blocks = []
        self._trs_total = 0.0
        # 每次转写换新会话 id，标错记录才能和这一份文件对应起来
        self._trs_session = feedback.new_session_id("file")
        self._fb_count = 0
        for b in self._trs_export_btns:
            b.setEnabled(False)
        self.bar_trs.setValue(0)
        self.set_transcribing(True)
        self.lbl_trs_state.setText("正在准备模型…")
        self.ctrl.transcribe_start(
            self._trs_path,
            translate=bool(self.chk_trs_translate.isChecked()),
            max_chars=int(self.spn_trs_chars.value()),
        )

    def _on_trs_cancel(self) -> None:
        self.ctrl.transcribe_cancel()

    def _on_trs_progress(self, pct: int) -> None:
        self.bar_trs.setValue(int(pct))

    def _on_trs_status(self, text: str) -> None:
        self.lbl_trs_state.setText(text)

    def _on_trs_done(self, payload) -> None:
        from core.transcribe import format_ts
        segments, total, cancelled = payload
        self._trs_segments = list(segments)
        self._trs_total = float(total or 0.0)
        if self._fb_selected is not None and self._fb_selected[0] == "file":
            self._fb_selected = None
        # 转写完成后标错目标切到这份结果；没有句段就退回实时快照
        self._set_fb_mode("file" if self._trs_segments else "live")
        self.set_transcribing(False)
        if not segments:
            self.lbl_trs_state.setText(
                "已取消，未转出内容。" if cancelled else
                "没有识别出文字：确认文件里有人声，或换更大的模型再试。")
            return
        self.bar_trs.setValue(100)
        has_tgt = any(getattr(s, "tgt", "") for s in segments)
        # 长音频可能上万句：预览截断，导出始终是完整内容
        lines = []
        for seg in segments:
            a, b, t = seg.start, seg.end, seg.text
            line = f"[{format_ts(a)} → {format_ts(b)}] {t}"
            if getattr(seg, "tgt", ""):
                line += f"\n    {seg.tgt}"
            lines.append(line)
        shown = lines[:400]
        text = "\n".join(shown)
        if len(lines) > len(shown):
            text += f"\n…（共 {len(lines)} 句，预览只显示前 {len(shown)} 句，导出为完整内容）"
        self.txt_trs_result.setPlainText(text)
        # 预览与导出同源，但点击标记只覆盖预览里真实显示出来的句子
        self._rebuild_trs_blocks(len(shown))
        for b in self._trs_export_btns:
            b.setEnabled(True)
        # 没开翻译时这两个按钮导不出东西，直接置灰
        self.btn_trs_bi.setEnabled(has_tgt)
        self.btn_trs_tsrt.setEnabled(has_tgt)
        tail = "；可导出 TXT / SRT / ASS"
        if has_tgt:
            tail += " / 双语 SRT / 译文 SRT"
        tail += " / 工程 JSON。"
        if cancelled:
            self.lbl_trs_state.setText(
                f"已取消：保留已转出的 {len(segments)} 句{tail}")
        else:
            self.lbl_trs_state.setText(
                f"完成：{len(segments)} 句 · 音频时长 {format_ts(total)}{tail}")

    def _on_trs_fail(self, msg: str) -> None:
        self.set_transcribing(False)
        self.lbl_trs_state.setText(msg)
        QMessageBox.warning(self, "音频转文字", msg)

    def _export_trs(self, kind: str) -> None:
        if not self._trs_segments:
            return
        from core.transcribe import (to_ass, to_project_json, to_srt,
                                     to_translation_srt, to_txt)
        base = os.path.splitext(self._trs_path)[0] or "转写结果"
        filt = {"txt": ("TXT (*.txt)", "txt"), "srt": ("SRT (*.srt)", "srt"),
                "ass": ("ASS (*.ass)", "ass"),
                "srt2": ("双语 SRT (*.srt)", "双语.srt"),
                "tsrt": ("译文 SRT (*.srt)", "译文.srt"),
                "json": ("工程 JSON (*.json)", "json")}.get(kind, ("SRT (*.srt)", "srt"))
        title = os.path.basename(self._trs_path) or "Translator"
        path, _ = QFileDialog.getSaveFileName(
            self, f"导出 {filt[0].split(' (')[0]}", f"{base}.{filt[1]}", filt[0])
        if not path:
            return
        if kind == "txt":
            data = to_txt(self._trs_segments)
        elif kind == "ass":
            data = to_ass(self._trs_segments, title=title,
                          bilingual=any(s.tgt for s in self._trs_segments))
        elif kind == "srt2":
            data = to_srt(self._trs_segments, bilingual=True)
        elif kind == "tsrt":
            data = to_translation_srt(self._trs_segments)
        elif kind == "json":
            segs = self._trs_segments
            # 元数据写「这次实际用的」：load() 可能已按显存降级到 small，
            # 设备也可能从 auto 落到具体值；duration 用解码出的音频总时长
            asr = getattr(getattr(self.ctrl, "pipeline", None), "asr", None)
            loaded = getattr(asr, "_loaded_size", None) or getattr(asr, "model_size", None)
            model = loaded or self.cfg.model_size
            device = getattr(asr, "dev_info", "") or self.cfg.asr_device
            if getattr(asr, "_force_ja", False):
                language = "ja"
            elif self.cfg.source_lang in ("", "auto"):
                language = "auto"
            else:
                language = self.cfg.source_lang
            duration = self._trs_total or max((s.end for s in segs), default=0.0)
            data = to_project_json(
                segs, media_path=self._trs_path, duration=duration,
                language=language, model=model, device=device,
                beam_size=5, word_timestamps=any(s.words for s in segs))
        else:
            data = to_srt(self._trs_segments)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")
            return
        self.lbl_trs_state.setText(f"已导出：{path}")
        self._log(f"已导出转写结果：{path}")

    def set_transcribing(self, busy: bool) -> None:
        """转写期间锁住「开始翻译 / 换模型 / 下载卸载」，避免两份任务抢同一份模型。"""
        self._transcribing = busy
        self.btn_trs_start.setEnabled(not busy)
        self.btn_trs_cancel.setEnabled(busy)
        self.btn_trs_pick.setEnabled(not busy)
        self.btn_start.setEnabled(not busy)
        self.cmb_model.setEnabled(not busy and not self._running)
        self.cmb_dev.setEnabled(not busy and not self._running)
        self._refresh_models()

    def set_running(self, running: bool, paused: bool = False) -> None:
        if running:
            if self._live_t0 is None:
                self._live_t0 = time.monotonic()
            if self._fb_mode != "live":
                self._set_fb_mode("live")
        elif self._live_t0 is not None:
            # 停止：末句结束时间取停止时刻；只有原文、没有译文的句子原样留在记录里
            self._live_frozen = self._live_origin + (time.monotonic() - self._live_t0)
            self._live_t0 = None
            self._live_seal_prev(self._live_frozen)
            self._live_open.clear()
            self._live_origin = self._live_frozen
            self._live_render()
        self._refresh_feedback_ui()
        self._refresh_counts()
        self.btn_start.setText("停止翻译" if running else "开始翻译")
        # 动态属性 + 重新 polish：让 QSS 里 [running="true"] 那套配色生效
        self.btn_start.setProperty("running", "true" if running else "false")
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)
        self.btn_pause.setText("继续" if paused else "暂停")
        self.btn_pause.setEnabled(running)
        # 互斥：运行中不能开转写；转写中「开始翻译」已由 set_transcribing 关掉
        self.cmb_model.setEnabled(not running and not self._transcribing)
        self.cmb_dev.setEnabled(not running and not self._transcribing)
        self.btn_trs_start.setEnabled(not running and not self._transcribing)
        # 运行中不让动模型缓存：换/删文件会让正在跑的识别器失去模型
        self._running = running
        self._refresh_models()

    def set_subtitle_visible(self, visible: bool) -> None:
        self.btn_sub.setText("隐藏字幕" if visible else "显示字幕")

    def _set_status(self, text: str) -> None:
        self.lbl_state.setText(text)

    def _save_cfg(self, reason: str = "面板设置") -> bool:
        """面板控件的保存统一入口：失败时由 save 回调弹出顶栏提示。"""
        return bool(self.cfg.save(reason=reason))

    def _on_save_event(self, ev) -> None:
        # save() 可能来自采集/翻译线程：只投递信号，回主线程再改控件
        self.sig_save_state.emit(ev)

    def _on_save_state(self, ev) -> None:
        """写盘结果上屏：成功收起提示条，失败写明键名/错误并标注显示值未保存。"""
        ev = ev or {}
        kind = ev.get("kind")
        if kind == "ok":
            self._save_failed = False
            self.lbl_save.clear()
            self.save_bar.setVisible(False)
            return
        if kind == "validate":
            detail = "；".join(ev.get("errors") or []) or "配置校验未通过"
        else:
            detail = ev.get("detail") or "未知错误"
        text = ("设置未保存：%s。config.json 未改动，当前显示值未保存。"
                "可修正后点「重试保存」。" % detail)
        if ev.get("reason"):
            text += "（来源：%s）" % ev["reason"]
        self._save_failed = True
        self.lbl_save.setText(text)
        self.save_bar.setVisible(True)

    def _set_level(self, v: float) -> None:
        self.bar.setValue(int(max(0.0, min(1.0, v)) * 100))
        self._update_diag()

    def _update_diag(self) -> None:
        """实时显示音量与触发线的关系——有声音却不出字幕时，看这里就知道原因。"""
        # 配置里把 min_speech_ms 调到 200ms 以下时，VAD 会切出更短的句子，
        # 但采集线程只放行 >0.2s 的段（core/pipeline.py 的硬门槛），短句会被静默丢弃。
        # 默认 250ms 不受影响；这里给调过配置的人一个看得见的提示。
        if int(getattr(self.cfg, "min_speech_ms", 250) or 0) < 200:
            self.lbl_diag.setText(
                f"min_speech_ms={int(self.cfg.min_speech_ms)}ms 低于 200ms："
                "采集线程只放行时长 >0.2s 的句子，更短的会被丢弃。建议调回 250ms 以上。")
            self.lbl_diag.setStyleSheet(f"color:{theme.WARN}; font-size:11px;")
            return
        if float(getattr(self.cfg, "sensitivity", 0.7) or 0) > 0.92:
            self.lbl_diag.setText("灵敏度接近拉满：背景音乐会被一直算作语音，句子切不开，"
                                  "字幕要等好几秒才出。建议调回 0.7 左右。")
            self.lbl_diag.setStyleSheet(f"color:{theme.DANGER}; font-size:11px;")
            return
        try:
            vad = self.ctrl.pipeline.vad
        except Exception:
            return
        lv = float(getattr(vad, "last_level", 0.0) or 0.0)
        th = float(getattr(vad, "threshold", 0.0) or 0.0)
        if lv <= 0.0 and th <= 0.0:
            return
        if lv < 0.0015:
            txt = f"音量 {lv:.4f} —— 几乎没声音，检查采集设备是不是选错了"
            color = theme.DANGER
        elif lv > th:
            txt = f"音量 {lv:.4f} > 触发线 {th:.4f} —— 正常，正在送去识别"
            color = theme.OK
        else:
            txt = (f"音量 {lv:.4f} < 触发线 {th:.4f} —— 音量没到触发线，"
                   f"请把灵敏度调高（或调大播放器音量）")
            color = theme.DANGER
        self.lbl_diag.setText(txt)
        self.lbl_diag.setStyleSheet(f"color:{color}; font-size:11px;")

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def eventFilter(self, obj, ev):
        """统一拦截取值控件的滚轮：没焦点时让给页面滚动。

        面板里 QComboBox / QSpinBox / QDoubleSpinBox / QSlider 很多，逐个装过滤器
        容易漏、以后新加控件还会忘，所以在 QApplication 上统一过滤：
        - 控件自己拿到焦点（用户先点过它）→ 放行，滚轮照旧调值；
        - 没焦点 → 把滚轮事件转给滚动区 viewport，页面按原生手感滚动，值不变。
        只拦 Wheel：键盘上下键、拖动滑块走 keyPress/mouseMove，不受影响。
        """
        if (ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton
                and obj in (self.txt_live.viewport(), self.txt_trs_result.viewport())):
            # 点击整句选中（而不是把光标落在一个字上），标错时按句取
            self._select_at(obj, ev.position().toPoint())
            return True
        if ev.type() == QEvent.Wheel and hasattr(self, "scroll"):
            w = obj
            # 数值框内部可能是 QLineEdit，往上找一两层认领控件
            for _ in range(3):
                if isinstance(w, (QComboBox, QAbstractSpinBox, QSlider)):
                    if w.hasFocus():
                        break
                    QApplication.sendEvent(self.scroll.viewport(), ev)
                    return True
                w = w.parentWidget() if hasattr(w, "parentWidget") else None
                if w is None:
                    break
        return super().eventFilter(obj, ev)

    def closeEvent(self, event):
        # 关闭主窗口 = 真正退出程序，不再「最小化到托盘」。
        # 以前 ignore()+hide() 会让进程留在后台、字幕窗继续挂在桌面上，
        # 单实例锁也一直被占着，再双击只会提示「已经在运行」。收尾交给控制器。
        # 术语表是延迟保存：退出前把还没落盘的那一次补上，别让刚打的字丢掉
        if self._glossary_timer.isActive():
            self._glossary_timer.stop()
            self._on_glossary()
        # 先注销系统级热键，避免进程退出后热键仍被占用
        self.shutdown_hotkey()
        event.accept()
        self.ctrl.quit()
