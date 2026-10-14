# -*- coding: utf-8 -*-
"""控制面板：设备 / 模型 / 翻译 / 字幕 参数调节与运行状态。"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QComboBox,
    QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPlainTextEdit,
    QProgressBar, QMessageBox, QLineEdit,
)

from core.asr import MODEL_SIZES
from core.translator import CHANNELS, CHANNEL_NAMES, LANGS


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

    def __init__(self, cfg, controller):
        super().__init__()
        self.cfg = cfg
        self.ctrl = controller
        self.setWindowTitle("VTuber 实时翻译器")
        self.setWindowIcon(_icon())
        self.setMinimumWidth(520)
        self.sig_status.connect(self._set_status)
        self.sig_level.connect(self._set_level)
        self.sig_log.connect(self._log)
        self.sig_channel_result.connect(self._show_channel_result)
        self._build()

    # ------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # 状态
        box_state = QGroupBox("运行状态")
        h = QHBoxLayout(box_state)
        self.lbl_state = QLabel("未启动")
        self.lbl_state.setStyleSheet("font-weight:bold;font-size:14px;color:#5ec8ff;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedWidth(150)
        self.bar.setTextVisible(False)
        h.addWidget(self.lbl_state, 1)
        h.addWidget(QLabel("音量"))
        h.addWidget(self.bar)
        root.addWidget(box_state)

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

        # 识别
        box_asr = QGroupBox("语音识别（Whisper）")
        f = QFormLayout(box_asr)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODEL_SIZES)
        # 填初值时先屏蔽信号：否则一构建就触发保存，把 gpu_ok / last_channel 冲掉
        self.cmb_model.blockSignals(True)
        self.cmb_model.setCurrentText(self.cfg.model_size)
        self.cmb_model.blockSignals(False)
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
        f.addRow("解码宽度", self.sp_beam)
        self.cmb_model.setToolTip(
            "small：最快（约 0.2s），直播口语识别一般。\n"
            "medium：口语/专有名词识别明显更准，GPU 下约 0.5s。\n"
            "口语多的直播建议换 medium。\n"
            "large-v3：识别最准（尤其口音/外来语/嘈杂），GPU 下约 1s，最吃显存；\n"
            "6~8GB 显卡会自动改用 int8 跑。\n"
            "显存被游戏占用放不下时会自动临时降级到 medium/small 保速度\n"
            "（状态栏会提示），关掉游戏重启即可回到大模型。")
        root.addWidget(box_asr)

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
        self.lbl_diag.setStyleSheet("color:#888; font-size:11px;")
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

        # 按钮
        row8 = QHBoxLayout()
        self.btn_start = QPushButton("开始翻译")
        self.btn_start.setFixedHeight(38)
        self.btn_start.clicked.connect(self.ctrl.toggle_start)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setFixedHeight(38)
        self.btn_pause.clicked.connect(self.ctrl.toggle_pause)
        self.btn_sub = QPushButton("隐藏字幕")
        self.btn_sub.setFixedHeight(38)
        self.btn_sub.clicked.connect(self.ctrl.toggle_subtitle)
        row8.addWidget(self.btn_start, 2)
        row8.addWidget(self.btn_pause, 1)
        row8.addWidget(self.btn_sub, 1)
        root.addLayout(row8)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setFixedHeight(150)
        self.log.setStyleSheet("background:#101318;color:#c8d0dc;font-family:Consolas;")
        root.addWidget(self.log)

        tip = QLabel("提示：字幕窗可用鼠标拖动，右下角拖动改宽度，右键可调字号/背景/穿透。")
        tip.setStyleSheet("color:#8892a4;")
        tip.setWordWrap(True)
        root.addWidget(tip)

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
        self.cfg.save()
        self.ctrl.on_model_changed()

    def _on_wordmap(self) -> None:
        self.cfg.word_map = self.txt_wordmap.text().strip()
        self.cfg.save()

    def _on_names(self) -> None:
        self.cfg.name_roster = self.txt_names.text().strip()
        self.cfg.enable_namefix = True
        self.cfg.save()

    def _on_hotwords(self, *_a) -> None:
        self.cfg.enable_hotwords = self.chk_hotwords.isChecked()
        self.cfg.save()

    def _on_denoise(self, *_a) -> None:
        self.cfg.enable_denoise = self.chk_denoise.isChecked()
        self.cfg.denoise_strength = float(self.sp_denoise.value())
        self.cfg.save()
        self.ctrl.on_denoise_changed()

    def _on_vocal(self, *_a) -> None:
        self.cfg.vocal_extract = self.chk_vocal.isChecked()
        self.cfg.vocal_strength = float(self.sp_vocal.value())
        self.cfg.save()
        self.ctrl.on_vocal_changed()

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
        self.cfg.save()
        self.ctrl.on_beam_changed()

    def _on_trans_changed(self, *_a) -> None:
        idx = self.cmb_channel.currentIndex()
        self.cfg.channel = CHANNELS[idx] if 0 <= idx < len(CHANNELS) else "auto"
        self.cfg.target_lang = LANGS.get(self.cmb_lang.currentText(), "zh-Hans")
        self.cfg.save()
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
        self.cfg.save()
        self.ctrl.on_trans_changed()

    def _on_proxy(self) -> None:
        self.cfg.proxy = self.txt_proxy.text().strip()
        self.cfg.save()
        self.ctrl.on_trans_changed()

    def _on_sens(self, v: int) -> None:
        self.cfg.sensitivity = round(v / 100.0, 2)
        self.lbl_sens.setText(f"{self.cfg.sensitivity:.2f}")
        self.ctrl.on_vad_changed()

    def _on_vad(self, *_a) -> None:
        self.cfg.min_silence_ms = int(self.sp_silence.value())
        self.cfg.max_speech_ms = int(self.sp_max.value())
        self.ctrl.on_vad_changed()

    def _on_bg(self, v: int) -> None:
        self.cfg.opacity = round(v / 100.0, 2)
        self.ctrl.on_style_changed()

    def _on_style(self, *_a) -> None:
        self.cfg.font_size = int(self.sp_font.value())
        self.cfg.source_font_size = max(9, int(self.sp_font.value()) - 10)
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

    def set_running(self, running: bool, paused: bool = False) -> None:
        self.btn_start.setText("停止翻译" if running else "开始翻译")
        self.btn_pause.setText("继续" if paused else "暂停")
        self.btn_pause.setEnabled(running)
        self.cmb_model.setEnabled(not running)
        self.cmb_dev.setEnabled(not running)

    def set_subtitle_visible(self, visible: bool) -> None:
        self.btn_sub.setText("隐藏字幕" if visible else "显示字幕")

    def _set_status(self, text: str) -> None:
        self.lbl_state.setText(text)

    def _set_level(self, v: float) -> None:
        self.bar.setValue(int(max(0.0, min(1.0, v)) * 100))
        self._update_diag()

    def _update_diag(self) -> None:
        """实时显示音量与触发线的关系——有声音却不出字幕时，看这里就知道原因。"""
        if float(getattr(self.cfg, "sensitivity", 0.7) or 0) > 0.92:
            self.lbl_diag.setText("灵敏度接近拉满：背景音乐会被一直算作语音，句子切不开，"
                                  "字幕要等好几秒才出。建议调回 0.7 左右。")
            self.lbl_diag.setStyleSheet("color:#c62828; font-size:11px;")
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
            color = "#c62828"
        elif lv > th:
            txt = f"音量 {lv:.4f} > 触发线 {th:.4f} —— 正常，正在送去识别"
            color = "#2e7d32"
        else:
            txt = (f"音量 {lv:.4f} < 触发线 {th:.4f} —— 音量没到触发线，"
                   f"请把灵敏度调高（或调大播放器音量）")
            color = "#c62828"
        self.lbl_diag.setText(txt)
        self.lbl_diag.setStyleSheet(f"color:{color}; font-size:11px;")

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.ctrl.on_panel_hidden()
