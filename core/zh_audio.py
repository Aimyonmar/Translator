# -*- coding: utf-8 -*-
"""中文音频 → 英文：独立预设模块。

把「抓取电脑正在播放的中文音频」需要同时成立的三处配置收敛到一处：
音频语言（Whisper 解码语言）、识别模型（不能是日语专用档）、目标语言。
面板的「中文音频」页只调用这里，避免三个下拉各改一半、实际跑起来仍是日语。
"""
from __future__ import annotations

SOURCE = "zh"
TARGET = "en"
# kotoba-whisper-v2.0 是日语专用档；中文音频必须换通用档
FALLBACK_MODEL = "small"

SOURCE_LABELS = {"ja": "日本語", "zh": "中文", "en": "英语"}
TARGET_LABELS = {"zh-Hans": "简体中文", "zh-Hant": "繁體中文", "en": "English", "ko": "한국어"}


def is_ja_only_model(size: str) -> bool:
    return "kotoba" in (size or "").lower()


def apply(cfg) -> list:
    """把三处配置一次性切到中文音频 → 英文，返回需要告知用户的变更说明。"""
    notes = []
    if getattr(cfg, "source_lang", "") != SOURCE:
        cfg.source_lang = SOURCE
        notes.append("音频语言 → 中文")
    if getattr(cfg, "target_lang", "") != TARGET:
        cfg.target_lang = TARGET
        notes.append("目标语言 → English")
    if is_ja_only_model(getattr(cfg, "model_size", "")):
        old = cfg.model_size
        cfg.model_size = FALLBACK_MODEL
        notes.append(f"识别模型 {old} 为日语专用，已换 {FALLBACK_MODEL}")
    return notes


def verify(cfg, asr=None) -> dict:
    """核对三处是否真的到位；asr 传入时可同时核对实际加载档位。"""
    loaded = getattr(asr, "_loaded_size", None) or ""
    return {
        "source": getattr(cfg, "source_lang", "") == SOURCE,
        "target": getattr(cfg, "target_lang", "") == TARGET,
        "model": not is_ja_only_model(getattr(cfg, "model_size", "")),
        "loaded_model": loaded,
        "loaded_ok": bool(loaded) and not is_ja_only_model(loaded),
        "asr_language": getattr(asr, "language", ""),
    }


def status_text(cfg, asr=None) -> str:
    """面板状态区文案：逐项显示「配置值 + 是否到位 + 实际加载档」。"""
    st = verify(cfg, asr)
    loaded = st["loaded_model"] or "未加载"
    lines = [
        "音频语言：%s（%s）" % (SOURCE_LABELS.get(cfg.source_lang, cfg.source_lang),
                              "已切到中文" if st["source"] else "未切换"),
        "识别模型：%s（%s；实际加载 %s）" % (
            cfg.model_size,
            "通用档" if st["model"] else "日语专用，中文会失真",
            loaded),
        "目标语言：%s（%s）" % (
            TARGET_LABELS.get(cfg.target_lang, cfg.target_lang),
            "已切到英文" if st["target"] else "未切换"),
    ]
    if asr is not None and st["loaded_model"]:
        if not st["loaded_ok"]:
            lines.append("实际运行的仍是日语专用模型，开始翻译前会重新加载通用档。")
        elif st["asr_language"] != SOURCE:
            lines.append("实际解码语言是 %s，开始翻译前会重新对齐为中文。" % (
                SOURCE_LABELS.get(st["asr_language"], st["asr_language"] or "未知")))
    return "\n".join(lines)
