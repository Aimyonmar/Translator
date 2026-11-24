# -*- coding: utf-8 -*-
"""输入语言三档预设：English / 中文 / 日本語。

每档绑定「源语言 + 识别模型 + 目标语言」，切换时一次到位，避免只改了源语言、
模型还是日语专用档，或目标语言没跟着变的半套状态。

- English：Whisper 通用档 small，译文简体中文
- 中文：Whisper 通用档 small，译文 English（kotoba 是日语专用，绝不使用）
- 日本語：kotoba-whisper-v2.0，译文简体中文
"""
from __future__ import annotations

MODES = ("en", "zh", "ja")
MODE_LABELS = {"en": "English", "zh": "中文", "ja": "日本語"}
SOURCE_LABELS = {"ja": "日本語", "zh": "中文", "en": "英语"}
TARGET_LABELS = {"zh-Hans": "简体中文", "zh-Hant": "繁體中文", "en": "English", "ko": "한국어"}

PRESETS = {
    "en": {"source": "en", "target": "zh-Hans", "model": "small"},
    "zh": {"source": "zh", "target": "en", "model": "small"},
    "ja": {"source": "ja", "target": "zh-Hans", "model": "kotoba-whisper-v2.0"},
}


def is_ja_only_model(size: str) -> bool:
    return "kotoba" in (size or "").lower()


def detect(cfg) -> str:
    """按 cfg.source_lang 反查当前档位；未知值按日语处理（老配置兼容）。"""
    src = str(getattr(cfg, "source_lang", "ja") or "ja").lower()
    if src.startswith("en"):
        return "en"
    if src.startswith("zh"):
        return "zh"
    return "ja"


def apply(cfg, mode: str) -> list:
    """切到指定档位，返回需要告知用户的变更说明。"""
    preset = PRESETS.get(mode)
    if preset is None:
        raise ValueError(f"未知输入语言档位：{mode}")
    notes = []
    if getattr(cfg, "source_lang", "") != preset["source"]:
        cfg.source_lang = preset["source"]
        notes.append("输入语言 → %s" % SOURCE_LABELS[preset["source"]])
    if getattr(cfg, "target_lang", "") != preset["target"]:
        cfg.target_lang = preset["target"]
        notes.append("译文 → %s" % TARGET_LABELS[preset["target"]])
    if getattr(cfg, "model_size", "") != preset["model"]:
        old = cfg.model_size
        cfg.model_size = preset["model"]
        notes.append("识别模型 %s → %s" % (old, preset["model"]))
    return notes


def route_text(cfg) -> str:
    """顶栏常驻路由：输入语言 → 译文 · 模型。"""
    src = SOURCE_LABELS.get(getattr(cfg, "source_lang", ""), cfg.source_lang)
    tgt = TARGET_LABELS.get(getattr(cfg, "target_lang", ""), cfg.target_lang)
    return f"{src} → {tgt} · {cfg.model_size}"


def verify(cfg, asr=None) -> dict:
    """核对当前档位的三处是否到位；传 asr 时附带实际加载档。"""
    mode = detect(cfg)
    preset = PRESETS[mode]
    loaded = getattr(asr, "_loaded_size", None) or ""
    return {
        "mode": mode,
        "source": getattr(cfg, "source_lang", "") == preset["source"],
        "target": getattr(cfg, "target_lang", "") == preset["target"],
        "model": getattr(cfg, "model_size", "") == preset["model"],
        "loaded_model": loaded,
        "loaded_ok": bool(loaded) and (mode == "ja" or not is_ja_only_model(loaded)),
        "asr_language": getattr(asr, "language", ""),
    }


def status_text(cfg, asr=None) -> str:
    """逐项显示当前档位的配置值、是否到位、实际加载档。"""
    st = verify(cfg, asr)
    loaded = st["loaded_model"] or "未加载"
    lines = [
        "输入语言：%s（%s）" % (
            MODE_LABELS[st["mode"]],
            "已到位" if st["source"] else "未到位"),
        "识别模型：%s（%s；实际加载 %s）" % (
            cfg.model_size,
            "本档指定" if st["model"] else "与本档不一致",
            loaded),
        "译文：%s（%s）" % (
            TARGET_LABELS.get(cfg.target_lang, cfg.target_lang),
            "已到位" if st["target"] else "未到位"),
    ]
    if asr is not None and st["loaded_model"]:
        if not st["loaded_ok"]:
            lines.append("实际运行的仍是日语专用模型，开始翻译前会重新加载通用档。")
        elif st["asr_language"] != cfg.source_lang:
            lines.append("实际解码语言是 %s，开始翻译前会重新对齐。" % (
                SOURCE_LABELS.get(st["asr_language"], st["asr_language"] or "未知")))
    return "\n".join(lines)
