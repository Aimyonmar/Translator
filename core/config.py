# -*- coding: utf-8 -*-
"""应用配置：读写 JSON，记住设备、模型、字幕样式等偏好。"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

# 主线程、翻译线程（Translator._remember）、设备探测线程都会调 save()。
# 不加锁时两个线程可能同时写同一文件，写出半截 JSON，下次启动配置直接回落默认值。
_SAVE_LOCK = threading.Lock()


def _clamp(value: Any, lo, hi, cast):
    """把配置值夹到合法区间；类型不对时返回 None（由调用方决定是否保留原值）。"""
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return None
    return min(max(v, lo), hi)


@dataclass
class AppConfig:
    # 音频
    device: Optional[Dict[str, Any]] = None
    # 识别
    model_size: str = "small"
    asr_device: str = "auto"
    compute_type: str = "auto"
    source_lang: str = "ja"
    # 解码宽度：1=最快（贪心），2=快且稳（默认），3~5=更准但明显更慢。
    # large-v* 模型默认自动降到 1（见 beam_auto）；其余模型保持 2。CPU 上别超过 2（否则极慢）。
    beam_size: int = 2
    # 解码宽度自动适配：True 时 large-v* 模型默认用最快档 beam=1 保实时性；
    # 用户在面板手动拖过解码宽度后会自动变 False（尊重手动选择，重启也不再自动覆盖）。
    beam_auto: bool = True
    # 纠错替换表：识别完成后把听错的人名/梗改回正确写法（零回声风险）。
    # 多行「听错的写法=正确写法」，例：うさだぺこら=兎田ぺこら
    word_map: str = ""
    # 专有名词「音近修正表」：一行一个「正确写法」或「正确写法=假名读音」（含汉字的名字
    # 要给读音，例：兎田ぺこら=うさだぺこら）。与 word_map 的区别——word_map 要写死
    # 「这一次听错的字样」，本表只写正确名字，程序按读音自动覆盖 ひらがな/カタカナ/
    # 長音/促音/近音行 等各种听错变体，维护量只随人名数增长。
    name_roster: str = ""
    # 音近修正总开关（关掉则只保留 word_map 精确替换）
    enable_namefix: bool = True
    # 把词表作为 Whisper 热词下发给解码器，从识别源头提高专有名词命中率。
    # 2026-10-13 实测（title_asr_test.py，6 个游戏标题 × 同一音频）：
    #   裸识别 CER 7.1% → 热词（展示写法）4.7%；「黒髪話、悟空」→「黒神話、悟空」。
    #   注意热词**必须用汉字/片假名原样**，用假名读音反而更差（CER 26.1%，还会吐空串）。
    #   误伤扫描：6 句普通直播句 × 6 个标题 → 凭空插入 0 处，CER 3.1% → 1.7%。故默认开启。
    enable_hotwords: bool = True
    # 是否把内置的游戏/番剧标题（namefix.BUILTIN_TITLES）也算进热词表
    builtin_titles: bool = True
    # 标题术语表：送翻译前把日语标题换成中文名（エルデンリング→艾尔登法环）。
    # 机翻不认识游戏名：实测「エルデンリングのDLC」→「埃尔金戒指DLC」、
    # 「モンハンワイルズ」→「蒙汉威尔斯」；换成中文名后分别变成
    # 「我买了艾尔登法环的DLC」与「怪物猎人：荒野的发售日好像决定了」。
    enable_title_glossary: bool = True
    # 语境辅助（设计见 docs/语境翻译设计.md）
    # 翻译层上下文总开关：最近几条原文/译文只注入自定义 LLM 通道，免费通道保持单句
    enable_context: bool = True
    context_window: int = 3        # 带几条上下文，0~8；0 等价关闭
    # 识别层同音词纠正（変身→返信）：按上下文线索判定，每次改动都会上报状态栏
    enable_homophone_fix: bool = True
    # 免费通道（腾讯/彩云/Edge/Google/DeepL…）也带语境：把上一句原文作为极短前缀
    # 一起送翻，再按分隔切回当前句。试验项，默认关闭（关闭时行为与原来完全一致）
    enable_free_context: bool = False
    # 语境术语表：多行「原词=译法」，当前句出现时按此译法（仅 LLM 通道）
    context_glossary: str = ""
    # 翻译侧术语表：多行「原文=译法」，送翻前按整词精确替换，译文里再兜底修正已知坏译法。
    # 与 enable_title_glossary 互补：内置标题表先执行，本表只补用户/诊断里新增的专名。
    # 默认关闭。
    enable_translate_glossary: bool = False
    translate_glossary: str = ""
    # LLM 质量通道（可选）：provider = deepseek | dashscope（均为 OpenAI 兼容接口）。
    # 没填 Key 或关掉开关时，该通道不进入候选列表，现有链路完全不受影响。默认关闭。
    enable_llm_quality: bool = False
    llm_provider: str = ""
    llm_key: str = ""
    llm_model: str = ""
    # 翻译
    channel: str = "auto"
    target_lang: str = "zh-Hans"
    # 可选：填了 Key 会优先走自定义 API（质量最好）；不填则只用免费通道
    custom_type: str = "openai"        # openai | deepl
    custom_key: str = ""
    custom_base: str = "https://api.deepseek.com/v1"
    custom_model: str = "deepseek-chat"
    deepl_free: bool = True
    # 彩云小译 token：口语/俗语翻得最准。这是官方公开的体验 token，
    # 限额用尽或失效时可在面板里换成自己申请的（彩云开放平台免费领）。
    caiyun_token: str = "3975l6lr5pcbvidl6jl2"
    # 日式英语 / 外来语处理：翻译前把片假名外来语（アプデ=更新、ガチ=认真…）
    # 替换成人工校准的中文义。这比「加英文翻译」更准——和製英語意思常与英语原词不同
    # （マンション=公寓、パンク=爆胎）。若翻译异常可设 False 关闭。
    enable_loanwords: bool = True
    # 译文口语化润色（core/zh_polish.py，确定性规则、0ms）：修免费通道的固定生硬模式
    # （伙计们→大家、您→你、想尝试→想试试、半角标点全角化、去句尾书面句号）。
    enable_zh_polish: bool = True
    # 规则化译后处理（core/zh_rules.py）：终助词语气、惯用句、数词单位、授受方向、
    # 省略主语、口语缩略。默认开启；命中计数见 zh_rules.take_stats()。
    # 误改回收：把对应规则名从 zh_rules 里删掉（或加 - 前缀）即可单独关闭。
    enable_zh_rules: bool = True
    zh_rules: str = "tone,idiom,number,direction,subject,contraction"
    # 英文句路由：主播真说英文（拉丁字母占多数）时自动改走 EN→ZH。可设 False 关闭。
    enable_en_route: bool = True
    # 主备式延迟补发（core/translator.py）：主通道超过 hedge_backup_ms 还没返回，
    # 就并行补发第二条线路，取先到的可用译文。实测 P50/P90 不变、P99 少约 0.3 s，
    # 代价是约 6% 的句子多发一次请求（见改进记录第二十节）。设 False 回到纯串行。
    enable_hedge: bool = True
    hedge_backup_ms: int = 400
    # 上次实际用通的翻译通道，下次启动直接复用，省去挨个试错的等待
    last_channel: str = ""
    # 音频转文字：上次选文件的目录，下次打开对话框直接落在这里
    transcribe_dir: str = ""
    # 文件转写：是否顺带翻译（复用实时那套通道/主备/缓存），默认关，避免无谓的网络耗时
    trs_translate: bool = False
    # 文件转写：按最大字数重新断句，0 = 不重新断句
    trs_max_chars: int = 0
    # 实时记录导出目录：留空时导出到「文档」，成功后自动记住
    export_dir: str = ""
    # 音频预处理：轻量谱减法降噪，压低稳态背景噪声（风扇/电流声/低频隆隆）。
    # 对 BGM 这类非稳态音乐帮助有限（那需要人声分离模型），但能改善整体信噪比。
    enable_denoise: bool = True
    denoise_strength: float = 0.5      # 0~1，越大消得越狠；0.5 较温和
    # 人声提取（中置声道提取，core/vocals.py）：回环是立体声时，把「左右不同相」的
    # 宽声场 BGM 压低、保留正中的说话声，再送 VAD/识别。对本来就正中的内容逐样本无损。
    # 实测：完全宽的成分可压 >20x；部分去相关的 BGM 约 3dB（原理上限——中置混音的 BGM 压不掉）。
    vocal_extract: bool = True
    vocal_strength: float = 0.8        # 0~1：越大压得越狠，留下的宽声场成分越少
    # GPU 加速是否可用：试过一次失败就记住，避免每次启动都白等二十秒
    gpu_ok: bool = True
    # 可选：HTTP 代理，国内网络下用来解锁 Google / Edge 通道
    proxy: str = ""
    auto_proxy: bool = True
    # VAD
    sensitivity: float = 0.70      # 越大越灵敏（阈值越低）；直播音量普遍偏小，默认给高一点
    min_speech_ms: int = 250
    min_silence_ms: int = 400      # 停顿 0.4 秒断句：实测 0.6s 平均多等 0.36s，0.3s 会把一句话剁成 1.5 倍碎片
    max_speech_ms: int = 4000      # 最长 4 秒强制切一刀（原来是 8 秒，实测延迟会到 5-6 秒）
    # 碎片段合并：VAD 偶尔剁出 <0.45s 的碎段，短暂等下一段拼上再识别
    # （碎段字幕没营养，且短段 ASR 固定开销占比极高）。代价：孤立碎段多等 ≤0.25s。
    merge_fragments: bool = True
    # 字幕
    show_source: bool = True
    font_size: int = 26
    source_font_size: int = 16
    opacity: float = 0.72
    bg_opacity: int = 150
    click_through: bool = False
    max_lines: int = 3
    auto_clear_sec: float = 0.0
    window_width: int = 0
    geometry: Optional[Dict[str, int]] = None
    screen_index: int = 0
    show_panel: bool = True

    def save(self) -> None:
        """原子写：先写临时文件再替换，避免中途失败/并发写留下半截 JSON。"""
        data = asdict(self)
        tmp = CONFIG_PATH + ".tmp"
        with _SAVE_LOCK:
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, CONFIG_PATH)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

    # 数值字段的合法区间：(下限, 上限, 类型)。config.json 是用户可手改的边界，
    # 越界/类型错误在这里夹回安全范围，别让 int(None) 之类的问题拖到界面构造或 VAD 里才炸。
    _NUM_RANGES = {
        "sensitivity": (0.0, 1.0, float),
        "denoise_strength": (0.0, 1.0, float),
        "vocal_strength": (0.0, 1.0, float),
        "min_speech_ms": (50, 5000, int),
        "min_silence_ms": (100, 3000, int),
        "max_speech_ms": (500, 60000, int),
        "hedge_backup_ms": (100, 2000, int),
        "beam_size": (1, 5, int),
        "context_window": (0, 8, int),
        "font_size": (10, 72, int),
        "source_font_size": (8, 48, int),
        "opacity": (0.1, 1.0, float),
        "bg_opacity": (0, 255, int),
        "max_lines": (1, 8, int),
        "auto_clear_sec": (0.0, 600.0, float),
        "window_width": (0, 10000, int),
        "screen_index": (0, 16, int),
        "trs_max_chars": (0, 200, int),
    }

    _STR_FIELDS = (
        "model_size", "asr_device", "compute_type", "source_lang", "channel", "target_lang",
        "custom_type", "custom_key", "custom_base", "custom_model", "caiyun_token",
        "last_channel", "word_map", "name_roster", "proxy", "context_glossary",
        "transcribe_dir", "export_dir", "translate_glossary",
        "llm_provider", "llm_key", "llm_model", "zh_rules",
    )

    def _sanitize(self) -> None:
        """读取后校验：类型兜底 + 区间夹取，坏值回落到默认语义。"""
        for name, (lo, hi, cast) in self._NUM_RANGES.items():
            v = _clamp(getattr(self, name, None), lo, hi, cast)
            if v is None:
                # 类型完全不对（如 "abc"）时回落到字段默认值，而不是把坏值原样带进运行时
                v = _clamp(getattr(type(self), name, None), lo, hi, cast)
            if v is not None:
                setattr(self, name, v)
        for name in self._STR_FIELDS:
            v = getattr(self, name, "")
            if not isinstance(v, str):
                setattr(self, name, "")
        if not isinstance(self.device, dict):
            self.device = None
        if not isinstance(self.geometry, dict):
            self.geometry = None
        if not self.target_lang:
            self.target_lang = "zh-Hans"
        try:
            from .asr import MODEL_SIZES
        except Exception:
            MODEL_SIZES = ()
        if MODEL_SIZES and self.model_size not in MODEL_SIZES:
            self.model_size = "small"

    @classmethod
    def load(cls) -> "AppConfig":
        cfg = cls()
        if not os.path.exists(CONFIG_PATH):
            return cfg
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # 只认 dataclass 字段：以前用 hasattr，JSON 里出现 "save" 之类的键
                # 会把方法覆盖成字符串，之后 cfg.save() 直接炸
                fields = type(cfg).__dataclass_fields__
                for k, v in data.items():
                    if k in fields:
                        setattr(cfg, k, v)
        except Exception:
            # 坏文件先留证：否则下一次 save() 用默认值把它覆盖掉，问题无从复查
            try:
                os.replace(CONFIG_PATH, CONFIG_PATH + ".bad")
            except Exception:
                pass
        cfg._sanitize()
        return cfg
