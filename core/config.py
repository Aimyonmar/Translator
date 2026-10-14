# -*- coding: utf-8 -*-
"""应用配置：读写 JSON，记住设备、模型、字幕样式等偏好。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")


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
    # 英文句路由：主播真说英文（拉丁字母占多数）时自动改走 EN→ZH。可设 False 关闭。
    enable_en_route: bool = True
    # 上次实际用通的翻译通道，下次启动直接复用，省去挨个试错的等待
    last_channel: str = ""
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
    min_silence_ms: int = 300      # 停顿 0.3 秒就断句——再长字幕就跟不上说话了
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
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def load(cls) -> "AppConfig":
        cfg = cls()
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        except Exception:
            pass
        return cfg
