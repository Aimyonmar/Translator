# -*- coding: utf-8 -*-
"""通道静态定义：通道清单、默认顺序、语言映射、UA 与本地代理端口。

这些表被 translator / panel / feedback 共同引用，集中在这里避免多处硬编码。
"""

CHANNELS = ["auto", "custom", "tencent", "caiyun", "mymemory", "edge", "google", "deepl"]
CHANNEL_NAMES = {"auto": "自动", "custom": "自定义 API", "tencent": "腾讯翻译君",
                 "caiyun": "彩云小译", "edge": "Edge·必应",
                 "google": "Google", "deepl": "DeepL 网页", "mymemory": "MyMemory",
                 "llm": "LLM 质量通道"}

# 国内实测（2026-10）：腾讯 217ms / 彩云 715ms / MyMemory 831ms，
# 且口语准确率 彩云 > 腾讯 > MyMemory。默认把最快、也够用的腾讯排前面。
FREE_ORDER = ["tencent", "caiyun", "mymemory", "edge", "google", "deepl"]
# 句子偏口语时优先彩云：机翻最容易把「めっちゃ眠いんだけど」翻成「今天，我睡得很好。」
COLLOQUIAL_ORDER = ["caiyun", "tencent", "mymemory", "edge", "google", "deepl"]

LANGS = {
    "简体中文": "zh-Hans",
    "繁體中文": "zh-Hant",
    "English": "en",
    "한국어": "ko",
}
LANG_NAME_BY_CODE = {v: k for k, v in LANGS.items()}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0")

# 常见本地代理端口（Clash / V2Ray / Shadowsocks / Mihomo …）
PROXY_PORTS = [7890, 7897, 7891, 10809, 10808, 1080, 2080, 8080, 8888, 20171, 33210, 4780]
