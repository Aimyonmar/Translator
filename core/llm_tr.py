# -*- coding: utf-8 -*-
"""LLM 质量通道：DeepSeek / 通义 DashScope（均为 OpenAI 兼容接口）。

设计取舍：
- **无 Key 即不存在**：云端 provider 无 Key 时 `available` 为 False，Translator 根本不会把
  "llm" 放进候选列表，现有链路的通道顺序、延迟、失败面完全不变（A/B 见改进记录第二十八节）。
- **不做重试**：实时字幕的预算是死的，重试等于把一次超时翻倍；失败直接回落到免费通道。
- **并发上限 2**：与 pipeline 的翻译线程数一致，超出的请求排队等锁，
  避免用户把开关打开后瞬间打满额度。
- **超时 6 秒**：比免费通道单次 4 秒宽（LLM 首包慢），但仍受 translate() 的总预算约束。
- **不写死任何 Key**：一律从 config.json（gitignored）读取，空串即视为未配置。
"""
from __future__ import annotations

import threading
from typing import Optional

import requests

# 不复用 translator.UA：两个模块互相 import 会绕成环，这里各留一份常量
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0")

# 两家都用 OpenAI 兼容端点，请求体一致，只有 host 与默认模型不同
PROVIDERS = {
    "deepseek": {"base": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "dashscope": {"base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "model": "qwen-plus"},
}

_SYS = (
    "你是资深日语同声传译，把 VTuber 直播里的日语口语翻成{target}。"
    "规则：只输出译文，不要解释、不要加引号；口语化、简短；保留语气词和情绪；"
    "人名/作品名用通行译法，没有通行译法就保留原文；拟声词意译；"
    "上下文只用于理解指代和省略，不要翻译或复述上下文。"
)


class LLMTranslator:
    TIMEOUT = 6.0
    MAX_CONCURRENT = 2

    def __init__(self, cfg=None, session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self._sem = threading.BoundedSemaphore(self.MAX_CONCURRENT)

    # ---------------------------------------------------------- 可用性
    @property
    def provider(self) -> str:
        return (getattr(self.cfg, "llm_provider", "") or "").strip().lower()

    @property
    def key(self) -> str:
        return (getattr(self.cfg, "llm_key", "") or "").strip()

    @property
    def available(self) -> bool:
        return bool(getattr(self.cfg, "enable_llm_quality", False)
                    and self.provider in PROVIDERS and self.key)

    def _model(self) -> str:
        m = (getattr(self.cfg, "llm_model", "") or "").strip()
        return m or PROVIDERS[self.provider]["model"]

    def _target(self) -> str:
        lang = getattr(self.cfg, "target_lang", "zh-Hans") or "zh-Hans"
        return {"zh-Hans": "简体中文", "zh-Hant": "繁體中文",
                "en": "English", "ko": "한국어"}.get(lang, "简体中文")

    # ---------------------------------------------------------- 调用
    def translate(self, text: str, context: str = "", src: str = "ja",
                  timeout: Optional[float] = None) -> Optional[str]:
        """翻译一句；不可用/失败返回 None，调用方负责回落。"""
        if not self.available or not text:
            return None
        base = PROVIDERS[self.provider]["base"].rstrip("/")
        sys_text = _SYS.format(target=self._target())
        if src == "en":
            sys_text = sys_text.replace("日语", "英语", 1)
        if context:
            sys_text += "\n\n" + context
        body = {
            "model": self._model(),
            "temperature": 0.2,
            "max_tokens": 400,
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": text}],
        }
        # 并发闸：拿不到就等（最多等一个超时周期），等不到直接放弃，不排队拖垮字幕
        tmo = timeout or self.TIMEOUT
        if not self._sem.acquire(timeout=tmo):
            return None
        try:
            r = self.session.post(
                base + "/chat/completions", timeout=tmo,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"},
                json=body,
            )
            r.raise_for_status()
            out = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return out or None
        except Exception:
            return None
        finally:
            self._sem.release()
