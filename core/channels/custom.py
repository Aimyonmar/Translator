# -*- coding: utf-8 -*-
"""自定义 API 通道：DeepL 免费版 Key，或任意 OpenAI 兼容接口（DeepSeek、通义、豆包…）。"""
from __future__ import annotations

import json
from typing import Optional

import requests

from .constants import LANG_NAME_BY_CODE

_SYS_PROMPT = (
    "你是资深日语同声传译。把用户输入的日语（VTuber 直播/视频口语）翻译成{target}。"
    "要求：只输出译文，不要解释、不要加引号、不要保留原文；口语化、自然、简短；"
    "保留语气词和情绪；人名/作品名/梗按常见译法或保留原文；遇到拟声词意译。"
)


def custom(session: requests.Session, cfg, target: str,
           text: str, src: str = "ja", context: str = "") -> Optional[str]:
    kind = getattr(cfg, "custom_type", "openai")
    key = (getattr(cfg, "custom_key", "") or "").strip()
    if not key:
        return None
    if kind == "deepl":
        host = "https://api-free.deepl.com" if getattr(cfg, "deepl_free", True) else "https://api.deepl.com"
        lang = "ZH" if target in ("zh-Hans", "zh-Hant") else target.upper()
        r = session.post(f"{host}/v2/translate", timeout=12,
                         data={"auth_key": key, "text": text,
                               "source_lang": src.upper(), "target_lang": lang})
        r.raise_for_status()
        return r.json()["translations"][0]["text"]
    base = (getattr(cfg, "custom_base", "") or "https://api.deepseek.com/v1").rstrip("/")
    model = getattr(cfg, "custom_model", "") or "deepseek-chat"
    tname = LANG_NAME_BY_CODE.get(target, target)
    src_name = "日语" if src in ("ja", "ja-JP", "japanese") else "英语"
    sys_text = _SYS_PROMPT.format(target=tname).replace("日语", src_name, 1)
    if context:
        # 上下文只作为理解材料：先声明「不要翻译/复述」，再要求只输出当前句，
        # 防止模型把前几轮一起翻出来（字幕里出现两句话）
        sys_text += ("\n\n" + context
                     + "\n只翻译用户消息里的当前句，不要复述或翻译上面的对话，"
                       "不要补充原文没有的信息。")
    r = session.post(
        base + "/chat/completions", timeout=25,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps({"model": model, "temperature": 0.2, "max_tokens": 400,
                         "messages": [{"role": "system", "content": sys_text},
                                      {"role": "user", "content": text}]}, ensure_ascii=False).encode("utf-8"),
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()
