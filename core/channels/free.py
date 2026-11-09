# -*- coding: utf-8 -*-
"""六个免费通道的 HTTP 适配：腾讯 / 彩云 / Edge / Google / DeepL 网页 / MyMemory。

只负责「发一次请求并解析出译文」，通道选择、重试与降级都在 Translator 里。
"""
from __future__ import annotations

import html
import json
import time
import urllib.parse
import uuid
from typing import Optional

import requests


class EdgeAuth:
    """Edge 鉴权 token 的内存状态（约 8 分钟过期）。"""

    __slots__ = ("token", "ts")

    def __init__(self) -> None:
        self.token: Optional[str] = None
        self.ts = 0.0


def edge_auth(session: requests.Session, state: EdgeAuth, timeout: float = 3.0) -> Optional[str]:
    r = session.get("https://edge.microsoft.com/translate/auth", timeout=timeout)
    r.raise_for_status()
    state.token = r.text.strip()
    state.ts = time.time()
    return state.token


def edge(session: requests.Session, state: EdgeAuth, target: str,
         text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
    if not state.token or time.time() - state.ts > 480:
        edge_auth(session, state, timeout)
    tgt = "zh-Hant" if target == "zh-Hant" else target
    url = (f"https://api-edge.cognitive.microsofttranslator.com/translate"
           f"?api-version=3.0&from={src}&to={tgt}")
    for attempt in (0, 1):
        if attempt:
            state.token = None
            edge_auth(session, state)
        r = session.post(url, timeout=4,
                         headers={"Authorization": f"Bearer {state.token}",
                                  "Content-Type": "application/json"},
                         data=json.dumps([{"Text": text}], ensure_ascii=False).encode("utf-8"))
        if r.status_code == 401 and attempt == 0:
            continue
        r.raise_for_status()
        return r.json()[0]["translations"][0]["text"]
    return None


def google(session: requests.Session, target: str,
           text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
    tl = {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}.get(target, target)
    url = ("https://translate.googleapis.com/translate_a/single?client=gtx"
           f"&sl={src}&tl={tl}&dt=t&q={urllib.parse.quote(text)}")
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return "".join(p[0] for p in r.json()[0] if p and p[0])


def deepl(session: requests.Session, target: str,
          text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
    lang = "ZH" if target in ("zh-Hans", "zh-Hant") else target.upper()
    body = {"jsonrpc": "2.0", "method": "LMT_handle_texts",
            "params": {"texts": [{"text": text, "requestAlternatives": 0}],
                       "splitting": "newlines",
                       "lang": {"source_lang_user_selected": src.upper(), "target_lang": lang},
                       "commonJobParams": {"formality": None, "mode": "translate"},
                       "timestamp": int(time.time() * 1000)},
            "id": int(uuid.uuid4().int % 100000000) * 1000}
    r = session.post("https://www2.deepl.com/jsonrpc", json=body, timeout=timeout,
                     headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "deepl error"))
    return data["result"]["texts"][0]["text"]


def mymemory(session: requests.Session, target: str,
             text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
    tl = {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}.get(target, target)
    r = session.get("https://api.mymemory.translated.net/get", timeout=timeout,
                    params={"q": text, "langpair": f"{src}|{tl}"})
    r.raise_for_status()
    data = r.json()
    out = (data.get("responseData") or {}).get("translatedText") or ""
    return html.unescape(out).strip()


def tencent(session: requests.Session, no_proxy: dict, target: str,
            text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
    """腾讯翻译君网页接口：国内直连，实测 90~300ms，是目前最快的免费通道。"""
    tgt = {"zh-Hans": "zh", "zh-Hant": "zh-TW"}.get(target, target)
    r = session.post(
        "https://transmart.qq.com/api/imt", timeout=timeout,
        proxies=no_proxy,          # 国内直连，强制不走代理
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "header": {"fn": "auto_translation",
                       "client_key": "browser-chrome-124-0-Windows-10-0"},
            "source": {"text_list": [text], "lang": src},
            "target": {"lang": tgt},
        }, ensure_ascii=False).encode("utf-8"),
    )
    r.raise_for_status()
    data = r.json()
    if data.get("header", {}).get("ret_code") != "succ":
        raise RuntimeError(data.get("message") or "tencent error")
    out = (data.get("auto_translation") or [""])[0]
    return html.unescape(out).strip()


def caiyun(session: requests.Session, no_proxy: dict, cfg, target: str,
           text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
    """彩云小译：口语/俗语理解明显好于机翻，实测「めっちゃ眠いんだけど」能翻对。"""
    token = (getattr(cfg, "caiyun_token", "") or "").strip()
    if not token:
        raise RuntimeError("未配置彩云 token")
    tgt = {"zh-Hans": "zh", "zh-Hant": "zh-Hant", "en": "en"}.get(target, "zh")
    pair = f"{src}2{tgt}"
    r = session.post(
        "https://api.interpreter.caiyunai.com/v1/translator", timeout=timeout,
        proxies=no_proxy,          # 国内直连，强制不走代理
        headers={"Content-Type": "application/json", "X-Authorization": f"token {token}"},
        data=json.dumps({"source": [text], "trans_type": pair},
                        ensure_ascii=False).encode("utf-8"),
    )
    r.raise_for_status()
    data = r.json()
    if data.get("rc") not in (0, None):
        raise RuntimeError(f"caiyun rc={data.get('rc')}")
    return html.unescape((data.get("target") or [""])[0]).strip()
