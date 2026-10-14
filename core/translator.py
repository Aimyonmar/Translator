# -*- coding: utf-8 -*-
"""翻译模块：免费在线通道 + 可选自定义 API + 代理支持。

免费通道（无需 Key）：Edge·必应 / Google / DeepL网页 / MyMemory，自动探测可用的那个。
自定义 API（可选，质量最好）：DeepL 免费版 Key，或任意 OpenAI 兼容接口（DeepSeek、通义、豆包、本地 Ollama…）。
代理：国内网络可填 HTTP 代理（如 http://127.0.0.1:7890）解锁 Google / Edge，程序也会自动探测本机代理端口。
"""
from __future__ import annotations

import html
import json
import socket
import threading
import time
import urllib.parse
import uuid
from collections import OrderedDict
from typing import Callable, List, Optional

import requests

from .phrases import is_colloquial, lookup as phrase_lookup
from .loanwords import apply_loanwords
from .ja_norm import prepare as ja_prepare
from .zh_polish import zh_polish

CHANNELS = ["auto", "custom", "tencent", "caiyun", "mymemory", "edge", "google", "deepl"]
CHANNEL_NAMES = {"auto": "自动", "custom": "自定义 API", "tencent": "腾讯翻译君",
                 "caiyun": "彩云小译", "edge": "Edge·必应",
                 "google": "Google", "deepl": "DeepL 网页", "mymemory": "MyMemory"}

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

_SYS_PROMPT = (
    "你是资深日语同声传译。把用户输入的日语（VTuber 直播/视频口语）翻译成{target}。"
    "要求：只输出译文，不要解释、不要加引号、不要保留原文；口语化、自然、简短；"
    "保留语气词和情绪；人名/作品名/梗按常见译法或保留原文；遇到拟声词意译。"
)


class Translator:
    # 免费通道的翻译总预算（秒）。字幕是实时的，宁可这句只显示原文，
    # 也不能让画面卡在那里等网络。
    BUDGET = 5.0

    # 健康检查/探测用的短句：短、且译文确定，便于判断通道是不是真在工作
    PROBE_TEXT = "こんにちは"

    def __init__(self, cfg=None, on_log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.log_cb = on_log
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.lock = threading.Lock()
        self._edge_token: Optional[str] = None
        self._edge_token_ts = 0.0
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        # 上次用通的通道：冷启动直接复用，避免每次都从头把不通的通道挨个超时一遍
        # （edge/google/deepl 在国内常常要各等 10 秒才失败，首次出字幕会拖到 20-30 秒后）
        self._active: Optional[str] = getattr(cfg, "last_channel", "") or None
        self._dead: set = set()      # 本次会话内已确认不通的通道，不再重试
        self.proxy: str = getattr(cfg, "proxy", "") or ""
        # 腾讯/彩云是国内直连通道，本来就不需要代理；而本机代理（127.0.0.1:7897）经常不稳，
        # 一旦让它们走代理，译文请求会集体超时被放弃 —— 字幕就只剩原文、译文消失。
        # requests 的 merge_setting 会把值为 None 的键从 session.proxies 里删掉，
        # 所以传它等于「本次请求不走代理」。
        self._no_proxy = {"http": None, "https": None}

    def _say(self, msg: str) -> None:
        if self.log_cb:
            try:
                self.log_cb(msg)
            except Exception:
                pass

    def _remember(self, ch: str) -> None:
        """记住这次用通的通道，下次开机直接走它。"""
        try:
            if self.cfg is not None:
                self.cfg.last_channel = ch
                self.cfg.save()
        except Exception:
            pass

    # ---------------------------------------------------------- 配置
    @property
    def channel(self) -> str:
        return getattr(self.cfg, "channel", "auto") or "auto"

    @property
    def source(self) -> str:
        return getattr(self.cfg, "source_lang", "ja") or "ja"

    @property
    def target(self) -> str:
        return getattr(self.cfg, "target_lang", "zh-Hans") or "zh-Hans"

    def apply_proxy(self, proxy: Optional[str] = None) -> None:
        self.proxy = proxy if proxy is not None else (getattr(self.cfg, "proxy", "") or "")
        if self.proxy:
            self.session.proxies.update({"http": self.proxy, "https": self.proxy})
        else:
            self.session.proxies.clear()

    def _custom_ready(self) -> bool:
        c = self.cfg
        return bool(getattr(c, "custom_key", "") and getattr(c, "custom_type", "openai"))

    # ---------------------------------------------------------- 通道
    def _tr_custom(self, text: str, src: str = "ja") -> Optional[str]:
        c = self.cfg
        kind = getattr(c, "custom_type", "openai")
        key = (getattr(c, "custom_key", "") or "").strip()
        if not key:
            return None
        if kind == "deepl":
            host = "https://api-free.deepl.com" if getattr(c, "deepl_free", True) else "https://api.deepl.com"
            lang = "ZH" if self.target in ("zh-Hans", "zh-Hant") else self.target.upper()
            r = self.session.post(f"{host}/v2/translate", timeout=12,
                                  data={"auth_key": key, "text": text,
                                        "source_lang": src.upper(), "target_lang": lang})
            r.raise_for_status()
            return r.json()["translations"][0]["text"]
        base = (getattr(c, "custom_base", "") or "https://api.deepseek.com/v1").rstrip("/")
        model = getattr(c, "custom_model", "") or "deepseek-chat"
        tname = LANG_NAME_BY_CODE.get(self.target, self.target)
        src_name = "日语" if src in ("ja", "ja-JP", "japanese") else "英语"
        r = self.session.post(
            base + "/chat/completions", timeout=25,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps({"model": model, "temperature": 0.2, "max_tokens": 400,
                             "messages": [{"role": "system",
                                           "content": _SYS_PROMPT.format(target=tname).replace("日语", src_name, 1)},
                                          {"role": "user", "content": text}]}, ensure_ascii=False).encode("utf-8"),
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

    def _edge_auth(self, timeout: float = 3.0) -> Optional[str]:
        r = self.session.get("https://edge.microsoft.com/translate/auth", timeout=timeout)
        r.raise_for_status()
        self._edge_token = r.text.strip()
        self._edge_token_ts = time.time()
        return self._edge_token

    def _tr_edge(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        if not self._edge_token or time.time() - self._edge_token_ts > 480:
            self._edge_auth(timeout)
        tgt = "zh-Hant" if self.target == "zh-Hant" else self.target
        url = (f"https://api-edge.cognitive.microsofttranslator.com/translate"
               f"?api-version=3.0&from={src}&to={tgt}")
        for attempt in (0, 1):
            if attempt:
                self._edge_token = None
                self._edge_auth()
            r = self.session.post(url, timeout=4,
                                  headers={"Authorization": f"Bearer {self._edge_token}",
                                           "Content-Type": "application/json"},
                                  data=json.dumps([{"Text": text}], ensure_ascii=False).encode("utf-8"))
            if r.status_code == 401 and attempt == 0:
                continue
            r.raise_for_status()
            return r.json()[0]["translations"][0]["text"]
        return None

    def _tr_google(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        tl = {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}.get(self.target, self.target)
        url = ("https://translate.googleapis.com/translate_a/single?client=gtx"
               f"&sl={src}&tl={tl}&dt=t&q={urllib.parse.quote(text)}")
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        return "".join(p[0] for p in r.json()[0] if p and p[0])

    def _tr_deepl(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        lang = "ZH" if self.target in ("zh-Hans", "zh-Hant") else self.target.upper()
        body = {"jsonrpc": "2.0", "method": "LMT_handle_texts",
                "params": {"texts": [{"text": text, "requestAlternatives": 0}],
                           "splitting": "newlines",
                           "lang": {"source_lang_user_selected": src.upper(), "target_lang": lang},
                           "commonJobParams": {"formality": None, "mode": "translate"},
                           "timestamp": int(time.time() * 1000)},
                "id": int(uuid.uuid4().int % 100000000) * 1000}
        r = self.session.post("https://www2.deepl.com/jsonrpc", json=body, timeout=timeout,
                              headers={"Content-Type": "application/json"})
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message", "deepl error"))
        return data["result"]["texts"][0]["text"]

    def _tr_mymemory(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        tl = {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}.get(self.target, self.target)
        r = self.session.get("https://api.mymemory.translated.net/get", timeout=timeout,
                             params={"q": text, "langpair": f"{src}|{tl}"})
        r.raise_for_status()
        data = r.json()
        out = (data.get("responseData") or {}).get("translatedText") or ""
        return html.unescape(out).strip()

    def _tr_tencent(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        """腾讯翻译君网页接口：国内直连，实测 90~300ms，是目前最快的免费通道。"""
        tgt = {"zh-Hans": "zh", "zh-Hant": "zh-TW"}.get(self.target, self.target)
        r = self.session.post(
            "https://transmart.qq.com/api/imt", timeout=timeout,
            proxies=self._no_proxy,          # 国内直连，强制不走代理
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

    def _tr_caiyun(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        """彩云小译：口语/俗语理解明显好于机翻，实测「めっちゃ眠いんだけど」能翻对。"""
        token = (getattr(self.cfg, "caiyun_token", "") or "").strip()
        if not token:
            raise RuntimeError("未配置彩云 token")
        tgt = {"zh-Hans": "zh", "zh-Hant": "zh-Hant", "en": "en"}.get(self.target, "zh")
        pair = f"{src}2{tgt}"
        r = self.session.post(
            "https://api.interpreter.caiyunai.com/v1/translator", timeout=timeout,
            proxies=self._no_proxy,          # 国内直连，强制不走代理
            headers={"Content-Type": "application/json", "X-Authorization": f"token {token}"},
            data=json.dumps({"source": [text], "trans_type": pair},
                            ensure_ascii=False).encode("utf-8"),
        )
        r.raise_for_status()
        data = r.json()
        if data.get("rc") not in (0, None):
            raise RuntimeError(f"caiyun rc={data.get('rc')}")
        return html.unescape((data.get("target") or [""])[0]).strip()

    def _dispatch(self, ch: str, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        fn = {"custom": self._tr_custom, "tencent": self._tr_tencent, "caiyun": self._tr_caiyun,
              "edge": self._tr_edge, "google": self._tr_google,
              "deepl": self._tr_deepl, "mymemory": self._tr_mymemory}.get(ch)
        if fn is None:
            return None
        if ch == "custom":
            return fn(text, src)          # 用户自己配的通道，愿意多等，不受实时预算限制
        return fn(text, src, timeout)

    def _order(self, text: str = "") -> List[str]:
        # 口语化的句子优先走彩云（理解最准），普通句优先走腾讯（最快）。
        # 这里不把「上次用通的通道」提到最前：免费通道的优劣是固定的，
        # 上次记住的可能只是当时唯一能通的慢通道（比如 MyMemory 831ms），
        # 让它一直排在最前，用户就永远用不上快 4 倍的腾讯。
        base = list(COLLOQUIAL_ORDER if (text and is_colloquial(text)) else FREE_ORDER)
        free = [c for c in base if c not in self._dead] or list(base)
        if self._custom_ready():
            return ["custom"] + free
        ch = self.channel
        if ch in CHANNELS and ch != "auto":
            return [ch] + [c for c in free if c != ch]
        return free

    # ---------------------------------------------------------- 对外
    def translate(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None
        # 只在这一小段里持锁：网络请求绝不能放进锁里——
        # 之前整段都锁着，2 路并发翻译实际退化成串行排队，实测单句要等 4.7 秒。
        # 先查本地口语表：命中就是 0ms，且译文是人工校准过的，比任何免费机翻都准
        with self.lock:
            if text in self._cache:
                return self._cache[text]
            hit = phrase_lookup(text)
            if hit:
                self._cache[text] = hit
                self._phrase_hits = getattr(self, "_phrase_hits", 0) + 1
                if len(self._cache) > 800:
                    self._cache.popitem(last=False)
                return hit
            order = self._order(text)
            custom = self._custom_ready()
            # 送翻前的两级清洗（都在本地、0ms）：
            # 1) ja_norm：展开口语缩约（「置いてってます」→「置いていきました」）+
            #    折叠 ASR 重复子句（「相当してください、相当ください」→ 只留一句）；
            # 2) loanwords：片假名外来语换成人工校准的中文义（アプデ=更新…）。
            # 两者都对着「上屏的原文」做，缓存 key 仍用原始 text，译文不串味。
            pro = ja_prepare(text)
            if getattr(self.cfg, "enable_loanwords", True):
                pro = apply_loanwords(pro)
            # 整句基本是拉丁字母（主播真在说英文，而非片假名外来语）：改走 EN→ZH 路由
            en_route = getattr(self.cfg, "enable_en_route", True) and self._looks_english(text)
            src = "en" if en_route else self.source

        last_err = None
        # 实时字幕有总预算：免费通道挨个试过去最坏能拖到二三十秒（实测 21 秒），
        # 字幕就彻底跟不上说话了。超预算立刻放弃译文、只显示原文。
        deadline = time.time() + (25.0 if custom else self.BUDGET)
        switched = False
        for ch in order:
            left = deadline - time.time()
            if left <= 0.2:
                break
            try:
                # 只剩多少时间就只给多少超时：只在循环开头检查预算是拦不住的，
                # 单次请求（实测 mymemory 4s + edge 鉴权 3s = 8 秒）会把预算整个冲破
                out = self._fire(ch, pro, src, timeout=min(4.0, left))
                if out and out.strip():
                    out = html.unescape(out.strip())
                    # 免费通道（尤其 MyMemory 匿名额度）对很短的句子会返回一大串拼凑乱码，
                    # 比如「こんにちは」→ "penian ferein OU潘托斯所有安卓sofou"。
                    # 这里绝不能 continue 去试其它通道：edge/google/deepl 都要等满超时，
                    # 实测会把这句的翻译拖到 12 秒，字幕就彻底跟不上说话了。
                    # 直接放弃译文，让界面只显示原文。
                    if self._is_junk(text, out):
                        return None
                    # 译文口语化润色（确定性规则、0ms）：免费通道的固定生硬模式
                    # （伙计们→大家、您→你、想尝试→想试试、半角标点、句尾书面句号）。
                    # 规则只动词级与标点，绝不动句法——润色失败顶多不自然，不引入新错误。
                    if getattr(self.cfg, "enable_zh_polish", True):
                        out = zh_polish(out)
                    with self.lock:
                        self._cache[text] = out
                        if len(self._cache) > 800:
                            self._cache.popitem(last=False)
                        if self._active != ch:
                            self._active = ch
                            self._remember(ch)
                            switched = True
                    if switched:
                        self._say(f"翻译通道：{CHANNEL_NAMES.get(ch, ch)}")
                    return out
            except Exception as exc:
                last_err = exc
                with self.lock:
                    self._dead.add(ch)      # 这个通道确认不通，本次别再等它超时了
                continue
            if last_err:
                self._say(f"翻译失败（{type(last_err).__name__}）：可填 API Key 或代理后重试")
                self._reprobe_later()
            return None

    def _fire(self, ch: str, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        """发一次请求并记下时间（后台保活靠它判断是不是已经空闲了）。"""
        try:
            return self._dispatch(ch, text, src, timeout)
        finally:
            self._last_req = time.time()

    def start_keepalive(self) -> None:
        """空闲时每 15 秒发一个极小请求，把 HTTPS 连接保住。

        实测：连接冷掉的第一次请求要 1.3 秒（DNS+TCP+TLS），热连接只要 0.2~0.8 秒。
        直播里两句话之间常常隔十几秒，正好落在连接被掐断的窗口上。
        """
        self._last_req = getattr(self, "_last_req", 0.0) or time.time()
        if getattr(self, "_ka_started", False):
            return
        self._ka_started = True

        def ping(ch: str) -> None:
            try:
                self._dispatch(ch, "あ", timeout=2.5)
                self._last_req = time.time()
            except Exception:
                pass

        def run():
            while True:
                time.sleep(15)
                try:
                    if self._custom_ready():       # 用户自己配了 API，不必占免费通道额度
                        continue
                    if time.time() - self._last_req < 15:
                        continue
                    # 两个通道都保温：口语句走彩云、普通句走腾讯，冷掉哪个都会拖一句
                    for ch in ("tencent", "caiyun"):
                        if ch not in self._dead:
                            ping(ch)
                except Exception:
                    pass
        threading.Thread(target=run, daemon=True).start()
        if not self._custom_ready():
            threading.Thread(target=lambda: (
                ping("tencent"), ping("caiyun")), daemon=True).start()

    def _reprobe_later(self) -> None:
        """全部通道都挂了：后台悄悄再探一次，网络恢复了下一句就能自动用上。
        绝不在前台重试——那会让每句话都去等满超时。"""
        now = time.time()
        if now - getattr(self, "_last_probe_ts", 0.0) < 20.0:
            return
        self._last_probe_ts = now

        def run():
            try:
                if self.probe():
                    self._dead.clear()
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _looks_english(text: str) -> bool:
        """整句基本是拉丁字母（主播真在说英文，而不是片假名外来语）：
        这种情况直接走 EN→ZH 路由，比默认 JA→ZH 准得多。
        片假名外来语不算——它们由 apply_loanwords 在翻译前处理成正确中文义。"""
        if len([c for c in text if c.isascii() and c.isalpha()]) < 3:
            return False
        latin = sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))
        non_space = sum(1 for c in text if not c.isspace())
        if non_space == 0:
            return False
        return latin / non_space > 0.7

    @staticmethod
    def _is_junk(src: str, out: str) -> bool:
        """垃圾译文识别，命中就丢弃（只显示原文），两不换通道重试：
        1. 原文很短、译文却长到离谱 —— 免费通道返回的拼凑乱码。
        2. 译文是同一个短片段重复 4 次以上 —— 实测 MyMemory 把「やばいやばいやばい」
           翻成「狗屎狗屎狗屎狗屎狗屎狗屎狗屎」。注意阈值取 4：彩云给的
           「糟糕糟糕糟糕」是正确译文，只有 3 次重复，不能误杀。
        """
        if len(src) <= 12 and len(out) > max(20, len(src) * 4):
            return True
        n = len(out)
        for unit in (1, 2, 3):
            if n >= unit * 4 and n % unit == 0:
                piece = out[:unit]
                if out == piece * (n // unit) and n // unit >= 4:
                    return True
        return False

    def health_check(self, timeout: float = 2.5) -> List[tuple]:
        """逐个试各翻译通道并计时，供面板「通道健康检查」用。

        只做诊断、不改 _dead / _active：检查本身不该影响正在跑的翻译。
        返回 [(通道名, 是否可用, 耗时ms, 说明)]，耗时真实反映各通道快慢。
        """
        results: List[tuple] = []
        order: List[str] = []
        if self._custom_ready():
            order.append("custom")
        order += list(FREE_ORDER)
        for ch in order:
            name = CHANNEL_NAMES.get(ch, ch)
            t0 = time.time()
            try:
                out = self._dispatch(ch, self.PROBE_TEXT, src="ja", timeout=timeout)
            except Exception as exc:
                ms = int((time.time() - t0) * 1000)
                msg = f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")
                results.append((name, False, ms, msg[:64]))
                continue
            ms = int((time.time() - t0) * 1000)
            if not out or not out.strip():
                results.append((name, False, ms, "返回空"))
            elif self._is_junk(self.PROBE_TEXT, out):
                results.append((name, False, ms, f"疑似乱码：{out.strip()[:20]}"))
            else:
                # 通道本身能通，但被前面某次失败拉黑了 —— 这类最容易误判成「译文消失」
                extra = "（已被判不可用，重启或改设置会重试）" if ch in self._dead else ""
                results.append((name, True, ms, f"{out.strip()[:20]}{extra}"))
        return results

    def probe(self) -> Optional[str]:
        """依次尝试各通道，返回第一个可用的；全挂返回 None。"""
        for ch in self._order():
            try:
                out = self._dispatch(ch, "こんにちは")
                if out and out.strip():
                    self._active = ch
                    self._last_req = time.time()
                    self.start_keepalive()
                    self._say(f"翻译通道探测成功：{CHANNEL_NAMES.get(ch, ch)}")
                    return ch
            except Exception:
                continue
        self._say("所有翻译通道都不可用 —— 可在「翻译」里填 API Key 或代理")
        return None

    def clear_cache(self) -> None:
        with self.lock:
            self._cache.clear()

    # ---------------------------------------------------------- 代理
    @staticmethod
    def _port_open(port: int) -> bool:
        s = socket.socket()
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except Exception:
            return False
        finally:
            s.close()

    def detect_local_proxy(self) -> Optional[str]:
        """扫描本机常见代理端口，能用就返回代理地址。"""
        saved = dict(self.session.proxies)
        try:
            for port in PROXY_PORTS:
                if not self._port_open(port):
                    continue
                url = f"http://127.0.0.1:{port}"
                try:
                    r = requests.get(
                        "https://translate.googleapis.com/translate_a/single"
                        "?client=gtx&sl=ja&tl=zh-CN&dt=t&q=" + urllib.parse.quote("こんにちは"),
                        headers={"User-Agent": UA}, proxies={"http": url, "https": url}, timeout=7)
                    if r.status_code == 200 and r.json():
                        return url
                except Exception:
                    continue
        finally:
            self.session.proxies.clear()
            self.session.proxies.update(saved)
        return None
