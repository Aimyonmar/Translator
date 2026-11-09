# -*- coding: utf-8 -*-
"""翻译模块：免费在线通道 + 可选自定义 API + 代理支持。

免费通道（无需 Key）：Edge·必应 / Google / DeepL网页 / MyMemory，自动探测可用的那个。
自定义 API（可选，质量最好）：DeepL 免费版 Key，或任意 OpenAI 兼容接口（DeepSeek、通义、豆包…）。
代理：国内网络可填 HTTP 代理（如 http://127.0.0.1:7890）解锁 Google / Edge，程序也会自动探测本机代理端口。

通道适配、主备竞速、保活等实现见 core/channels/；本文件保留 Translator 门面与翻译编排。
"""
from __future__ import annotations

import html
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional

from .channels import custom as ch_custom
from .channels import free as ch_free
from .channels import http as ch_http
from .channels import postprocess, preprocess, racing
from .channels.constants import (CHANNELS, CHANNEL_NAMES, COLLOQUIAL_ORDER, FREE_ORDER,
                                 LANG_NAME_BY_CODE, LANGS, PROXY_PORTS, UA)
from .channels.keepalive import Keepalive
from .phrases import is_colloquial, lookup as phrase_lookup


class Translator:
    # 免费通道的翻译总预算（秒）。字幕是实时的，宁可这句只显示原文，
    # 也不能让画面卡在那里等网络。
    BUDGET = 5.0

    # 健康检查/探测用的短句：短、且译文确定，便于判断通道是不是真在工作
    PROBE_TEXT = "こんにちは"

    def __init__(self, cfg=None, on_log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.log_cb = on_log
        self.session = ch_http.build_session()
        self.lock = threading.Lock()
        self._edge = ch_free.EdgeAuth()
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        # 上次用通的通道：冷启动直接复用，避免每次都从头把不通的通道挨个超时一遍
        # （edge/google/deepl 在国内常常要各等 10 秒才失败，首次出字幕会拖到 20-30 秒后）
        self._active: Optional[str] = getattr(cfg, "last_channel", "") or None
        self._dead: set = set()      # 本次会话内已确认不通的通道，不再重试
        # 计时诊断用：记录「这次请求最终由谁返回」。两个翻译线程并发，
        # 所以用 thread-local，避免互相覆盖（见 core/timing.py）。
        self._tls = threading.local()
        # 双发翻译（见改进记录第二十节）：默认主备式——主通道超过 hedge_backup_ms
        # 未返回才补发第二条，取先到的可用译文。目的不是降均值，而是砍掉单条线路
        # 偶发的长尾；实测 P50/P90 不变、P99 少约 0.3 s，请求只多约 6%。
        # single=纯串行（旧行为），dual=同时发（请求翻倍，仅实验用）。
        self.hedge_mode = "backup" if getattr(cfg, "enable_hedge", True) else "single"
        self.backup_delay_ms = int(getattr(cfg, "hedge_backup_ms", 400) or 400)
        # 请求/失败计数与每条线路的实测往返，只用于 A/B 统计
        self.req_count = 0
        self.fail_count = 0
        self.route_log: list = []
        # 最近一条字幕命中的译后规则（供诊断/面板展示，不影响翻译）
        self.last_rule_hits: list = []
        self._route_lock = threading.Lock()
        self._gid = 0
        # 保活线程：默认不起（cfg.enable_keepalive=False）。同一时刻最多一条，
        # 每轮换一个新的 Event，避免 stop→start 之间把上一轮线程「复活」。
        self._ka = Keepalive(self)
        self.proxy: str = getattr(cfg, "proxy", "") or ""
        # 腾讯/彩云是国内直连通道，本来就不需要代理；而本机代理（127.0.0.1:7897）经常不稳，
        # 一旦让它们走代理，译文请求会集体超时被放弃 —— 字幕就只剩原文、译文消失。
        # requests 的 merge_setting 会把值为 None 的键从 session.proxies 里删掉，
        # 所以传它等于「本次请求不走代理」。
        self._no_proxy = {"http": None, "https": None}
        # LLM 质量通道（可选）：开关关着、provider 不认识或没填 Key 时 available=False，
        # 此时 "llm" 根本不会进入 _order()，现有链路的行为与延迟一字不变。
        self._llm = None
        try:
            from .llm_tr import LLMTranslator
            self._llm = LLMTranslator(cfg)
        except Exception:
            self._llm = None

    def _llm_ready(self) -> bool:
        try:
            return bool(self._llm and self._llm.available)
        except Exception:
            return False

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
        ch_http.apply_proxy(self.session, self.proxy)

    def _custom_ready(self) -> bool:
        c = self.cfg
        # 去掉空白再判空：否则手工把 Key 清成 "   " 时会被当成已配置，白占 custom 优先级
        # （custom 通道里也是 strip 后为空就直接返回 None）。
        key = (getattr(c, "custom_key", "") or "").strip()
        return bool(key and getattr(c, "custom_type", "openai"))

    # ---------------------------------------------------------- 通道
    def _tr_custom(self, text: str, src: str = "ja", context: str = "") -> Optional[str]:
        return ch_custom.custom(self.session, self.cfg, self.target, text, src, context)

    def _tr_edge(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        return ch_free.edge(self.session, self._edge, self.target, text, src, timeout)

    def _tr_google(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        return ch_free.google(self.session, self.target, text, src, timeout)

    def _tr_deepl(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        return ch_free.deepl(self.session, self.target, text, src, timeout)

    def _tr_mymemory(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        return ch_free.mymemory(self.session, self.target, text, src, timeout)

    def _tr_tencent(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        return ch_free.tencent(self.session, self._no_proxy, self.target, text, src, timeout)

    def _tr_caiyun(self, text: str, src: str = "ja", timeout: float = 4.0) -> Optional[str]:
        return ch_free.caiyun(self.session, self._no_proxy, self.cfg, self.target,
                              text, src, timeout)

    def _tr_llm(self, text: str, src: str = "ja", context: str = "") -> Optional[str]:
        """LLM 质量通道：不可用或失败都返回 None，由调用方回落免费通道。"""
        if not self._llm_ready():
            return None
        return self._llm.translate(text, context=context, src=src)

    def _dispatch(self, ch: str, text: str, src: str = "ja", timeout: float = 4.0,
                  context: str = "") -> Optional[str]:
        fn = {"custom": self._tr_custom, "llm": self._tr_llm,
              "tencent": self._tr_tencent, "caiyun": self._tr_caiyun,
              "edge": self._tr_edge, "google": self._tr_google,
              "deepl": self._tr_deepl, "mymemory": self._tr_mymemory}.get(ch)
        if fn is None:
            return None
        if ch in ("custom", "llm"):
            # 用户自己配的通道 / LLM 质量通道：能接受上下文，愿意多等，不受实时预算限制
            return fn(text, src, context)
        # 免费通道没有上下文入口：传了也不用，避免把前文拼进请求
        return fn(text, src, timeout)

    def _order(self, text: str = "") -> List[str]:
        # 口语化的句子优先走彩云（理解最准），普通句优先走腾讯（最快）。
        # 这里不把「上次用通的通道」提到最前：免费通道的优劣是固定的，
        # 上次记住的可能只是当时唯一能通的慢通道（比如 MyMemory 831ms），
        # 让它一直排在最前，用户就永远用不上快 4 倍的腾讯。
        base = list(COLLOQUIAL_ORDER if (text and is_colloquial(text)) else FREE_ORDER)
        free = [c for c in base if c not in self._dead] or list(base)
        ch = self.channel
        # 用户在面板里显式选了某个通道时，尊重选择，不把 LLM 插到它前面
        if ch in CHANNELS and ch != "auto":
            if ch == "custom" and self._custom_ready():
                return ["custom"] + free
            return [ch] + [c for c in free if c != ch]
        head = []
        if self._llm_ready():
            head.append("llm")
        if self._custom_ready():
            head.append("custom")
        return head + free

    # ---------------------------------------------------------- 对外
    def translate(self, text: str, context=()) -> Optional[str]:
        """翻译一句。context=((原文, 译文), …) 只对自定义 LLM 通道生效。

        上下文为空、开关关闭或没配 API Key 时，走的完全是原来的单句路径。
        """
        text = (text or "").strip()
        if not text:
            return None
        self._tls.channel = ""
        # 本句的主备赛跑结果（是否备胎救回、晚到多少、谁胜出），没走主备时为 None
        self._tls.hedge = None
        # 上下文块在本地拼装（<0.1ms）：只有能接受 system prompt 的 custom 通道用得上
        ctx_block, free_ctx = preprocess.build_context(
            self.cfg, context, self._custom_ready() or self._llm_ready())
        cache_key = preprocess.cache_key(text, ctx_block, free_ctx)
        # 只在这一小段里持锁：网络请求绝不能放进锁里——
        # 之前整段都锁着，2 路并发翻译实际退化成串行排队，实测单句要等 4.7 秒。
        # 先查本地口语表：命中就是 0ms，且译文是人工校准过的，比任何免费机翻都准
        with self.lock:
            if cache_key in self._cache:
                # 命中要移到队尾：淘汰走 popitem(last=False)，不刷新就是按写入顺序的 FIFO，
                # 反复出现的口头禅会被一次性新句挤掉，白白重发请求
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            hit = phrase_lookup(text)
            if hit:
                self._cache[cache_key] = hit
                self._phrase_hits = getattr(self, "_phrase_hits", 0) + 1
                if len(self._cache) > 800:
                    self._cache.popitem(last=False)
                return hit
            order = self._order(text)
            custom = self._custom_ready()
            # 送翻前的两级清洗都在本地（0ms），详见 core/channels/preprocess.py
            pro, src = preprocess.prepare(text, self.cfg, self.source, self._say)

        last_err = None
        gid = self._next_gid()      # A/B：同一句的线路记录共用这个号
        # 实时字幕有总预算：免费通道挨个试过去最坏能拖到二三十秒（实测 21 秒），
        # 字幕就彻底跟不上说话了。超预算立刻放弃译文、只显示原文。
        deadline = time.time() + (25.0 if (custom or self._llm_ready()) else self.BUDGET)
        switched = False
        # A/B：只对前两条免费通道做双发/主备；custom / llm 和带免费语境前缀的路径不参与
        hedge_chs = [c for c in order if c not in ("custom", "llm")][:2]
        use_hedge = self.hedge_mode != "single" and len(hedge_chs) == 2 and not free_ctx
        # 双发的位置必须是「第一条免费通道」，不能是 i==0：
        # 配了 custom 时 order 的第一项是 custom，那样会直接把 custom 顶掉。
        hedge_i = order.index(hedge_chs[0]) if use_hedge else -1
        for i, ch in enumerate(order):
            left = deadline - time.time()
            if left <= 0.2:
                break
            # 免费通道没有上下文参数：把极短前缀用换行拼在待翻译文本前面，
            # 再靠换行把当前句那段切出来（custom 通道走 system 上下文，不拼前缀）
            use_free = bool(free_ctx) and ch not in ("custom", "llm")
            send = (free_ctx + "\n" + pro) if use_free else pro
            if i == hedge_i:
                pick = self._race(
                    hedge_chs, send, src, timeout=min(4.0, left),
                    delay_ms=self.backup_delay_ms if self.hedge_mode == "backup" else 0,
                    gid=gid)
                if pick is None:
                    continue
                out, ch = pick
            else:
                try:
                    # 只剩多少时间就只给多少超时：只在循环开头检查预算是拦不住的，
                    # 单次请求（实测 mymemory 4s + edge 鉴权 3s = 8 秒）会把预算整个冲破
                    out = self._timed_fire(ch, send, src, timeout=min(4.0, left),
                                           context=ctx_block if ch in ("custom", "llm") else "",
                                           gid=gid)
                except Exception as exc:
                    last_err = exc
                    with self.lock:
                        self._dead.add(ch)      # 这个通道确认不通，本次别再等它超时了
                    continue
            # 空返回多半是通道临时抽风：换下一个通道，别让整句直接没有译文
            if not (out and out.strip()):
                continue
            out = html.unescape(out.strip())
            self._tls.channel = ch
            # 免费通道语境护栏：请求里拼了一段前文，模型可能把前文一起翻出来，
            # 甚至整段改写成一句话。只认换行分段、只取最后一段（当前句）；
            # 切不开或长度明显超长就当作污染，用干净单句重发一次，拿不到就换通道。
            if use_free:
                seg = postprocess.split_free_segment(out)
                if seg is None or len(seg) > max(60, len(text) * 3):
                    self._say("免费语境分段失败，回退单句")
                    left = deadline - time.time()
                    plain = None
                    if left > 0.2:
                        try:
                            plain = self._fire(ch, pro, src, timeout=min(4.0, left))
                        except Exception as exc:
                            last_err = exc
                    if not (plain and plain.strip()):
                        continue
                    out = html.unescape(plain.strip())
                else:
                    out = seg
            # 上下文污染护栏：模型偶尔会把前文一起翻出来（字幕里出现两句以上），
            # 长度明显超出当前句就直接丢弃，让界面只显示原文
            if ctx_block and len(out) > max(60, len(text) * 3):
                self._say("译文疑似混入上下文，已丢弃")
                return None
            # 免费通道（尤其 MyMemory 匿名额度）对很短的句子会返回一大串拼凑乱码，
            # 比如「こんにちは」→ "penian ferein OU潘托斯所有安卓sofou"。
            # 这里绝不能 continue 去试其它通道：edge/google/deepl 都要等满超时，
            # 实测会把这句的翻译拖到 12 秒，字幕就彻底跟不上说话了。
            # 直接放弃译文，让界面只显示原文。
            if postprocess.is_junk(text, out):
                return None
            # 译后润色与规则改写（都在本地、0ms），详见 core/channels/postprocess.py
            out, rule_hits = postprocess.polish(self.cfg, text, out, self._say)
            if rule_hits:
                self.last_rule_hits = rule_hits
            with self.lock:
                self._cache[cache_key] = out
                if len(self._cache) > 800:
                    self._cache.popitem(last=False)
                if self._active != ch:
                    self._active = ch
                    # "llm" 不在 CHANNELS 里，写进 last_channel 会让下次开机
                    # 出现一个面板选不中的值；它的可用性完全取决于 Key，不该被记住
                    if ch != "llm":
                        self._remember(ch)
                    switched = True
            if switched:
                self._say(f"翻译通道：{CHANNEL_NAMES.get(ch, ch)}")
            return out

        # 走到这里说明所有通道都没给出可用译文（超预算 / 空返回 / 全部异常）
        if last_err:
            self._say(f"翻译失败（{type(last_err).__name__}）：可填 API Key 或代理后重试")
        self._reprobe_later()
        return None

    def _next_gid(self) -> int:
        """给每句翻译分配一个分组号：A/B 时同一句的两条线路记录靠它配对。"""
        with self._route_lock:
            self._gid += 1
            return self._gid

    def _timed_fire(self, ch: str, text: str, src: str = "ja", timeout: float = 4.0,
                    context: str = "", gid: int = 0) -> Optional[str]:
        """_fire + 记录这条线路的实际往返，供 A/B 统计（不影响正常路径）。"""
        t0 = time.perf_counter()
        ok = False
        try:
            out = self._fire(ch, text, src, timeout, context)
            ok = bool(out and out.strip())
            return out
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            with self._route_lock:
                self.route_log.append({"gid": gid, "ch": ch, "ms": round(ms, 1),
                                       "ok": ok, "text": text[:24]})
                if len(self.route_log) > 2000:      # 长直播别无限涨
                    del self.route_log[:-2000]

    def _fire(self, ch: str, text: str, src: str = "ja", timeout: float = 4.0,
              context: str = "") -> Optional[str]:
        """发一次请求并记下时间（后台保活靠它判断是不是已经空闲了）。"""
        with self.lock:
            self.req_count += 1
        try:
            return self._dispatch(ch, text, src, timeout, context)
        except Exception:
            with self.lock:
                self.fail_count += 1
            raise
        finally:
            self._last_req = time.time()

    def _race(self, chs: List[str], text: str, src: str, timeout: float,
              delay_ms: int, gid: int = 0) -> Optional[tuple]:
        """主备/双发：先发 chs[0]；delay_ms 内没结果（或先失败了）就补发 chs[1]。

        返回 (译文, 通道)；两条都不给可用译文时返回 None，交回原来的串行兜底。
        败方请求在后台跑完后会把往返记进 route_log，供 A/B 统计，不阻塞上屏。
        """
        def work(ch):
            try:
                return self._timed_fire(ch, text, src, timeout=timeout, gid=gid)
            except Exception:
                # 与串行路径一致：确认不通就本次会话不再重试，否则主备模式会每句都
                # 先撞一次同一个坏通道
                with self.lock:
                    self._dead.add(ch)
                return None

        best, done_at, fired = racing.race(
            chs, work, lambda out: not self._is_junk(text, out), timeout, delay_ms)
        # 上报口径：rescued=胜出的是备胎；late_ms=备胎完成比主通道完成晚多少（都拿到时刻才算，
        # 主通道超时/失败时没有参照，记 None）；winner=最终胜出通道。
        info = None
        if best is not None:
            winner = best[1]
            primary = chs[0]
            late = None
            if winner != primary:
                pw, bw = done_at.get(primary), done_at.get(winner)
                if pw is not None and bw is not None:
                    late = max(0.0, (bw - pw) * 1000.0)
            info = {"primary": primary, "backup": chs[1], "winner": winner,
                    "rescued": winner != primary, "backup_fired": fired > 1,
                    "late_ms": None if late is None else round(late, 1)}
            if info["rescued"]:
                self._say("备胎救回：%s 晚 %s（主通道 %s%s）"
                          % (winner,
                             "%.0fms" % late if late is not None else "未返回",
                             primary, "" if late is not None else "，无参照"))
        self._tls.hedge = info
        return best

    # ---------------------------------------------------------- 保活
    def start_keepalive(self) -> None:
        """空闲时每 15 秒发一个极小请求，把 HTTPS 连接保住。

        只有 cfg.enable_keepalive 为真时才创建线程（默认关闭）：关闭时零线程、零请求。
        开关可反复切换，同一时刻最多一条保活线程。

        实测：连接冷掉的第一次请求要 1.3 秒（DNS+TCP+TLS），热连接只要 0.2~0.8 秒。
        直播里两句话之间常常隔十几秒，正好落在连接被掐断的窗口上。
        """
        self._ka.start()

    def stop_keepalive(self) -> None:
        """停掉保活并阻止再次自启（直到 resume_keepalive）。

        线程阻塞在 Event.wait 上，set() 后立即返回并退出；Pipeline.stop() 也走这里，
        所以程序停止后不会再有任何 ping。
        """
        self._ka.stop()

    def resume_keepalive(self) -> None:
        """解除 stop_keepalive 的抑制（新一轮开始翻译时调用），不主动起线程。"""
        self._ka.resume()

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
        """整句基本是拉丁字母（主播真在说英文，而不是片假名外来语）。"""
        return preprocess.looks_english(text)

    @staticmethod
    def _split_free_segment(out: str) -> Optional[str]:
        """从「前缀+当前句」的译文里取出当前句：只认换行分段。"""
        return postprocess.split_free_segment(out)

    @staticmethod
    def _is_junk(src: str, out: str) -> bool:
        """垃圾译文识别，命中就丢弃（只显示原文），不换通道重试。"""
        return postprocess.is_junk(src, out)

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

    def last_channel(self) -> str:
        """当前线程最近一次 translate 实际用到（或拿到响应）的通道，仅诊断用。"""
        return getattr(self._tls, "channel", "") or ""

    def last_hedge(self) -> Optional[dict]:
        """当前线程这一句的主备结果：是否备胎救回、晚到毫秒、胜出通道。

        没走主备（单通道、关闭 hedge、或有上下文前缀）时为 None；只读，不影响翻译。
        """
        return getattr(self._tls, "hedge", None)

    # ---------------------------------------------------------- 代理
    @staticmethod
    def _port_open(port: int) -> bool:
        return ch_http.port_open(port)

    def detect_local_proxy(self) -> Optional[str]:
        """扫描本机常见代理端口，能用就返回代理地址。"""
        return ch_http.detect_local_proxy(self.session)
