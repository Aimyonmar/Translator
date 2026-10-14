# -*- coding: utf-8 -*-
"""翻译目标语言对比实验：直译中文 vs 直译英文 vs 英文中转回中文。

回答「翻成中文是不是太难了 / 换成英文会不会更好」。
对同一批 12 句直播口语，分别测：
  腾讯·中 / 彩云·中 / 腾讯·英 / 彩云·英 / 腾讯(英→中) / 彩云(英→中) (+Edge，若可达)
刻意绕开生产链路的本地加成（短语表命中会直接返回、不进网络），
只测「引擎在不同目标语言下的原始能力」；同时标注每句在生产链路里
会被哪些本地层预修（ja_norm 展约 / 外来语 / 标题术语表）。

结果：控制台表格 + mt_target_result.json
用法：.runtime\\venv\\Scripts\\python.exe mt_target_test.py
"""
from __future__ import annotations

import html
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import AppConfig                                # noqa: E402
from core.translator import Translator                           # noqa: E402
from core.phrases import is_colloquial, lookup as phrase_lookup  # noqa: E402
from core.ja_norm import prepare as ja_prepare                   # noqa: E402
from core.loanwords import apply_loanwords                       # noqa: E402
from core.namefix import apply_title_glossary                    # noqa: E402

SENTS = [
    "え、私のブレード置いてってますっていう",            # 缩约+引用（历史截图句）
    "めっちゃ眠いんだけど、今日はもう配信終わるね",        # 口语上句（历史句）
    "このボス、ガチで無理なんだけど！",                   # 游戏词：ボス / ガチ
    "神ってる、このプレイ！まじですごい",                  # 网络语：神ってる
    "やっとくね、あとで連絡するわ",                        # 缩约：やっとく
    "えぐい、このダメージえぐすぎるでしょ",                # 新网络语：えぐい
    "とりま、ちょっと待っててくれる？",                    # 缩约：とりま
    "それな、わかるわかる。俺もそう思う",                  # 相槌
    "あざます、助かったわ",                                # 网络语：あざます
    "普通にうまくない？この立ち回り",                      # 普通に（易翻成「普通地」）
    "今日の配信、エルデンリングのDLCやっていくよ",         # 标题（生产走术语表）
    "モンハンワイルズの発売日、やっと決まったらしいよ",    # 标题（生产走术语表）
]


def timed(fn, *a, **kw):
    t0 = time.time()
    try:
        out = fn(*a, **kw)
        return (out or "").strip(), int((time.time() - t0) * 1000), None
    except Exception as exc:                                     # noqa: BLE001
        return None, int((time.time() - t0) * 1000), f"{type(exc).__name__}: {exc}"


def call(tr: Translator, name: str, text: str, src: str, tgt: str,
         timeout: float = 6.0, retries: int = 0, gap: float = 0.0):
    """切 target → 调通道 →（可选）失败重试。彩云 demo token 会限流，必须能重试。"""
    tr.cfg.target_lang = tgt
    fn = getattr(tr, name)
    out = ms = err = None
    for k in range(retries + 1):
        if gap:
            time.sleep(gap)
        out, ms, err = timed(fn, text, src, timeout)
        if err is None and out:
            return out, ms, None
        if k < retries:
            time.sleep(1.5)
    return out, ms, err


def caiyun_many(tr: Translator, texts, src: str, tgt: str, timeout: float = 20.0):
    """与 core.translator._tr_caiyun 同端点、同参数，只是 source 传列表（省限额）。

    彩云官方 demo token 一天能用的次数有限，逐句 36 次调用很容易中途被限流；
    批量一次往返能把次数压到个位数。失败时调用方会降级为逐句。
    """
    token = (getattr(tr.cfg, "caiyun_token", "") or "").strip()
    tgtc = {"zh-Hans": "zh", "zh-Hant": "zh-Hant", "en": "en"}.get(tgt, "zh")
    r = tr.session.post(
        "https://api.interpreter.caiyunai.com/v1/translator", timeout=timeout,
        proxies=tr._no_proxy,                                    # 国内直连
        headers={"Content-Type": "application/json", "X-Authorization": f"token {token}"},
        data=json.dumps({"source": list(texts), "trans_type": f"{src}2{tgtc}"},
                        ensure_ascii=False).encode("utf-8"),
    )
    r.raise_for_status()
    data = r.json()
    if data.get("rc") not in (0, None):
        raise RuntimeError(f"caiyun rc={data.get('rc')}")
    out = [html.unescape(x).strip() for x in (data.get("target") or [])]
    if len(out) != len(texts):
        raise RuntimeError(f"批量返回条数不符：{len(out)} != {len(texts)}")
    return out


def pre_marks(s: str) -> str:
    """标注该句在生产链路里会被哪些本地层预修（0ms、不联网）。"""
    marks = []
    p = ja_prepare(s)
    if p != s:
        marks.append("展约")
    l = apply_loanwords(p)
    if l != p:
        marks.append("外来语")
    g = apply_title_glossary(l, s)
    if g != l:
        marks.append("术语表")
    return "、".join(marks) or "无"


def main() -> None:
    cfg = AppConfig.load()
    tr = Translator(cfg, on_log=lambda m: None)

    # ---- Edge 可用性探测（直连 → 代理），最多 ~7 秒；不可用就跳过
    edge_via = None
    for label, px in (("直连", None), ("代理", cfg.proxy or None)):
        if label == "代理" and not px:
            continue
        try:
            if px:
                tr.session.proxies.update({"http": px, "https": px})
            else:
                tr.session.proxies.clear()
            tr._edge_auth(timeout=3.0)
            edge_via = label
            break
        except Exception:
            tr.session.proxies.clear()
            continue
    if edge_via is None:
        tr.session.proxies.clear()
    print(f"Edge 通道：{edge_via or '不可用（跳过）'}")
    print(f"彩云 token：{'已配置' if cfg.caiyun_token else '未配置'}")
    print("=" * 100)

    # ---- 彩云批量（3 次请求拿全所有彩云列），失败则标 None 走逐句降级
    c_zh_list = c_en_list = None
    try:
        c_zh_list = caiyun_many(tr, SENTS, "ja", "zh-Hans")
        time.sleep(0.6)
        c_en_list = caiyun_many(tr, SENTS, "ja", "en")
    except Exception as exc:                                     # noqa: BLE001
        print(f"[提示] 彩云批量失败（{type(exc).__name__}），降级为逐句：{exc}")

    rows = []
    fails = {"tencent": 0, "caiyun": 0}
    for i, ja in enumerate(SENTS):
        row = {"i": i + 1, "ja": ja,
               "colloquial": is_colloquial(ja),
               "phrase": phrase_lookup(ja),
               "pre": pre_marks(ja)}

        t_zh, t_zh_ms, t_zh_e = call(tr, "_tr_tencent", ja, "ja", "zh-Hans", gap=0.15)
        t_en, t_en_ms, t_en_e = call(tr, "_tr_tencent", ja, "ja", "en", gap=0.15)

        if c_zh_list is not None and c_zh_list[i]:
            c_zh, c_zh_ms, c_zh_e = c_zh_list[i], 0, None
        else:
            c_zh, c_zh_ms, c_zh_e = call(tr, "_tr_caiyun", ja, "ja", "zh-Hans",
                                         retries=1, gap=0.35)
        if c_en_list is not None and c_en_list[i]:
            c_en, c_en_ms, c_en_e = c_en_list[i], 0, None
        else:
            c_en, c_en_ms, c_en_e = call(tr, "_tr_caiyun", ja, "ja", "en",
                                         retries=1, gap=0.35)

        p_t, p_t_ms, p_t_e = ((None, 0, "无英文输出") if not t_en else
                              call(tr, "_tr_tencent", t_en, "en", "zh-Hans", gap=0.15))
        if not c_en:
            p_c, p_c_ms, p_c_e = None, 0, "无英文输出"
        elif c_zh_list is not None:
            p_c, p_c_ms, p_c_e = None, 0, "见彩云批量（英→中另批）"
        else:
            p_c, p_c_ms, p_c_e = call(tr, "_tr_caiyun", c_en, "en", "zh-Hans",
                                      retries=1, gap=0.35)

        if t_zh_e or t_en_e:
            fails["tencent"] += 1
        if c_zh_e or c_en_e:
            fails["caiyun"] += 1

        row.update({"t_zh": t_zh, "t_zh_ms": t_zh_ms, "t_zh_e": t_zh_e,
                    "c_zh": c_zh, "c_zh_ms": c_zh_ms, "c_zh_e": c_zh_e,
                    "t_en": t_en, "t_en_ms": t_en_ms, "t_en_e": t_en_e,
                    "c_en": c_en, "c_en_ms": c_en_ms, "c_en_e": c_en_e,
                    "p_t": p_t, "p_t_ms": p_t_ms, "p_t_e": p_t_e,
                    "p_c": p_c, "p_c_ms": p_c_ms, "p_c_e": p_c_e})
        if edge_via:
            row["e_zh"], row["e_zh_ms"], row["e_zh_e"] = call(
                tr, "_tr_edge", ja, "ja", "zh-Hans", timeout=6)
            row["e_en"], row["e_en_ms"], row["e_en_e"] = call(
                tr, "_tr_edge", ja, "ja", "en", timeout=6)
        rows.append(row)

        show = lambda v, ms, e: (e if e else (v or "（空）")) + f"  [{ms}ms]"  # noqa: E731
        print(f"[{row['i']}] {ja}")
        print(f"  口语标记:{'是' if row['colloquial'] else '否'}  "
              f"本地表:{row['phrase'] or '—'}  生产预修:{row['pre']}")
        print(f"  腾讯→中 : {show(t_zh, t_zh_ms, t_zh_e)}")
        print(f"  彩云→中 : {show(c_zh, c_zh_ms, c_zh_e)}")
        print(f"  腾讯→英 : {show(t_en, t_en_ms, t_en_e)}")
        print(f"  彩云→英 : {show(c_en, c_en_ms, c_en_e)}")
        print(f"  腾讯 英→中: {show(p_t, p_t_ms, p_t_e)}")
        print(f"  彩云 英→中: {show(p_c, p_c_ms, p_c_e)}")
        if edge_via:
            print(f"  Edge→中  : {show(row['e_zh'], row['e_zh_ms'], row['e_zh_e'])}")
            print(f"  Edge→英  : {show(row['e_en'], row['e_en_ms'], row['e_en_e'])}")
        print("-" * 100)

    # 彩云英→中中转（用批量再省一次限额）
    if c_en_list is not None and all(c_en_list):
        try:
            p_c_list = caiyun_many(tr, c_en_list, "en", "zh-Hans")
            for row, val in zip(rows, p_c_list):
                row["p_c"] = val
                row["p_c_e"] = None
                row["p_c_ms"] = 0
        except Exception as exc:                                 # noqa: BLE001
            print(f"[提示] 彩云 英→中 批量失败：{exc}")
    print("=" * 100)
    print("----- 彩云英→中（批量补跑）-----")
    for row in rows:
        note = row.get("p_c_e") or ""
        print(f"[{row['i']}] {row['p_c'] or note}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mt_target_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("=" * 100)
    print(f"失败计数：腾讯 {fails['tencent']} 句 / 彩云 {fails['caiyun']} 句")
    print("彩云限流重试后仍失败的句子会在表里显示错误名，对照 JSON 可查")
    print(f"结果已存：{out_path}")


if __name__ == "__main__":
    main()
