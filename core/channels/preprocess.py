# -*- coding: utf-8 -*-
"""送翻前处理：上下文块、缓存 key、口语缩约/外来语/术语清洗、英文句路由。"""
from __future__ import annotations

from typing import Tuple

from ..context import format_context, format_free_context
from ..ja_norm import prepare as ja_prepare
from ..loanwords import apply_loanwords
from ..zh_rules import restore_contractions


def build_context(cfg, context, can_context: bool) -> Tuple[str, str]:
    """返回 (给 LLM/custom 的上下文块, 拼在免费通道文本前的极短前缀)。"""
    ctx_block = ""
    if context and cfg is not None and getattr(cfg, "enable_context", True) and can_context:
        try:
            ctx_block = format_context(
                context,
                getattr(cfg, "context_glossary", "") or "",
                int(getattr(cfg, "context_window", 3)),
            )
        except Exception:
            ctx_block = ""
    # 免费通道的语境前缀：独立开关、默认关闭。只取最近一句原文且截到 60 字，
    # 前缀越短，「整段一起翻/复述」的概率和延迟增量越小。
    free_ctx = ""
    if context and getattr(cfg, "enable_free_context", False):
        try:
            free_ctx = format_free_context(context)
        except Exception:
            free_ctx = ""
    return ctx_block, free_ctx


def cache_key(text: str, ctx_block: str, free_ctx: str):
    """带上下文/带前缀的译文与单句译文可能不同，缓存 key 必须区分，避免串味。"""
    if ctx_block:
        return (text, hash(ctx_block))
    if free_ctx:
        return (text, "free", free_ctx)
    return text


def prepare(text: str, cfg, source: str, say) -> Tuple[str, str]:
    """两级本地清洗后返回 (待送翻文本, 源语言)。

    1) ja_norm：展开口语缩约（「置いてってます」→「置いていきました」）+
       折叠 ASR 重复子句（「相当してください、相当ください」→ 只留一句）；
    2) loanwords：片假名外来语换成人工校准的中文义（アプデ=更新…）。
    术语表把专名换成人定的中文名再送翻（整词精确匹配）。
    两者都对着「上屏的原文」做，缓存 key 仍用原始 text，译文不串味。
    """
    pro = text
    if getattr(cfg, "enable_zh_rules", False):
        # 缩略还原属于送翻前处理：只展开歧义小的口语形（っつー/すんの…）
        pro, _c_notes = restore_contractions(pro)
    pro = ja_prepare(pro)
    if getattr(cfg, "enable_loanwords", True):
        pro = apply_loanwords(pro)
    # 翻译侧术语表：标题术语表在上游已经先做过一轮，这里只补用户/诊断新增的词，不会重复替换。
    if getattr(cfg, "enable_translate_glossary", False):
        try:
            from ..tr_glossary import apply_source
            pro, g_notes = apply_source(
                pro, getattr(cfg, "translate_glossary", "") or "")
            for n in g_notes:
                say(f"术语替换：{n}")
        except Exception:
            pass
    # 整句基本是拉丁字母（主播真在说英文，而非片假名外来语）：改走 EN→ZH 路由
    en_route = getattr(cfg, "enable_en_route", True) and looks_english(text)
    src = "en" if en_route else source
    return pro, src


def looks_english(text: str) -> bool:
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
