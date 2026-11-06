# -*- coding: utf-8 -*-
"""语境辅助：滑动窗口、日语同音词纠正、翻译上下文拼装。

设计取舍见 docs/语境翻译设计.md：
- 上下文**不进** Whisper 解码器：asr.py 记录过 initial_prompt / 热词被解码器
  复述的实测问题，字幕宁可少一句也不能上屏提示语；
- 识别层只做「有线索才改」的确定性纠正（変身→返信），默认 4 对规则，可开关、
  每次改动都上报；
- 翻译上下文只给能接受 system prompt 的自定义 LLM 通道，免费通道保持单句行为。
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# 同音词规则：wrong = 容易听错/写错的形，right = 语境正确形，
# cues = 支持 right 的线索词（当前句或最近几句里出现），
# guards = 反向线索（**只看当前句**，防止把真的「变身」改掉）。
# 只收高置信度、低误伤的对；用户自己的 word_map / name_roster 仍然优先于本表。
HOMOPHONES = (
    {
        "wrong": "変身", "right": "返信",
        "cues": ("メール", "LINE", "ライン", "返事", "連絡", "メッセージ", "DM", "既読",
                 "送る", "送った", "届く", "届いた"),
        "guards": ("姿", "魔法", "ヒーロー", "ベルト", "チェンジ", "スーツ", "ポーズ"),
    },
    {
        "wrong": "対照", "right": "対象",
        "cues": ("向け", "ユーザー", "年齢", "層", "ターゲット", "施策", "企画", "対象者"),
        "guards": ("比較", "比べ", "実験", "データ", "対照群", "対照的"),
    },
)
# 只收「错形在聊天语境里几乎不会出现」的对。像 機能/昨日、機関/期間 这种两边
# 都常用的同音词不进来——上下文线索很容易误命中，改错比不改更糟。

# 译文兜底：识别层已经把词认对了（原文含 返信），但译文仍出现「变身」时改回来。
# 只对单条明确规则生效——宁可少改，也不做模糊替换。
TRANSLATION_GUARDS = (
    {"src": "返信", "bad": "变身", "good": "回复"},
)


@dataclass
class Turn:
    """一句已经上屏的对话（原文 + 译文）。"""
    src: str
    tgt: str = ""


class ContextBuffer:
    """最近几句的滚动窗口。

    容量固定 8（够 cfg.context_window 的最大值），实际取几条由调用方按配置切片，
    这样运行中改 context_window 不需要重建对象。多线程读写加锁。
    """

    def __init__(self, maxlen: int = 8):
        self._lock = threading.Lock()
        self._turns: "deque[Turn]" = deque(maxlen=max(1, int(maxlen)))

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()

    def push(self, src: str, tgt: str = "") -> None:
        src = (src or "").strip()
        if not src:
            return
        with self._lock:
            self._turns.append(Turn(src, (tgt or "").strip()))

    def snapshot(self, limit: int = 3) -> Tuple[Tuple[str, str], ...]:
        """取最近 limit 条 (原文, 译文)；limit<=0 返回空。"""
        try:
            n = max(0, int(limit))
        except (TypeError, ValueError):
            n = 0
        if n == 0:
            return ()
        with self._lock:
            items = list(self._turns)[-n:]
        return tuple((t.src, t.tgt) for t in items)

    def sources(self, limit: int = 3) -> Tuple[str, ...]:
        """只要原文，供同音词纠正做线索匹配。"""
        return tuple(src for src, _ in self.snapshot(limit))


def _any_in(text: str, words: Iterable[str]) -> str:
    """返回命中的第一个线索词（空串 = 没命中），便于把依据写进日志。"""
    for w in words:
        if w and w in text:
            return w
    return ""


def fix_homophones(text: str, recent: Sequence[str] = (), enabled: bool = True
                   ) -> Tuple[str, List[str]]:
    """按上下文线索纠正日语同音词。

    返回 (纠正后文本, 说明列表)。规则未命中时原样返回，说明列表为空。
    """
    if not enabled or not text:
        return text, []
    ctx = "\n".join(list(recent) + [text])
    # 只剩一个词的感叹句（「変身！」「変身!!」）几乎都是字面意思，不纠正；
    # 这条结构化护栏比往 guards 里堆标点更稳
    bare = re.sub(r"[\s！!。、,.?？…〜~ー]+", "", text)
    notes: List[str] = []
    out = text
    for rule in HOMOPHONES:
        wrong, right = rule["wrong"], rule["right"]
        if wrong not in out or right in out:
            continue
        if bare == wrong:
            continue
        # 反向线索只看当前句：前一句在聊魔法，不代表这一句不是「返信」
        if _any_in(text, rule["guards"]):
            continue
        hit = _any_in(ctx, rule["cues"])
        if not hit:
            continue
        out = out.replace(wrong, right)
        notes.append(f"{wrong}→{right}（线索：{hit}）")
    return out, notes


def fix_translation(source: str, target: str, enabled: bool = True) -> Tuple[str, List[str]]:
    """译文兜底纠正：原文含明确词、译文却出现固定错译时改回。

    只处理 TRANSLATION_GUARDS 里的单条映射，命中才动。
    """
    if not enabled or not source or not target:
        return target, []
    out = target
    notes: List[str] = []
    for rule in TRANSLATION_GUARDS:
        if rule["src"] in source and rule["bad"] in out:
            out = out.replace(rule["bad"], rule["good"])
            notes.append(f"{rule['bad']}→{rule['good']}（原文含「{rule['src']}」）")
    return out, notes


def _parse_glossary(raw: str, limit: int = 20) -> List[Tuple[str, str]]:
    """解析「原词=译法」（支持 = 和 ＝），忽略空行/坏行。"""
    items: List[Tuple[str, str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("=", "＝"):
            if sep in line:
                k, v = line.split(sep, 1)
                k, v = k.strip(), v.strip()
                if k and v:
                    items.append((k[:60], v[:60]))
                break
        if len(items) >= limit:
            break
    return items


def format_free_context(pairs: Sequence[Tuple[str, str]], max_chars: int = 60) -> str:
    """给免费通道用的极短前缀：只取最近一句非空原文，压成一行并截断。

    免费通道没有上下文参数，只能把前缀拼进待翻译文本；前缀越短，
    「整段一起翻/复述」的概率和延迟增量越小。
    """
    for src, _tgt in reversed(list(pairs)):
        line = " ".join((src or "").split())
        if line:
            return line[:max_chars]
    return ""


def format_context(pairs: Sequence[Tuple[str, str]], glossary: str = "",
                   limit: int = 3, max_chars: int = 600) -> str:
    """把最近几轮 + 术语表拼成给 LLM 的上下文块（不需要时返回空串）。

    只保留有译文的轮次；整体截断到 max_chars，超长时从最旧的一条开始丢。
    """
    try:
        n = max(0, int(limit))
    except (TypeError, ValueError):
        n = 0
    rows = [(s.strip(), t.strip()) for s, t in list(pairs)[-n:] if s and t] if n else []
    terms = _parse_glossary(glossary)

    def render(rs, ts) -> str:
        if not rs and not ts:
            return ""
        parts = ["最近的对话（仅供理解语境，不要翻译、不要复述）："]
        for s, t in rs:
            parts.append(f"原文：{s}")
            parts.append(f"译文：{t}")
        if ts:
            parts.append("术语表（当前句出现时按此译法）：")
            parts.extend(f"{k}→{v}" for k, v in ts)
        return "\n".join(parts)

    block = render(rows, terms)
    # 超长：先丢最旧的对话，术语表最后才丢
    while block and len(block) > max_chars:
        if rows:
            rows = rows[1:]
        elif terms:
            terms = terms[:-1]
        else:
            break
        block = render(rows, terms)
    return block if block and (rows or terms) else ""
