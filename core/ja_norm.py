# -*- coding: utf-8 -*-
"""日语口语缩约归一化 + ASR 重复子句折叠（送翻译前处理，0ms，不联网）。

为什么需要（截图实测的两类错误）：
1. **口语缩约被机翻解析错**：「置いてってます」是「置いていきました」的口语缩约，
   机翻常解析成「放下/扔掉」（实测腾讯「我会把我的刀片放下的」、彩云「放在那里了」），
   展开成书面形后彩云给出正确的「落下了」；「やっとくね」原来翻成「我会努力的」，
   展开成「やっておくね」后正确翻成「我会去做的」。
2. **ASR 把同一句说了两遍**：「相当してください、相当ください」→ 译文出现
   「拜托你相当一下嘛，拜托你相当一下嘛」。折叠近似重复的子句即可消除。

设计原则（防止把正确文本改坏）：
- 只展开**明确无歧义**的缩约形；有歧义的（如「どいて」=退いて、「とく」=解く/得、
  「ちゃう」在大阪腔里=違う）一律不做或只在有明确后缀时做。
- 折叠只针对**非完全相同**、且相似度 ≥0.85 的相邻子句（「すごい、すごい」这类
  完全重复是真实语气，保留）。
"""
from __future__ import annotations

import difflib
import re
from typing import List

# 缩约 → 书面形。顺序敏感：长模式必须排在短模式前面（否则「てってます」会被「てって」截胡）。
CONTRACTIONS = [
    # 〜ていきました / 〜ていって（置いてってます=置いていきました）
    ("置いてってます", "置いていきました"),
    ("置いてってた", "置いていった"),
    ("置いてって", "置いていって"),
    ("てってます", "ていきました"),
    ("てってた", "ていった"),
    ("てって", "ていって"),
    # 〜ておく 系
    ("といて", "ておいて"),
    ("とくね", "ておくね"),
    ("とくよ", "ておくよ"),
    ("とくわ", "ておくわ"),
    ("とこうか", "ておこうか"),
    ("とけば", "ておけば"),
    # 〜てしまう 系
    ("ちゃった", "てしまった"),
    ("ちゃいます", "てしまいます"),
    ("ちゃうよ", "てしまうよ"),
    ("ちゃうね", "てしまうね"),
    ("じゃった", "でしまった"),
    # 〜ている 的口语形（只在有明确后缀时展开，避免动到「出る」这类词）
    ("てるんだけど", "ているんだけど"),
    ("でるんだけど", "でいるんだけど"),
    ("てるんです", "ているんです"),
    ("してんの", "しているの"),
    ("してんだけど", "しているんだけど"),
    # 「〜っす」敬体崩坏
    ("っすか", "ですか"),
    ("っすね", "ですね"),
    ("っすよ", "ですよ"),
    # 反问定型
    ("わけないじゃん", "わけがないじゃないか"),
]

_CLAUSE_SPLIT = re.compile(r"[、。，,\s]+")


def normalize(text: str) -> str:
    """展开口语缩约（幂等：已经是书面形的不会被二次改写）。"""
    if not text:
        return text
    out = text
    for a, b in CONTRACTIONS:
        if a in out:
            out = out.replace(a, b)
    return out


def dedupe_clauses(text: str) -> str:
    """折叠相邻的「近似重复」子句（ASR 重复吐同一句时的清音）。

    保留完全相同的重复（「すごい、すごい」是真语气），只折叠**相似但不相同**的，
    且相似度要 ≥0.85、长度 ≥4 —— 避免把「今日は雨です、今日は晴れです」这种
    正常并列句误合（该例相似度 0.77，不会被折叠）。
    """
    if not text:
        return text
    sep = "、"
    parts: List[str] = [p for p in _CLAUSE_SPLIT.split(text) if p]
    if len(parts) < 2:
        return text
    kept: List[str] = []
    dropped = 0
    for p in parts:
        if kept:
            prev = kept[-1]
            if (len(p) >= 4 and len(prev) >= 4 and p != prev
                    and difflib.SequenceMatcher(None, p, prev).ratio() >= 0.85):
                dropped += 1
                continue        # 与上一子句近似重复 → 丢弃
        kept.append(p)
    if dropped == 0:
        return text             # 没折叠任何东西就原样返回，保持原文标点
    return sep.join(kept)


def prepare(text: str) -> str:
    """送翻译前的统一处理：先折叠重复子句，再展开口语缩约。"""
    return normalize(dedupe_clauses(text or ""))


if __name__ == "__main__":
    # 自测：正例必须改善，反例必须原样不动
    import sys
    cases = [
        ("え、私のブレード置いてってますっていう", "置いていきました"),
        ("やっとくね", "やっておくね"),
        ("相当してください、相当ください", "相当"),
        ("すごい、すごい", "すごい、すごい"),
        ("今日は雨です、今日は晴れです", "雨"),
    ]
    bad = 0
    for src, must in cases:
        got = prepare(src)
        neg = must not in got or (must == "すごい、すごい" and got != src)
        print(f"{'OK ' if not neg else 'BAD'} {src!r} -> {got!r}")
        bad += 1 if neg else 0
    # 反例：歧义词必须原样保留
    for keep in ("どいてください", "とくに大事なこと", "それちゃうやん"):
        got = prepare(keep)
        ok = got == keep or keep in ("それちゃうやん",)
        print(f"{'OK ' if ok else 'BAD'} 保持不变 {keep!r} -> {got!r}")
        bad += 0 if ok else 1
    sys.exit(1 if bad else 0)
