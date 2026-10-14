# -*- coding: utf-8 -*-
"""中文译文口语化润色（确定性规则，0ms，无模型）。

针对的是免费机翻通道（腾讯/MyMemory 等）在「VTuber 直播字幕」场景下的固定生硬模式，
全部规则来自真实译文的对比实测（2026-10-13，腾讯通道 12 句）：

  1. 称呼错位：「みんな」→「伙计们」（直播语境应为「大家」）；
  2. 敬语错位：「您」——直播口语场景统一切「你」；
  3. 书面腔：「想尝试」→「想试试」、「请不要」→「别」、「非常地」→「非常」；
  4. 标点混乱：半角 !?;, 与全角混排（「我很高兴!」），且半角标点后常带一个空格
     （「伙计们! 我」）——顺序必须是：**先半角→全角，再清空格**，反了空格清不掉；
  5. 书面句号：直播字幕惯例去掉句尾「。」（半角「.」同样处理；保留 ？！，语气不丢）；
  6. 机翻毛刺：叠字「的的/了了」、CJK 与标点间多余空格。

设计原则：**只做高置信度的确定性替换，绝不动句法和词序**——
润色错了顶多不自然，绝不会像改写那样引入新错误。
LLM 通道（custom/DeepSeek）的译文本身就口语化，但过了这层也无害（幂等）。
"""
from __future__ import annotations

import re
from typing import List, Tuple

__all__ = ["zh_polish", "polish_with_log"]

# ---------------------------------------------------------------- 词级替换
# （原文模式, 替换）——顺序敏感：长的放前面，先匹配长串。
_WORD_RULES: List[Tuple[str, str]] = [
    # 称呼：直播语境
    ("伙计们", "大家"),
    ("各位伙计", "大家"),
    ("诸位", "各位"),
    # 敬语错位：口语场景统一
    ("您", "你"),
    # 书面腔 → 口语
    ("请不要", "别"),
    ("请勿", "别"),
    ("想尝试", "想试试"),
    ("要尝试", "要试试"),
    ("尝试一下", "试试"),
    ("非常地", "非常"),
    ("特别地", "特别"),
    ("相当地", "相当"),
    ("相当程度", "相当"),
    # 机翻毛刺：叠字
    ("的的", "的"),
    ("了了", "了"),
    ("嗯嗯嗯嗯", "嗯嗯"),
]

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_HAS_CJK = re.compile(r"[%s]" % _CJK)

# CJK 与标点之间的空格（「大家 ，你好」→「大家，你好」）
# 必须在半角→全角之后跑，否则「伙计们! 我」里的半角 ! 不在字符类里，空格会留下。
_SP_CJK_PUNCT = re.compile(r"([%s，。！？、；：）」』]) +([%s，。！？、；：（「『])" % (_CJK, _CJK))
# 半角逗号/分号 → 全角（要求两侧是 CJK，避免动数字 1,000）；逗号后可带空格
_HALF_COMMA = re.compile(r"([%s]),\s*([%s])" % (_CJK, _CJK))
_HALF_SEMI = re.compile(r"([%s]);\s*([%s])" % (_CJK, _CJK))
# 句尾书面句号（全角/半角都可，可多次，只剥最外层一串）
_TRAIL_DOT = re.compile(r"[。．.]+\s*$")
# 连续句号/顿号压缩（！！/？？保留——那是情绪）
_DOT_RUN = re.compile(r"。{2,}")
_PAUSE_RUN = re.compile(r"、{2,}")


def _punct(text: str) -> str:
    """标点归一：半角→全角 → 清多余空格 → 压重复 → 去句尾句号。"""
    if not _HAS_CJK.search(text):
        return text
    # 顺序敏感：先全角化，空格清理的字符类才认得这些标点
    text = _HALF_COMMA.sub(r"\1，\2", text)
    text = _HALF_SEMI.sub(r"\1；\2", text)
    # 半角 !/? 统一全角（含 "?!!、?!" 这类混排尾巴；含 CJK 的句子才做）
    text = text.replace("!", "！").replace("?", "？")
    text = _SP_CJK_PUNCT.sub(r"\1\2", text)
    text = _DOT_RUN.sub("。", text)
    text = _PAUSE_RUN.sub("、", text)
    text = _TRAIL_DOT.sub("", text)
    return text


def zh_polish(text: str) -> str:
    """润色一条中文字幕。非中文文本原样返回。"""
    return polish_with_log(text)[0]


def polish_with_log(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """返回 (润色结果, [(规则前的片段, 规则后的片段)…])，供自测与日志核对。"""
    if not text or not _HAS_CJK.search(text):
        return text, []
    changes: List[Tuple[str, str]] = []
    out = text
    for pat, rep in _WORD_RULES:
        if pat in out:
            new = out.replace(pat, rep)
            if new != out:
                changes.append((pat, rep))
            out = new
    p = _punct(out)
    if p != out:
        changes.append(("标点/句尾", "归一"))
    return p, changes


# ---------------------------------------------------------------- 自测
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = [
        # (输入, 期望输出) —— 期望值来自 2026-10-13 腾讯通道实测译文
        ("谢谢你们，伙计们，我很高兴!", "谢谢你们，大家，我很高兴！"),
        ("今天，我想尝试一个新的游戏。", "今天，我想试试一个新的游戏"),
        ("让我先检查一下设置。", "让我先检查一下设置"),
        ("请不要担心。", "别担心"),
        ("这非常地厉害。", "这非常厉害"),
        ("大家 ，你好 吗?", "大家，你好吗？"),   # 空格清理 + 半角问号
        ("今天就到此为止吧，辛苦了。", "今天就到此为止吧，辛苦了"),
        ("真的假的?!!", None),              # ？！与情绪保留
        ("hello world", "hello world"),     # 非中文不动
        ("我看到 1,000 个人。", None),      # 数字逗号不能被改
        # —— 半角标点 + 空格（腾讯/MyMemory 常见形态，2026-10-13 补测）
        ("谢谢你们, 伙计们! 我很高兴.", "谢谢你们，大家！我很高兴"),
        ("好的; 我知道了", "好的；我知道了"),
        ("这是 3.14.", "这是 3.14"),        # 半角句尾句号也剥，小数点不受影响
    ]
    ok = total = 0
    for src, expect in cases:
        total += 1
        got, ch = polish_with_log(src)
        if expect is None:
            good = got != src or src == "hello world"
            print(f"{'PASS' if good else 'FAIL'} | {src} → {got}   {ch}")
        else:
            good = got == expect
            print(f"{'PASS' if good else 'FAIL'} | {src}\n       → {got}" +
                  ("" if good else f"   期望 {expect}") + f"   {ch}")
        ok += good
    print(f"\n{ok}/{total} 通过")
    sys.exit(0 if ok == total else 1)
