# -*- coding: utf-8 -*-
"""译后处理与译文质量判定：口语化润色、规则改写、术语回改、垃圾译文识别。"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..zh_polish import zh_polish
from ..zh_rules import apply_rules as apply_zh_rules, format_hits as format_rule_hits


def polish(cfg, text: str, out: str, say) -> Tuple[str, List[str]]:
    """润色 + 规则改写 + 术语回改，返回 (译文, 命中的规则列表)。"""
    # 译文口语化润色（确定性规则、0ms）：免费通道的固定生硬模式
    # （伙计们→大家、您→你、想尝试→想试试、半角标点、句尾书面句号）。
    # 规则只动词级与标点，绝不动句法——润色失败顶多不自然，不引入新错误。
    if getattr(cfg, "enable_zh_polish", True):
        out = zh_polish(out)
    # 规则化译后处理（默认开）：终助词语气、惯用句、数词单位、授受方向、主语。
    # 放在 zh_polish 之后，避免标点归一化把刚补的语气词再改一遍。
    hits: List[str] = []
    if getattr(cfg, "enable_zh_rules", True):
        before_rules = out
        out, hits = apply_zh_rules(
            text, out, getattr(cfg, "zh_rules", "") or "")
        if hits:
            # 日志带规则 id：一条错改能直接定位到「关掉哪个 id」
            say("规则[%s] %s → %s"
                % (",".join(format_rule_hits(hits)), before_rules, out))
    # 术语表译文兜底：只回改诊断里实测过的坏译法（蒙汉世界→怪物猎人：世界），
    # 命中才动；用户词表不参与译后替换，避免把正常中文改回日文。
    if getattr(cfg, "enable_translate_glossary", False):
        try:
            from ..tr_glossary import apply_target
            out, t_notes = apply_target(out)
            for n in t_notes:
                say(f"术语回改：{n}")
        except Exception:
            pass
    return out, hits


def split_free_segment(out: str) -> Optional[str]:
    """从「前缀+当前句」的译文里取出当前句：只认换行分段。

    免费通道没有上下文参数，只能把前文拼在待翻译文本前面；译文一般会保留
    换行，取最后一段就是当前句。切不开（只有一段）返回 None，交给调用方
    判定为污染并回退单句——绝不猜句子边界，猜错就会把上一句译文顶上来。
    """
    if not out:
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    return lines[-1]


def is_junk(src: str, out: str) -> bool:
    """垃圾译文识别，命中就丢弃（只显示原文），不换通道重试：
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
