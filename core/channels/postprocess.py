# -*- coding: utf-8 -*-
"""译后处理与译文质量判定：口语化润色、规则改写、术语回改、垃圾译文识别。"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..zh_polish import zh_polish
from ..zh_rules import apply_rules as apply_zh_rules, format_hits as format_rule_hits

# 通道把「提示页/错误页」当译文返回时的特征。DeepL 旧版提示页是最典型的：
# HTTP 200 + 合法 JSON（result.texts[0].text 里就是 15 种语言的升级提示），
# 只靠长度/重复规则拦不住，必须按内容特征判。
_HTML_TAG = re.compile(
    r"<\s*/?\s*(html|head|body|div|span|p|br|script|style|a|h[1-6]|iframe|meta)\b[^>]*>",
    re.I)
_NOTICE_MARKERS = (
    "older version of deepl",
    "using an older version",
    "deepl.com/pro",
    "please update to the latest version",
    "please update the app",
    "bitte aktualisieren sie",
    "aktualisieren sie die app",
    "обновите",
    "mettre à jour",
    "actualice la aplicación",
    "atualize o aplicativo",
    "aggiorna l'app",
    "güncelleyin",
    "aktualizuj",
    "请更新到最新版本",
    "請更新到最新版本",
    "请更新至最新版本",
    "請更新至最新版本",
    "最新バージョンに更新",
    "최신 버전으로 업데이트",
)

_VERSION_WORDS = ("版本", "version", "バージョン", "버전", "versión", "versione",
                  "versie", "wersj", "verzi")
_UPDATE_WORDS = ("更新", "update", "upgrade", "アップデート", "업데이트",
                 "actualiz", "aggiorn", "aktualis", "güncelle", "atualiz",
                 "mettre à jour")


def _cjk_count(s: str) -> int:
    return sum(1 for ch in s
               if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")


def _letter_count(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha())


def invalid_reason(text: str, out: str, source: str = "ja",
                   target: str = "zh-Hans") -> str:
    """译文响应的来源格式/内容校验：命中返回原因，空串表示可用。

    专门拦「HTTP 200 + 合法 JSON，但正文是翻译器自己的提示页」这类污染，
    例如 DeepL 旧版停用提示（15 种语言重复段落）。命中后调用方必须换通道，
    绝不能上屏。
    """
    if not out or not out.strip():
        return "空译文"
    s = out.strip()
    low = s.lower()

    # 1) HTML 片段 / 错误页
    if _HTML_TAG.search(s) or "<!doctype" in low or "&lt;html" in low:
        return "HTML 片段"

    # 2) 翻译器专属提示语（DeepL 旧版提示等），大小写不敏感
    if any(m in low for m in _NOTICE_MARKERS):
        return "翻译器提示页"
    # 同一产品名在正文里出现 2 次以上，几乎只可能是「多语言提示块」
    if low.count("deepl") >= 2:
        return "多语言提示块"
    # 产品名 + 「版本」+「更新」同时出现：就是提示页的单语种片段
    if ("deepl" in low
            and any(w in low for w in _VERSION_WORDS)
            and any(w in low for w in _UPDATE_WORDS)):
        return "翻译器提示页"

    # 3) 多语言重复块：单句译文不会是 4 段以上、段段不同的长文
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    if len(paras) >= 4 and len(s) > 120:
        return "多段重复块"
    if len(paras) >= 3 and len(set(paras)) == 1:
        return "重复段落"

    # 4) 长度比例异常：单句字幕译文不可能比原文长一个数量级
    if len(s) > max(160, len(text) * 8):
        return "长度比例异常"

    # 5) 目标语言明显不符：按目标语种检查文字系统。
    #    纯数字/符号（"123"）不判语言——各通道原样回传是正常的。
    letters = _letter_count(s)
    if target.startswith("zh"):
        if letters >= 4 and not any("\u4e00" <= ch <= "\u9fff" for ch in s):
            return "语言不符"
    elif target == "en":
        cjk = _cjk_count(s)
        if letters >= 4 and cjk / len(s) > 0.5:
            return "语言不符"
    elif target == "ko":
        if letters >= 4 and not any("\uac00" <= ch <= "\ud7af" for ch in s):
            return "语言不符"
    elif target == "ja":
        if letters >= 4 and not _cjk_count(s):
            return "语言不符"
    return ""


def is_invalid(text: str, out: str, source: str = "ja",
               target: str = "zh-Hans") -> bool:
    """invalid_reason 的布尔包装，供竞速 accept 回调直接使用。"""
    return bool(invalid_reason(text, out, source, target))



def polish(cfg, text: str, out: str, say) -> Tuple[str, List[str]]:
    """润色 + 规则改写 + 术语回改，返回 (译文, 命中的规则列表)。

    zh_polish / zh_rules / 术语回改都只面向中文译文。目标语言不是中文
    （如中文音频配英文译文）时全部跳过，否则中文规则会改写英文输出。
    """
    target = str(getattr(cfg, "target_lang", "zh-Hans") or "zh-Hans")
    if not target.startswith("zh"):
        return out, []
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


def is_junk(src: str, out: str, source: str = "ja") -> bool:
    """垃圾译文识别，命中就丢弃（只显示原文），不换通道重试：
    1. 原文很短、译文却长到离谱 —— 免费通道返回的拼凑乱码。
    2. 译文是同一个短片段重复 4 次以上 —— 实测 MyMemory 把「やばいやばいやばい」
       翻成「狗屎狗屎狗屎狗屎狗屎狗屎狗屎」。注意阈值取 4：彩云给的
       「糟糕糟糕糟糕」是正确译文，只有 3 次重复，不能误杀。
    """
    if source == "ja" and len(src) <= 12 and len(out) > max(20, len(src) * 4):
        return True
    n = len(out)
    for unit in (1, 2, 3):
        if n >= unit * 4 and n % unit == 0:
            piece = out[:unit]
            if out == piece * (n // unit) and n // unit >= 4:
                return True
    return False
