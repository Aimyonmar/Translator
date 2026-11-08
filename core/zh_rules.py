# -*- coding: utf-8 -*-
"""规则化译后处理（确定性、0ms、不联网）。

只做「机翻固定错法」的高置信度改写，全部规则可单独开关，默认全开
（`enable_zh_rules=true`）。规则来自 37 句诊断语料里反复出现的四类问题：

  1. tone       终助词/语气：よ/ね/よね/じゃん 丢失，「んだけど」被翻成「虽然…的，」；
  2. idiom      惯用句直译：「〜てくれてありがとう」→「谢谢你给我的人」；
  3. number     数词单位：円→日元、時間→小时、回→次；
  4. direction  授受方向：てあげる 被翻成「给我」、てくれる 被翻成「给你」；
  5. subject    省略主语：情绪形容词句没有主语时补「我」；
  6. contraction 口语缩略还原（送翻前）：っつー→という、すんの→するの。

安全边界：
- 只在源句命中明确语法标记时才动目标句，且目标句里必须出现对应的错误形态；
- 每条规则独立计数，`take_stats()` 可查到命中次数；
- `EXAMPLES` 保留最近 20 条「原句 / 改前 / 改后 / 规则」，误改时按规则名关掉即可回收；
- 本模块不做模糊匹配、不改词序、不增删实词（subject 规则只加一个「我」）。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter, deque
from typing import Deque, List, Tuple

__all__ = ["RULES", "parse_rules", "restore_contractions", "apply_rules",
           "take_stats", "recent", "EXAMPLES", "TRACE_PATH"]

RULES = ("tone", "idiom", "number", "direction", "subject", "contraction")
DEFAULT_RULES = ",".join(RULES)

# 最近 20 条改写落盘位置：进程重启后仍能在面板/文件里回溯「规则改了什么」
TRACE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          ".runtime", "rule_changes.json")

# 中文语气词：句子里已有任意一个就不再追加，避免「呢呢」「啊吧」
_MODAL = set("啊呀呢吧嘛哦啦哟咯")
_END = r"[。！？!?…，,、\s]*"
# 情绪形容词/动词：subject 规则只在这些词前补「我」
_EMOTION = ("怖", "寂", "嬉", "眠", "疲", "泣", "恥ずか", "つら", "しんど",
            "やば", "不安", "心配", "悲")
# 源句里已有明确主语时，不补「我」
_HAS_SUBJECT = re.compile(r"(私|僕|俺|あなた|君|皆さん|みんな|彼女|彼ら|彼)")
# 目标句里已有任何人称/群体，就不再补
_PRONOUN = ("我", "你", "他", "她", "它", "大家", "我们", "你们", "他们", "她们")

_STATS: Counter = Counter()
_STAT_LOCK = threading.Lock()
EXAMPLES: Deque[Tuple[str, str, str, str]] = deque(maxlen=20)


def parse_rules(raw: str) -> set:
    """解析 `"tone,idiom,-number"`；未知项忽略，`-x` 表示从全集中去掉。

    空串 = 一条都不开（安全默认）；只有调用方显式传 DEFAULT_RULES 才全开。
    """
    if raw is None or not str(raw).strip():
        return set()
    items = [s.strip().lower() for s in str(raw).replace("，", ",").split(",") if s.strip()]
    plus = {s for s in items if not s.startswith("-")}
    minus = {s[1:] for s in items if s.startswith("-")}
    if not plus:
        plus = set(RULES)
    return (plus & set(RULES)) - minus


def _hit(rule: str) -> None:
    with _STAT_LOCK:
        _STATS[rule] += 1


def _record(src: str, before: str, after: str, rule: str) -> None:
    with _STAT_LOCK:
        EXAMPLES.append((src, before, after, rule))
    # 落盘不能放在翻译热路径上：只打个标记，由后台线程合并成一次写
    _TRACE_EVENT.set()
    _ensure_trace_thread()


_TRACE_EVENT = threading.Event()
_TRACE_THREAD: List[threading.Thread] = []
_TRACE_LOCK = threading.Lock()


def _ensure_trace_thread() -> None:
    with _TRACE_LOCK:
        if _TRACE_THREAD and _TRACE_THREAD[0].is_alive():
            return
        t = threading.Thread(target=_trace_loop, name="zh_rules_trace", daemon=True)
        _TRACE_THREAD[:] = [t]
        t.start()


def _trace_loop() -> None:
    while True:
        _TRACE_EVENT.wait()
        _TRACE_EVENT.clear()
        # 合并连续命中：等 0.5 秒再写，一批字幕只产生一次磁盘写
        time.sleep(0.5)
        _persist()


def _persist() -> None:
    """把 EXAMPLES 落到 .runtime/rule_changes.json，重启后仍可回溯。"""
    with _STAT_LOCK:
        rows = [{"src": s, "before": b, "after": a, "rule": r}
                for s, b, a, r in EXAMPLES]
    try:
        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        tmp = TRACE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        os.replace(tmp, TRACE_PATH)
    except Exception:
        pass


def recent(limit: int = 20) -> List[dict]:
    """最近 limit 条改写（原句 / 改前 / 改后 / 规则），新的在后。"""
    with _STAT_LOCK:
        rows = list(EXAMPLES)
    return [{"src": s, "before": b, "after": a, "rule": r}
            for s, b, a, r in rows[-max(1, int(limit)):]]


def take_stats() -> dict:
    """取走并清零计数（供诊断脚本/日志展示）。"""
    with _STAT_LOCK:
        snap = dict(_STATS)
        _STATS.clear()
    return snap


def _has_modal(text: str) -> bool:
    return any(ch in text for ch in _MODAL)


def _append_modal(text: str, modal: str) -> Tuple[str, bool]:
    """在句尾标点之前插入语气词；已有语气词则不动。"""
    if _has_modal(text):
        return text, False
    m = re.search(_END + r"$", text)
    if not m:
        return text, False
    head, tail = text[:m.start()], text[m.start():]
    if not head:
        return text, False
    return head + modal + tail, True


# ---------------------------------------------------------------- 送翻前：缩略
_CONTRACTIONS = [
    ("っつー", "という"),
    ("すんの", "するの"),
    ("すんだけど", "するんだけど"),
    ("してっから", "しているから"),
    ("じゃねぇ", "じゃない"),
    ("じゃねえ", "じゃない"),
]


def restore_contractions(text: str) -> Tuple[str, List[str]]:
    """把歧义较小的口语缩约展开为书面形（与 ja_norm 互补，不重复）。"""
    if not text:
        return text, []
    out = text
    notes: List[str] = []
    for a, b in _CONTRACTIONS:
        if a in out:
            out = out.replace(a, b)
            notes.append(f"contraction:{a}→{b}")
            _hit("contraction")
    return out, notes


# ---------------------------------------------------------------- 译后规则
def _rule_tone(src: str, tgt: str) -> Tuple[str, bool]:
    """终助词/「んだけど」的语气回收。"""
    s = src.rstrip("。！？!?…，,、 \t\r\n")
    before = tgt

    # 1) んだけど / けど：机翻常译成「虽然…的，」或「…，但是」
    if re.search(r"(んだけど|んだけども|だけど|けど)$", s):
        if tgt.startswith("虽然"):
            tgt = tgt[2:]
            if tgt.startswith("，"):
                tgt = tgt[1:]
        m = re.search(r"的(" + _END + r")$", tgt)
        if m:
            tgt = tgt[:m.start()] + "啊" + m.group(1)
        # 「…，但是」「…，但」：收掉转折连词，句尾补语气
        tgt = re.sub(r"[，,]\s*(但是|但)$", "", tgt)
        tgt, _ = _append_modal(tgt, "啊")
    # 2) じゃん / じゃんね：反问语气
    elif re.search(r"じゃん(ね|か)?$", s):
        tgt, _ = _append_modal(tgt, "嘛")
    # 3) よね：偏确认，用「呢」
    elif re.search(r"よね$", s):
        tgt, _ = _append_modal(tgt, "呢")
    # 4) よ：告知/强调，用「哦」
    elif re.search(r"[^ゃゅょ]よ$", s):
        tgt, _ = _append_modal(tgt, "哦")
    # 5) ね：动作后「哦」，状态后「呢」
    elif re.search(r"ね$", s):
        modal = "哦" if re.search(r"(する|した|して|やる|やった|行く|来る|見る|食べる|寝る|帰る)ね$", s) else "呢"
        tgt, _ = _append_modal(tgt, modal)

    # 6) ましょうか / ましょう：建议语气，句尾补「吧」
    if re.search(r"(ましょうか|ましょう)$", s):
        tgt, _ = _append_modal(tgt, "吧")
    return tgt, tgt != before


def _rule_idiom(src: str, tgt: str) -> Tuple[str, bool]:
    """惯用句直译回正（只收录诊断语料里确实出现的形态）。"""
    before = tgt
    if "来てくれてありがとう" in src and "给我的人" in tgt:
        tgt = tgt.replace("给我的人", "来看我")
    if "ありがとう" in src and tgt.strip() in ("谢谢你给我的人", "谢谢你给我的人。"):
        tgt = "谢谢来看我"
    return tgt, tgt != before


def _rule_number(src: str, tgt: str) -> Tuple[str, bool]:
    """数词/单位：只在源句带数字+单位、且目标句用了错单位时替换。"""
    before = tgt
    if re.search(r"\d+\s*円", src):
        tgt = re.sub(r"(\d+)\s*圆", r"\1日元", tgt)
    if re.search(r"\d+\s*時間", src):
        tgt = re.sub(r"(\d+)\s*时间", r"\1小时", tgt)
    if re.search(r"\d+\s*回", src):
        tgt = re.sub(r"(\d+)\s*回(?!合|答|复|忆|来|去)", r"\1次", tgt)
    if re.search(r"\d+\s*分", src) and "分" in src:
        tgt = re.sub(r"(\d+)\s*分(?!钟|享|开|数|析)", r"\1分钟", tgt)
    if "ドル" in src:
        tgt = tgt.replace("美金", "美元")
    return tgt, tgt != before


def _rule_direction(src: str, tgt: str) -> Tuple[str, bool]:
    """授受动词方向：てあげる≠给我，てくれる/てもらう≠给你。"""
    # 「来てくれてありがとう」天然带「给我」，不是方向错误
    if "ありがとう" in src:
        return tgt, False
    before = tgt
    if re.search(r"てあげ(る|た|ます)", src) and "给我" in tgt:
        tgt = tgt.replace("给我", "给你")
    elif re.search(r"てくれ(る|た|て)|てもら(う|った)", src) and "给你" in tgt:
        tgt = tgt.replace("给你", "给我")
    return tgt, tgt != before


def _rule_subject(src: str, tgt: str) -> Tuple[str, bool]:
    """省略主语补全：仅限「无主语 + 情绪词 + 译文无人称」这一种情况。"""
    if _HAS_SUBJECT.search(src):
        return tgt, False
    if not any(w in src for w in _EMOTION):
        return tgt, False
    if any(p in tgt for p in _PRONOUN):
        return tgt, False
    if not tgt or not ("\u4e00" <= tgt[0] <= "\u9fff"):
        return tgt, False
    return "我" + tgt, True


def apply_rules(src: str, tgt: str, rules: str = DEFAULT_RULES
                ) -> Tuple[str, List[str]]:
    """对一条译文应用已启用的规则，返回 (新译文, 命中说明)；rules 为空则不动。"""
    if not src or not tgt:
        return tgt, []
    enabled = parse_rules(rules)
    out = tgt
    notes: List[str] = []
    for name, fn in (("tone", _rule_tone), ("idiom", _rule_idiom),
                     ("number", _rule_number), ("direction", _rule_direction),
                     ("subject", _rule_subject)):
        if name not in enabled:
            continue
        before = out
        out, changed = fn(src, out)
        if changed:
            notes.append(name)
            _hit(name)
            _record(src, before, out, name)
    return out, notes


# ---------------------------------------------------------------- 自测
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cases = [
        ("めっちゃ怖いんだけど、", "我超级害怕的，", "我超级害怕啊，"),
        ("泣きそうなんだけど、", "虽然我快哭出来了，", "我快哭出来了啊，"),
        ("泣きそうなんだけど。", "我好像快哭出来了", "我好像快哭出来了啊"),
        ("今日も配信を見に来てくれてありがとう。", "谢谢你给我的人", "谢谢你来看我"),
        ("はい、1000円です", "好的，1000圆", "好的，1000日元"),
        ("3時間待った", "等待了3时间", "等待了3小时"),
        ("2回やった", "做了2回", "做了2次"),
        ("教えてあげる", "我会教给我", "我会教给你"),
        ("寂しいな", "寂寞啊", "我寂寞啊"),
    ]
    bad = 0
    for src, tgt, want in cases:
        got, notes = apply_rules(src, tgt)
        ok = got == want
        bad += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} | {src} | {tgt} → {got}" +
              ("" if ok else f"  期望 {want}") + f"  {notes}")
    # 不该动的样例：已有语气词、已有单位、明确主语
    keep = [
        ("そうだよね。", "是啊，对吧？"),
        ("1000円です", "1000日元"),
        ("私は寂しい", "寂寞啊"),
    ]
    for src, tgt in keep:
        got, notes = apply_rules(src, tgt)
        print(f"{'OK ' if got == tgt else 'BAD'} 保持不变 {src} | {tgt} -> {got} {notes}")
        bad += 0 if got == tgt else 1
    print(f"\n失败 {bad} 条；统计 {take_stats()}")
    sys.exit(1 if bad else 0)
