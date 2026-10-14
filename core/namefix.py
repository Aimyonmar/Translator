# -*- coding: utf-8 -*-
"""专有名词「音近自动修正」：录一次正确写法，自动覆盖各种听错变体。

和 word_map（纠错替换表）的分工：
  - word_map：**精确替换**。把「某一次听错的具体字符串」写死改成正确写法。
    100% 确定、0ms，但听错的写法千变万化，维护量随「错法数量」线性增长。
  - namefix（本模块）：**音近匹配**。只录入**正确的人名**（含汉字的名字要给假名读音），
    程序把「读音相近」的纯假名片段自动归一成正确写法。
    维护量 ≈ 人名数（稳定），不随错法膨胀。

匹配两层，从严到宽：
  1. 读音键完全相同（仅ひらがな/カタカナ/長音ー/促音っ 等书写差异）→ 替换。
  2. 读音键长度相同（或差 1）且编辑距离很小（漏音/多音/浊音混淆）→ 替换。
     长度差 ≥2 一律不匹配 —— 这条是防止把后接助词（と/が/の）一起吞掉的护栏。

安全阀：
  - 只对**纯假名**片段做匹配（含汉字的片段无法可靠转音，不参与）；
  - 读音键长度 <4 的片段不参与匹配，避免「かな / それ」这类常用词被名字表误伤；
  - 命中常用词停用表（STOPWORDS）直接跳过；
  - 含汉字却没给读音的词条会被**跳过并记入 skipped**（无法取音，宁可不改也不错改）；
  - 每处替换都记进 changes，便于日志核对。

已知边界（实测）：读音请写**全名**（约 ≥3 拍）。若把很短的读音当词条（例：`さくらみこ=みこ`），
「みこ」会命中「みこし」这类同音长词并把「し」留下（实测 →「さくらみこし」）。
写全名读音（`さくらみこ`）或 `さくらみこ=さくらみこ` 不会有这个问题。

词表格式（一行一个，也可用逗号/顿号分隔，`#` 后注释）：
    さくらみこ                    # 名字本身是假名，直接写
    兎田ぺこら=うさだぺこら        # 含汉字：等号右边给假名读音，左边是上屏的正确写法
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

__all__ = ["phonetic_key", "NameCorrector", "apply_names", "parse_roster", "hotword_text"]

# ---------------------------------------------------------------- 假名 → 罗马字
# 已做「近音折叠」：し→si、ち→ti、つ→tu、じ→zi、ふ→hu（含 しゃ→sa、ちゃ→ta 等）
_SINGLE: Dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "si", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "zi", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "ti", "つ": "tu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "zi", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "hu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo",
    "ゔ": "bu",
    "っ": "", "ッ": "", "ー": "",   # 促音 / 长音符：丢掉（已折叠）
}

# 拗音・外来语合字（2 字符，已折叠：しゃ→sa、ちゃ→ta、ふぁ→ha）
_DIGRAPHS_HIRA: Dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sa", "しゅ": "su", "しょ": "so", "しぇ": "se",
    "じゃ": "za", "じゅ": "zu", "じょ": "zo", "じぇ": "ze",
    "ちゃ": "ta", "ちゅ": "tu", "ちょ": "to", "ちぇ": "te",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ふぁ": "ha", "ふぃ": "hi", "ふぇ": "he", "ふぉ": "ho", "ふゅ": "hyu",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "うぃ": "i", "うぇ": "e", "うぉ": "o",
    "すぃ": "si", "ずぃ": "zi",
}


def _to_katakana(hira: str) -> str:
    out = []
    for ch in hira:
        o = ord(ch)
        out.append(chr(o + 0x60) if 0x3041 <= o <= 0x3096 else ch)
    return "".join(out)


_DIGRAPHS: Dict[str, str] = dict(_DIGRAPHS_HIRA)
for _k, _v in _DIGRAPHS_HIRA.items():
    _DIGRAPHS[_to_katakana(_k)] = _v
_SINGLE.update({_to_katakana(k): v for k, v in list(_SINGLE.items()) if len(k) == 1})

_KANA_RUN = re.compile(r"[\u3041-\u3096\u30a1-\u30f6\u30fc]+")
_KANJI = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# 促音 / 長音：不参与取音（phonetic_key 里已丢），但**可能属于后面的词**。
# 「エルデンのリングって何？」里结尾的「っ」属于「って」，若吃进匹配窗口就会
# 变成「リングて何」——实测踩过，所以匹配窗口要先从右边剥掉这些无声字符。
_SILENT_TAIL = "っッー"

# 极常用的短假名词：绝不当作人名替换（防「かな / それ」这类被名字表误伤）
STOPWORDS = {
    "かな", "それ", "これ", "あれ", "どれ", "どこ", "いつ", "なに", "なん", "そう",
    "どう", "ここ", "そこ", "また", "まだ", "でも", "けど", "から", "まで", "だけ",
    "ので", "のに", "です", "ます", "した", "して", "する", "ある", "いる", "なる",
    "いい", "すご", "ちょ", "はい", "ええ", "うん", "もう", "やっ", "って",
    "わたし", "ぼく", "きみ", "みんな", "ほんと", "まじ", "やば", "もの",
    "こと", "とき", "ところ", "いま", "きょう", "あした", "きのう", "ください",
}


def phonetic_key(s: str) -> str:
    """把字符串折成「读音键」：ひらがな/カタカナ/長音/促音 差异全部抹平。

    非假名字符（拉丁字母）原样保留后统一小写，方便英文名也走同一套匹配。
    """
    if not s:
        return ""
    buf: List[str] = []
    i, n = 0, len(s)
    while i < n:
        if i + 1 < n:
            two = s[i:i + 2]
            hit = _DIGRAPHS.get(two)
            if hit is not None:
                buf.append(hit)
                i += 2
                continue
        ch = s[i]
        buf.append(_SINGLE.get(ch, ch))
        i += 1
    r = "".join(buf).lower()
    r = r.replace("l", "r")                      # 英文名 l/r 不分
    r = re.sub(r"[^a-z0-9]", "", r)
    r = re.sub(r"(.)\1+", r"\1", r)              # 连续同字符折叠：oo→o、tt→t
    return r


def _lev(a: str, b: str) -> int:
    """编辑距离（小字符串 DP，带长度差早退）。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cur = [j] + [0] * la
        bj = b[j - 1]
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost)
        prev = cur
    return prev[la]


class _Entry:
    __slots__ = ("display", "key", "klen")

    def __init__(self, display: str, reading: str) -> None:
        self.display = display
        self.key = phonetic_key(reading)
        self.klen = len(reading)      # 取音假名的字符数：用来限定滑窗宽度


class NameCorrector:
    """按词表把听错的纯假名片段归一成正确写法。只读，可多线程共用。"""

    #: 读音键最小长度：低于此值不参与匹配（约 2 拍以内太容易误伤）
    MIN_KEY = 4

    def __init__(self, raw: str) -> None:
        self.entries: List[_Entry] = []
        self.skipped: List[str] = []          # 含汉字却没给读音、无法取音的词条
        seen = set()
        for disp, reading in parse_roster(raw):
            if _KANJI.search(reading):
                self.skipped.append(disp)
                continue
            key = phonetic_key(reading)
            if not key or len(key) < self.MIN_KEY:
                self.skipped.append(disp)
                continue
            if (disp, key) in seen:
                continue
            seen.add((disp, key))
            self.entries.append(_Entry(disp, reading))

        # 长名字优先（避免短名字先抢走子串）
        self.entries.sort(key=lambda e: -e.klen)
        # 索引：key 完全相同 → O(1) 命中；否则只与「键长 ±1」的候选比编辑距离
        self._by_key: Dict[str, _Entry] = {}
        self._by_len: Dict[int, List[_Entry]] = {}
        for ent in self.entries:
            self._by_key.setdefault(ent.key, ent)
            self._by_len.setdefault(len(ent.key), []).append(ent)
        self._stopkeys = {phonetic_key(w) for w in STOPWORDS}
        # 需要试探的窗口宽度（字符数）：按各词条取音长度 ±2 去重
        self._widths = sorted({w for ent in self.entries
                               for w in range(max(2, ent.klen - 2), ent.klen + 3)})
        self.signature = raw or ""
        self._keycache: Dict[str, str] = {}

    # ------------------------------------------------------------ 内部
    def _key(self, sub: str) -> str:
        k = self._keycache.get(sub)
        if k is None:
            k = phonetic_key(sub)
            if len(self._keycache) > 8192:
                self._keycache.clear()
            self._keycache[sub] = k
        return k

    def _match(self, sub: str) -> Tuple[float, Optional[_Entry]]:
        """给一个假名片段，返回 (相似度, 词条)；相似度 0 表示不匹配。"""
        if sub in STOPWORDS:
            return 0.0, None
        k = self._key(sub)
        lk = len(k)
        if lk < self.MIN_KEY or k in self._stopkeys:
            return 0.0, None

        ent = self._by_key.get(k)                     # 第 1 层：读音键完全相同
        if ent is not None:
            return 1.0, ent

        # 第 2 层：键长相同（或差 1）且编辑距离很小；键长差 ≥2 直接拒绝，
        # 这条保证了「名字后面跟的助词（と/が/の）」不会被一起吞掉。
        best_s, best_e = 0.0, None
        for L in (lk - 1, lk, lk + 1):
            if L < self.MIN_KEY:
                continue
            for cand in self._by_len.get(L, ()):
                d = _lev(k, cand.key)
                allowed = (2 if (L == lk and max(lk, L) >= 8) else 1)
                if d > allowed:
                    continue
                s = 1.0 - d / max(lk, L)
                if s < 0.70:
                    continue
                if s > best_s:
                    best_s, best_e = s, cand
        return best_s, best_e

    def _fix_run(self, run: str, changes: List[Tuple[str, str]]) -> str:
        n = len(run)
        if n < 2 or not self._widths:
            return run
        widths = [w for w in self._widths if w <= n]
        out: List[str] = []
        i = 0
        while i < n:
            best: Optional[Tuple[float, int, _Entry]] = None
            tried = set()
            for w in widths:
                if i + w > n:
                    continue
                # 右边先剥掉促音/長音：它们不发音，落在词尾多半属于后一个词（って/ー）
                ww = w
                while ww > 1 and run[i + ww - 1] in _SILENT_TAIL:
                    ww -= 1
                if ww in tried:
                    continue
                tried.add(ww)
                s, e = self._match(run[i:i + ww])
                if s <= 0 or e is None:
                    continue
                if best is None or (s, ww) > (best[0], best[1]):
                    best = (s, ww, e)
            if best is None:
                out.append(run[i])
                i += 1
                continue
            _s, w, ent = best
            sub = run[i:i + w]
            if sub != ent.display:
                changes.append((sub, ent.display))
            out.append(ent.display)
            i += w
        return "".join(out)

    # ------------------------------------------------------------ 对外
    def fix(self, text: str) -> str:
        """返回修正后的文本。"""
        return self.fix_with_log(text)[0]

    def fix_with_log(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """返回 (修正后文本, [(原文片段, 替换成)…])，供日志核对。"""
        if not text or not self.entries:
            return text, []
        changes: List[Tuple[str, str]] = []
        out: List[str] = []
        last = 0
        for m in _KANA_RUN.finditer(text):
            s, e = m.start(), m.end()
            out.append(text[last:s])
            out.append(self._fix_run(text[s:e], changes))
            last = e
        out.append(text[last:])
        return "".join(out), changes


_CACHE: Dict[str, NameCorrector] = {}
_CACHE_MAX = 8


def _corrector(raw: str) -> NameCorrector:
    c = _CACHE.get(raw)
    if c is None:
        c = NameCorrector(raw)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[raw] = c
    return c


def parse_roster(raw: str) -> List[Tuple[str, str]]:
    """解析词表 → [(显示写法, 用于取音的假名)]。"""
    out: List[Tuple[str, str]] = []
    if not raw:
        return out
    text = re.sub(r"#.*", "", raw)
    for chunk in re.split(r"[\n,、;；]+", text):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            disp, reading = item.split("=", 1)
            disp, reading = disp.strip(), reading.strip()
        else:
            disp = reading = item
        if disp and reading:
            out.append((disp, reading))
    return out


def apply_names(text: str, raw: str) -> str:
    """便捷入口：按词表 raw 修正 text（词表不变时零重建开销）。"""
    if not text or not raw:
        return text
    return _corrector(raw).fix(text)


# ---------------------------------------------------------------- 内置标题
#: 常用游戏 / 番剧标题（日语写法）。含汉字的名词给了读音，可同时参与「音近修正」；
#: 片假名标题天然能参与音近修正（例：エルジンリング → エルデンリング）。
#: 顺序 = 常见度，配合 `hotword_text(limit=…)` 截断时优先保留靠前的。
BUILTIN_TITLES = """
黒神話：悟空=こくしんわごくう
原神=げんしん
エルデンリング
モンハンワイルズ
モンハンライズ=もんはんらいず
ゼルダの伝説=ぜるだのでんせつ
スプラトゥーン
マインクラフト
ポケットモンスター=ぽけっともんすたー
あつまれどうぶつの森=あつまれどうぶつのもり
大乱闘スマッシュブラザーズ=だいらんとうすまっしゅぶらざーず
マリオカート=まりおかーと
崩壊スターレイル=かいほうすたーれいる
ゼンレスゾーンゼロ=ぜんれすぞーんぜろ
鳴潮=めいちょう
ブルーアーカイブ=ぶるーあーかいぶ
プロジェクトセカイ=ぷろじぇくとせかい
ウマ娘=うまむすめ
雀魂=じゃんたま
エーペックスレジェンズ=えーぺっくすれじぇんず
ヴァロラント
オーバーウォッチ=おーばーうぉっち
フォートナイト
アモングアス
ドラゴンクエスト=どらごんくえすと
ファイナルファンタジー=ふぁいなるふぁんたじー
バイオハザード=ばいおはざーど
ペルソナ=ぺるそな
龍が如く=りゅうがごとく
ニーアオートマタ=にーあおーとまた
ホロウナイト
スターデューバレー=すたーでゅーばれー
呪術廻戦=じゅじゅつかいせん
鬼滅の刃=きめつのやいば
進撃の巨人=しんげきのきょじん
葬送のフリーレン=そうそうのふりーれん
ワンピース
ドラゴンボール=どらごんぼーる
名探偵コナン=めいたんていこなん
チェンソーマン
""".strip()


def hotword_text(raw: str, builtin: bool = True, limit: int = 80) -> str:
    """给 Whisper `hotwords=` 用的字符串。

    **必须用「展示写法」（汉字/片假名原样），不要用假名读音**——2026-10-13 实测
    （6 个标题 × 三种解码设置，见 `title_asr_test.py`）：

      | 解码设置 | 标题命中 | 平均 CER |
      | --- | --- | --- |
      | 裸识别 | 5/6 | 7.1% |
      | 热词=展示写法 | 5/6 | **4.7%** |
      | 热词=假名读音 | 3/6 | **26.1%** |

    假名热词会把解码风格带跑偏（「黒神話、悟空」→「黒髪話、ごくうが」，
    「スプラトゥーン」→「ぷらとぅーん」，极端情况整句空输出），**不要再用**。

    limit 为字符预算：越靠前的词条越优先（用户词表在前、内置标题在后），
    避免 prompt 过长被 faster-whisper 按 token 截断（上限约 224 token）。
    """
    items: List[str] = []
    seen = set()
    sources = [raw]
    if builtin:
        sources.append(BUILTIN_TITLES)
    for src in sources:
        for disp, _reading in parse_roster(src):
            if disp in seen:
                continue
            seen.add(disp)
            items.append(disp)
    out: List[str] = []
    used = 0
    for disp in items:
        add = len(disp) + (1 if out else 0)
        if used + add > limit:
            break
        out.append(disp)
        used += add
    return "、".join(out)


#: 内置标题的**中文名**。识别对了还要翻对：把日语标题直接换成中文名再送去翻译，
#: 机翻就不会乱翻（实测「原神の新しいイベント」被腾讯翻成「上帝的新活动」，
#: 「エルデンリングのDLC」→「埃尔金戒指DLC」——识别没错，是翻译层不认识游戏名）。
#: 键是日文写法（须与 BUILTIN_TITLES / 用户词表的 display 一致）。
TITLE_ZH: Dict[str, str] = {
    "黒神話：悟空": "黑神话：悟空",
    "原神": "原神",
    "エルデンリング": "艾尔登法环",
    "モンハンワイルズ": "怪物猎人：荒野",
    "モンハンライズ": "怪物猎人：崛起",
    "ゼルダの伝説": "塞尔达传说",
    "スプラトゥーン": "斯普拉遁",
    "マインクラフト": "我的世界",
    "ポケットモンスター": "宝可梦",
    "あつまれどうぶつの森": "集合啦！动物森友会",
    "大乱闘スマッシュブラザーズ": "任天堂明星大乱斗",
    "マリオカート": "马力欧卡丁车",
    "崩壊スターレイル": "崩坏：星穹铁道",
    "ゼンレスゾーンゼロ": "绝区零",
    "鳴潮": "鸣潮",
    "ブルーアーカイブ": "蔚蓝档案",
    "プロジェクトセカイ": "世界计划",
    "ウマ娘": "赛马娘",
    "雀魂": "雀魂",
    "エーペックスレジェンズ": "Apex英雄",
    "ヴァロラント": "无畏契约",
    "オーバーウォッチ": "守望先锋",
    "フォートナイト": "堡垒之夜",
    "アモングアス": "Among Us",
    "ドラゴンクエスト": "勇者斗恶龙",
    "ファイナルファンタジー": "最终幻想",
    "バイオハザード": "生化危机",
    "ペルソナ": "女神异闻录",
    "龍が如く": "如龙",
    "ニーアオートマタ": "尼尔：机械纪元",
    "ホロウナイト": "空洞骑士",
    "スターデューバレー": "星露谷物语",
    "呪術廻戦": "咒术回战",
    "鬼滅の刃": "鬼灭之刃",
    "進撃の巨人": "进击的巨人",
    "葬送のフリーレン": "葬送的芙莉莲",
    "ワンピース": "海贼王",
    "ドラゴンボール": "龙珠",
    "名探偵コナン": "名侦探柯南",
    "チェンソーマン": "电锯人",
}


def apply_title_glossary(text: str, raw: str = "", builtin: bool = True) -> str:
    """把识别文本里的日语标题换成中文名，**送翻译之前**用。

    上屏的「原文」仍用识别原样（调用方保留原文即可），这里只影响送去翻译的文本。
    用户词表里的「误=正」也会按同一套逻辑换（正=中文名时正好当术语表用）。
    """
    if not text:
        return text
    pairs: List[Tuple[str, str]] = []
    for src in ((raw, BUILTIN_TITLES) if builtin else (raw,)):
        for disp, _reading in parse_roster(src):
            zh = TITLE_ZH.get(disp)
            if zh and zh != disp:
                pairs.append((disp, zh))
    if not pairs:
        return text
    # 长标题优先，避免「モンハン」先于「モンハンワイルズ」抢先替换
    pairs.sort(key=lambda kv: -len(kv[0]))
    out = text
    for ja, zh in pairs:
        if ja in out:
            out = out.replace(ja, zh)
    return out


def prompt_echo(text: str, hotwords: str) -> bool:
    """识别结果是不是把热词表「复述」了出来（prompt 回声）。

    Whisper 的 prompt 通道有复述风险：整段输出若与热词表高度重合、且几乎不含
    其他内容，就当回声丢掉，宁可少一句也不要上屏一串词表。
    """
    if not text or not hotwords:
        return False
    words = [w for w in hotwords.replace("、", " ").split() if w]
    if not words:
        return False
    body = text
    for ch in " 、，。！？!?：:・「」『』":
        body = body.replace(ch, "")
    if not body:
        return True
    hit = 0
    for w in words:
        w2 = w.replace("：", "")
        if w2 and w2 in body:
            hit += len(w2)
    return hit / len(body) >= 0.8 and len(body) <= len(hotwords) * 1.5


# ---------------------------------------------------------------- 自测
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ROSTER = ("兎田ぺこら=うさだぺこら\n星街すいせい=ほしまちすいせい\nさくらみこ\n"
              "アキ・ローゼンタール\n白上フブキ=しらかみふぶき\n常闇トワ=とこやみとわ")
    c = NameCorrector(ROSTER)

    # (原文, 期望必须出现 / 期望必须原样不变)
    should_fix = [
        ("うさだぺこらが来た", "兎田ぺこらが来た"),
        ("ウサダペコーラが来た", "兎田ぺこらが来た"),
        ("うさたぺこらが来た", "兎田ぺこらが来た"),          # 浊音混淆
        ("ほしまちすいせいの歌", "星街すいせいの歌"),        # 助词「の」不能被吞
        ("ホシマチスイセイの歌", "星街すいせいの歌"),
        ("アキローゼンタールだよ", "アキ・ローゼンタールだよ"),
        ("うさだぺこらとほしまちすいせいが来た",
         "兎田ぺこらと星街すいせいが来た"),                  # 助词「と」「が」都不能被吞
    ]
    must_not_fix = [
        "かなりすごいと思った",
        "それはそうですね",
        "ちょっとまってください",
        "きょうはここまでにしておきます",
        "それでいいかな",
        "ありがとうございます",
        "おはようございます",
        "こんなことするわけがないじゃないか",
    ]

    ok = total = 0
    print("── 应修正（助词不得被吞） ──")
    for src, expect in should_fix:
        total += 1
        got, ch = c.fix_with_log(src)
        good = (got == expect)
        ok += good
        print(f"{'PASS' if good else 'FAIL'} | {src}\n       → {got}" + ("" if good else f"   期望 {expect}") )
    print("── 不应改动 ──")
    for src in must_not_fix:
        total += 1
        got, ch = c.fix_with_log(src)
        good = (got == src)
        ok += good
        print(f"{'PASS' if good else 'FAIL'} | {src} → {got}" + ("" if good else f"   (改动 {ch})"))

    # ── 内置标题的误伤扫描：40 个标题（含片假名，会参与音近匹配）对普通句的影响 ──
    print("── 内置标题误伤扫描 ──")
    builtin = NameCorrector(BUILTIN_TITLES)
    plain = [
        "こんにちは、みなさん！今日も元気に配信していきます",
        "えっと、まず最初に設定をちょっと確認しておきますね",
        "ありがとうございます、すごく嬉しいです",
        "ちょっと待ってください、今あとで説明します",
        "このゲームは本当に難しくて、何回も死んじゃいました",
        "じゃあ、そろそろ始めていきましょうか",
        "いやー、ほんとにびっくりしましたよ",
        "それでそれで、どうなったの？",
        "おつかれさまでした、また明日",
        "コメントありがとう、うれしいなあ",
        "すみません、ちょっと聞こえなかったです",
        "あ、そういえば昨日ね、こんなことがあって",
        "うんうん、わかるよ、その気持ち",
        "ちょっとトイレ行ってきます",
        "この曲いいよね、大好きなんだ",
        "むずかしいなあ、もう一回やってみよう",
        "なるほど、そういうことか",
        "まってまって、はやすぎるよ",
        "今日のごはんはカレーにしました",
        "ねむいなあ、何時までやってたっけ",
        "やったー、勝てたよ！",
        "ドラゴンみたいな敵が出てきた",
        "ペルソナみたいな雰囲気だね",      # 含「ペルソナ」但不该被改（本来就是这个写法）
        "エルデンのリングって何？",        # 近似词条：可以改，但**促音「っ」不得被吞**（见下）
        "あのさくらんぼ、おいしいよね",    # 含「さくら」但不在标题表
    ]
    hurt = []
    for src in plain:
        got = builtin.fix(src)
        if got != src:
            if src == "エルデンのリングって何？":
                # 这一句本来就判成「近似词条」，不算误伤——但必须保住「って」
                if "って" not in got:
                    hurt.append((src, got, "促音被吞"))
                continue
            hurt.append((src, got))
    total += 1
    ok += (not hurt)
    print(f"{'PASS' if not hurt else 'FAIL'} | {len(plain)} 句普通句 → 误伤 {len(hurt)} 处 "
          + ("" if not hurt else str(hurt)))

    print(f"\n{ok}/{total} 通过；因缺读音被跳过：{c.skipped}")
    sys.exit(0 if ok == total else 1)
