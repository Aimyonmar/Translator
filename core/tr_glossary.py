# -*- coding: utf-8 -*-
"""翻译侧术语表：专有名词整词精确替换（送翻前下发 + 译文兜底）。

和仓库里已有的三套表分工，避免规则打架：

| 表 | 介入时机 | 匹配方式 | 改的是什么 |
| --- | --- | --- | --- |
| `word_map` | 识别后 | 精确替换 | 上屏的原文 |
| `name_roster` / namefix | 识别后 | 按读音覆盖变体 | 上屏的原文 |
| `enable_title_glossary` | **送翻前** | 精确替换（内置标题） | 只改送翻文本，原文不动 |
| 本表 | **送翻前 + 译后** | 整词精确匹配 | 送翻文本 / 最终译文 |

执行顺序固定为：标题术语表 → 本表（送翻）→ 机翻 → 本表（译文兜底）。
都是精确子串匹配，不做模糊、不做读音、不做大小写折叠，因此不会出现同一处被
两套规则重复替换：前一步替换成功后，后一步的原文关键词已经不在文本里了。

为什么不把「蒙汉世界」这类坏译法写进送翻表：机翻的输出形态不可预测，
只能在译后用**已知的坏译法**做定点回改，命中就改、不命中不动。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

# 诊断时实测被机翻糟蹋过的专名（改进记录第二十七节）。
# 只收「机翻几乎不可能自己译对」的作品名，控制规模：业务词交给用户在面板里自己填。
BUILTIN_TERMS: Tuple[Tuple[str, str], ...] = (
    # 送翻用的名字里**不能带全角冒号**：实测「怪物猎人：世界」会被机翻拆成
    # 「怪物猎人：好像世界」，反而更糟。冒号只在译后兜底时出现。
    ("モンハンワイルズ", "怪物猎人荒野"),
    ("モンハンワールズ", "怪物猎人世界"),
    ("ゼルダの伝説", "塞尔达传说"),
    ("エルデンリング", "艾尔登法环"),
    ("エルジェンリング", "艾尔登法环"),
    ("エルデンリングのDLC", "艾尔登法环的DLC"),
)

# 译文兜底：机翻对这些词给出的实测坏译法 → 正确中文名。
POST_FIX: Tuple[Tuple[str, str], ...] = (
    ("蒙汉世界", "怪物猎人：世界"),
    ("蒙汉威尔斯", "怪物猎人：世界"),
    ("塞尔达的传奇", "塞尔达传说"),
    ("埃尔金戒指", "艾尔登法环"),
    ("艾尔金戒指", "艾尔登法环"),
    ("艾尔登戒指", "艾尔登法环"),
    ("Elgenring", "艾尔登法环"),
)


def parse(raw: str, limit: int = 80) -> List[Tuple[str, str]]:
    """解析用户词表：多行「原文=译法」，支持全角＝，忽略空行与注释。"""
    items: List[Tuple[str, str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("=", "＝"):
            if sep in line:
                k, v = line.split(sep, 1)
                k, v = k.strip(), v.strip()
                if k and v and k != v:
                    items.append((k[:40], v[:40]))
                break
        if len(items) >= limit:
            break
    return items


def _merge(extra: str) -> List[Tuple[str, str]]:
    """内置表 + 用户表；用户表同名键优先（后者覆盖前者）。"""
    table = dict(BUILTIN_TERMS)
    for k, v in parse(extra):
        table[k] = v
    return sorted(table.items(), key=lambda kv: len(kv[0]), reverse=True)


def apply_source(text: str, extra: str = "") -> Tuple[str, List[str]]:
    """送翻前把日语专名换成人定的中文名，返回 (新文本, 说明)。

    长词优先，避免「エルデンリング」先被「エルデン」这类短词切走；
    译文已经在文本里时不重复插入。
    """
    if not text:
        return text, []
    out = text
    notes: List[str] = []
    for src, dst in _merge(extra):
        if src not in out or dst in out:
            continue
        out = out.replace(src, dst)
        notes.append(f"{src}→{dst}")
    return out, notes


def apply_target(text: str, extra: str = "") -> Tuple[str, List[str]]:
    """译文兜底：只回改内置表里的已知坏译法，命中才动。

    用户词表（原文=译法）是「日文→中文」，拿它反过来改中文会把正常译文改坏，
    所以这里刻意**不使用** extra。
    """
    if not text:
        return text, []
    out = text
    notes: List[str] = []
    for bad, good in sorted(POST_FIX, key=lambda kv: len(kv[0]), reverse=True):
        if bad in out and good not in out:
            out = out.replace(bad, good)
            notes.append(f"{bad}→{good}")
    return out, notes
