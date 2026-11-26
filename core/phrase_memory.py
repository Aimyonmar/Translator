# -*- coding: utf-8 -*-
"""会话级短语记忆：重复句与固定表达零延迟命中，同一集内保持译法一致。

作用域由调用方控制：
- 实时链路：每次「开始翻译」清空，同一场直播内复用；
- 文件转写：每次转写任务开始前清空，同一文件内复用。

只记「已经过内容校验和润色」的最终译文，命中时直接返回，
不碰网络、不进通道预算；键只含语言对 + 归一化原文，不含上下文，
避免同一句因前文不同而重复请求。
"""
from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict

# 归一化时只做「句末标点/空白/大小写」级别的收敛，绝不改动词句结构。
# 目的：同一句在断句、标点、全半角上有细微差异时仍能命中。
_TRAIL_PUNCT = re.compile(r"[\s\u3000。、，,．.！!？?…‥~〜\-—ー:：;；\"'“”‘’）」』】]+$")
_INNER_SPACE = re.compile(r"[\s\u3000]+")


def normalize(text: str) -> str:
    """原文归一化键：NFKC + 小写 + 去内部空白 + 去句末标点。"""
    s = unicodedata.normalize("NFKC", text or "").strip().lower()
    s = _INNER_SPACE.sub("", s)
    s = _TRAIL_PUNCT.sub("", s)
    return s


class PhraseMemory:
    """有界 LRU：键 (源语言, 目标语言, 归一化原文) → 最终译文。"""

    MAX = 2000

    def __init__(self) -> None:
        self._data: "OrderedDict[tuple, str]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.stores = 0
        self.name_hits = 0

    def get(self, text: str, source: str, target: str):
        key = (source, target, normalize(text))
        if not key[2]:
            return None
        with self._lock:
            out = self._data.get(key)
            if out is None:
                return None
            # 命中移到队尾：反复出现的口头禅不能被一次性新句挤出 LRU
            self._data.move_to_end(key)
            self.hits += 1
            if is_name_like(key[2]):
                self.name_hits += 1
            return out

    def put(self, text: str, out: str, source: str, target: str) -> None:
        key = (source, target, normalize(text))
        if not key[2] or not (out or "").strip():
            return
        with self._lock:
            self._data[key] = out.strip()
            self._data.move_to_end(key)
            while len(self._data) > self.MAX:
                self._data.popitem(last=False)
            self.stores += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._data), "hits": self.hits,
                    "stores": self.stores, "name_hits": self.name_hits}


def is_name_like(text: str) -> bool:
    """短称呼/专名：没有句末语气与标点、整体很短。

    这类句子在台词里常单独出现（「アリス」「Alice」），命中即保证同一集里
    译名写法完全一致；长句不做词级对齐，只保证整句重复时一致。
    """
    t = (text or "").strip()
    if not t or len(t) > 20:
        return False
    if any(ch in t for ch in "。！？!?，,；;：:"):
        return False
    return " " not in t.strip()
