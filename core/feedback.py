# -*- coding: utf-8 -*-
"""人工标错：把「这句译得不对」变成可统计的 jsonl 数据。

设计边界：
- 写盘只走后台线程，submit() 只做一次 Queue.put_nowait，实时链路里没有同步 IO；
- 队列满、序列化失败、磁盘异常一律吞掉并计数，绝不把异常抛回字幕线程；
- 每条在后台线程里 flush + os.fsync 后才算落盘，进程被强杀不会丢已落盘的条目
  （仍在队列里、还没写的条目会丢，这是「不阻塞实时链路」的代价）；
- 最近 20 句快照只存内存（约 20KB），供快捷键在字幕滚走后仍能标记。

统计命令：
    .runtime\\venv\\Scripts\\python.exe -m core.feedback --stats
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import threading
import time
from collections import Counter, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

__all__ = [
    "FEEDBACK_PATH", "RECENT_MAX", "TAGS", "TAG_NAMES", "new_session_id",
    "RecentSentences", "FeedbackWriter", "load_entries", "summarize",
    "config_snapshot", "model_snapshot", "make_entry",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(ROOT, ".runtime")
FEEDBACK_PATH = os.path.join(RUNTIME_DIR, "feedback.jsonl")

# 最近 20 句：按直播里 3~4 秒/句，约等于 60~80 秒回溯窗口。
# 用户通常是在译文上屏后一两句才发现译错，20 句足够兜住；每条约 0.5~1KB，
# 内存上限 <20KB，不会随直播时长增长。
RECENT_MAX = 20

# 错误类型按「能区分、能统计、能指导修复」来定：
# 识别错 → 查 ASR/热词；漏译 → 查断句/队列丢弃；错译/语气/术语 → 查翻译通道与规则。
TAGS = (
    ("asr", "识别错"),
    ("omission", "漏译"),
    ("mistranslation", "错译"),
    ("tone", "语气不对"),
    ("terminology", "术语错"),
    ("other", "其他"),
)
TAG_NAMES = dict(TAGS)

# 轮转阈值：8MB ≈ 8000 条（按每条约 1KB）。每天标 15 次可用一年多；
# 超过后把当前文件改名为 feedback.jsonl.1（只留一份备份），避免无限增长。
MAX_BYTES = 8 * 1024 * 1024


def new_session_id(kind: str = "live") -> str:
    """一次运行/一次转写一个 id，便于把同一句的多次标错聚合到一起。"""
    return "%s-%s-%s" % (kind, time.strftime("%Y%m%d-%H%M%S"), secrets.token_hex(2))


class RecentSentences:
    """最近 N 句的原文/译文/通道快照，线程安全，只存内存。

    实时链路每句只做一次 deque.append（微秒级）；关闭标错时 pipeline 不会调用，
    所以关闭状态下的开销是 0。
    """

    def __init__(self, maxlen: int = RECENT_MAX):
        self._items: Deque[dict] = deque(maxlen=max(1, int(maxlen)))
        self._lock = threading.Lock()

    def add_text(self, seq: int, src: str, ts: Optional[float] = None) -> None:
        if not src:
            return
        row = {
            "seq": int(seq),
            "src": src,
            "tgt": "",
            "channel": "",
            "ts_src": float(ts if ts is not None else time.time()),
            "ts_tgt": None,
        }
        with self._lock:
            self._items.append(row)

    def set_result(self, seq: int, src: str, tgt: str, channel: str,
                   ts: Optional[float] = None) -> None:
        with self._lock:
            for row in reversed(self._items):
                if row["seq"] == int(seq):
                    if src:
                        row["src"] = src
                    row["tgt"] = tgt or ""
                    row["channel"] = channel or ""
                    row["ts_tgt"] = float(ts if ts is not None else time.time())
                    return

    def get(self, back: int = 0) -> Optional[dict]:
        """back=0 表示最近一句，1 表示倒数第二句，以此类推。"""
        with self._lock:
            n = len(self._items)
            idx = n - 1 - max(0, int(back))
            if idx < 0 or idx >= n:
                return None
            return dict(self._items[idx])

    def snapshot(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._items]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class FeedbackWriter:
    """追加写 jsonl 的后台写手；submit() 不阻塞、不抛异常。"""

    def __init__(self, path: Optional[str] = None, max_queue: int = 256,
                 max_bytes: int = MAX_BYTES):
        self.path = path or os.environ.get("TRANSLATOR_FEEDBACK_PATH") or FEEDBACK_PATH
        self.max_bytes = int(max_bytes)
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=max(8, int(max_queue)))
        self._lock = threading.Lock()
        self._stats: Counter = Counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="feedback-writer")
        self._thread.start()

    # ---------------------------------------------------------- 对外
    def submit(self, entry: Dict) -> bool:
        """入队即返回。返回 True 只代表「已受理」，落盘由后台线程完成。"""
        try:
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            self._bump("failed")
            return False
        try:
            self._q.put_nowait(line)
        except queue.Full:
            self._bump("dropped")
            return False
        except Exception:
            self._bump("failed")
            return False
        self._bump("queued")
        return True

    def flush(self, timeout: float = 2.0) -> bool:
        """等待队列写完（测试/退出收尾用，界面线程别在按键路径调用）。"""
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._q.unfinished_tasks == 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def close(self, timeout: float = 2.0) -> None:
        self.flush(timeout)
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    # ---------------------------------------------------------- 后台
    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._stats[key] += n

    def _loop(self) -> None:
        while True:
            try:
                first = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            batch = [first]
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._append(batch)
                self._bump("written", len(batch))
            except Exception:
                self._bump("failed", len(batch))
            finally:
                for _ in batch:
                    try:
                        self._q.task_done()
                    except Exception:
                        pass

    def _append(self, batch: List[str]) -> None:
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        payload = "".join(line + "\n" for line in batch)
        self._rotate_if_needed(len(payload.encode("utf-8")))
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size <= 0 or size + incoming <= self.max_bytes:
            return
        try:
            os.replace(self.path, self.path + ".1")
        except Exception:
            pass


# ---------------------------------------------------------------- 快照
def config_snapshot(cfg) -> Dict:
    """标记时刻的关键配置；不写 Key/token 等敏感字段。"""
    def g(name, default=None):
        return getattr(cfg, name, default)

    return {
        "source_lang": g("source_lang", ""),
        "target_lang": g("target_lang", ""),
        "channel": g("channel", "auto"),
        "model_size": g("model_size", ""),
        "beam_size": g("beam_size", 0),
        "enable_hedge": bool(g("enable_hedge", False)),
        "hedge_backup_ms": g("hedge_backup_ms", 0),
        "enable_context": bool(g("enable_context", False)),
        "context_window": g("context_window", 0),
        "enable_free_context": bool(g("enable_free_context", False)),
        "enable_translate_glossary": bool(g("enable_translate_glossary", False)),
        "enable_title_glossary": bool(g("enable_title_glossary", False)),
        "enable_homophone_fix": bool(g("enable_homophone_fix", False)),
        "enable_zh_rules": bool(g("enable_zh_rules", False)),
        "enable_zh_polish": bool(g("enable_zh_polish", False)),
        "enable_llm_quality": bool(g("enable_llm_quality", False)),
        "llm_provider": g("llm_provider", ""),
        "llm_model": g("llm_model", ""),
    }


def model_snapshot(cfg, pipeline=None) -> Dict:
    asr = getattr(pipeline, "asr", None) if pipeline is not None else None
    loaded = getattr(asr, "_loaded_size", None) or getattr(asr, "model_size", "")
    return {
        "asr": loaded or getattr(cfg, "model_size", ""),
        "asr_device": getattr(asr, "dev_info", "") or getattr(cfg, "asr_device", ""),
        "compute_type": getattr(cfg, "compute_type", ""),
        "llm": getattr(cfg, "llm_model", "") if getattr(cfg, "enable_llm_quality", False) else "",
    }


def make_entry(*, cfg, session_id: str, source: str, seq: int, back: int,
               src: str, tgt: str, channel: str, tags: List[str],
               start_s: Optional[float] = None, end_s: Optional[float] = None,
               has_audio: bool = False, media: str = "",
               pipeline=None, note: str = "") -> Dict:
    now = time.time()
    tags = [t for t in tags if t in TAG_NAMES] or ["other"]
    return {
        "ts": datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds"),
        "ts_unix": round(now, 3),
        "session_id": session_id,
        "source": source,                 # live | file
        "seq": int(seq),
        "back": int(back),                # 标记时该句是倒数第几句（0=最近）
        "start_s": None if start_s is None else round(float(start_s), 3),
        "end_s": None if end_s is None else round(float(end_s), 3),
        "src": src or "",
        "tgt": tgt or "",
        "channel": channel or "",
        "channel_name": CHANNEL_NAMES.get(channel, channel) if channel else "",
        "tags": tags,
        "model": model_snapshot(cfg, pipeline),
        "config": config_snapshot(cfg),
        # 实时流不保留音频；文件转写保留源文件路径，才有转成评测用例的可能
        "audio": {"has_audio": bool(has_audio), "media": media or ""},
        "note": note,
    }


try:
    from .translator import CHANNEL_NAMES
except Exception:       # 诊断脚本单独加载本模块时也不能炸
    CHANNEL_NAMES = {}


# ---------------------------------------------------------------- 统计
def load_entries(path: Optional[str] = None) -> List[dict]:
    path = path or os.environ.get("TRANSLATOR_FEEDBACK_PATH") or FEEDBACK_PATH
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize(entries: List[dict]) -> Dict:
    by_tag: Counter = Counter()
    by_channel: Counter = Counter()
    by_model: Counter = Counter()
    by_sentence: Dict[str, dict] = {}
    for e in entries:
        tags = [t for t in (e.get("tags") or []) if t]
        for t in tags:
            by_tag[TAG_NAMES.get(t, t)] += 1
        ch = e.get("channel") or "(未知)"
        by_channel[e.get("channel_name") or ch] += 1
        model = (e.get("model") or {}).get("asr") or "(未知)"
        by_model[model] += 1
        key = "%s#%s" % (e.get("session_id", ""), e.get("seq", ""))
        slot = by_sentence.setdefault(key, {
            "key": key, "count": 0, "src": e.get("src", ""),
            "tgt": e.get("tgt", ""), "tags": Counter(),
        })
        slot["count"] += 1
        for t in tags:
            slot["tags"][TAG_NAMES.get(t, t)] += 1
    sentences = sorted(by_sentence.values(), key=lambda s: (-s["count"], s["key"]))
    return {
        "total": len(entries),
        "by_tag": dict(by_tag.most_common()),
        "by_channel": dict(by_channel.most_common()),
        "by_model": dict(by_model.most_common()),
        "by_sentence": [
            {"key": s["key"], "count": s["count"], "src": s["src"],
             "tgt": s["tgt"], "tags": dict(s["tags"].most_common())}
            for s in sentences
        ],
    }


def _print_stats(rows: List[dict], as_json: bool = False) -> None:
    s = summarize(rows)
    if as_json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return
    print("标错条目：%d 条" % s["total"])
    print("\n按错误类型（可多选，合计可能大于条目数）")
    for k, v in s["by_tag"].items():
        print("  %-8s %d" % (k, v))
    print("\n按翻译通道")
    for k, v in (s["by_channel"].items() or [("(无)", 0)]):
        print("  %-14s %d" % (k, v))
    print("\n按识别模型")
    for k, v in (s["by_model"].items() or [("(无)", 0)]):
        print("  %-14s %d" % (k, v))
    print("\n同句重复标错（次数降序）")
    for row in s["by_sentence"][:20]:
        tags = ",".join(row["tags"].keys())
        print("  %-28s %dx  [%s]  %s" % (row["key"], row["count"], tags, row["src"][:30]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="标错记录的读取与统计")
    ap.add_argument("--stats", action="store_true", help="输出统计（默认）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出统计结果")
    ap.add_argument("--path", default="", help="覆盖 jsonl 路径（默认 .runtime/feedback.jsonl）")
    args = ap.parse_args(argv)
    path = args.path or FEEDBACK_PATH
    rows = load_entries(path)
    if not args.stats and not args.json:
        print("文件：%s" % path)
        if not rows:
            print("还没有标错记录。")
            return 0
    _print_stats(rows, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
