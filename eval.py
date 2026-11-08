# -*- coding: utf-8 -*-
"""固定语料评测：录制 / 回放 / 基线回归。

评测要能重复，先把两个不确定源钉死：

1. 识别侧：录下「解码器原始输出」，回放时不再加载模型，按音频哈希原样返回；
   词表 / 错词表 / 人名纠错仍在回放后执行，所以纠错类开关的 A/B 依旧有效。
2. 翻译侧：录下每个通道的原始响应（通道名、请求文本、响应文本、耗时、异常），
   回放时不再发任何网络请求；缺键直接报错，绝不静默回落到网络。

用法：
  .runtime\\venv\\Scripts\\python.exe eval.py --build-corpus
  .runtime\\venv\\Scripts\\python.exe eval.py --record --tag v1 --save-baseline
  .runtime\\venv\\Scripts\\python.exe eval.py --replay --tag v1
  .runtime\\venv\\Scripts\\python.exe eval.py --replay --tag v1 --ablate zh_rules

参数说明：
  --record      走真实识别 + 真实网络，把响应写入 eval/recordings/<tag>/
  --replay      只读录制，不加载模型、不发网络
  --ablate      逗号分隔：zh_rules / glossary / titles / asrfix / phrases /
                title_glossary / zh_polish。录制与回放的 ablation 必须一致
  --set k=v     配置覆盖（仅本进程，不写回 config.json）
  --save-baseline  把本次结果写成 eval/baseline.json
  --no-compare  不与基线比较
  --force       覆盖已有录制

退出码：0 = 与基线一致；1 = 有差异；2 = 回放缺失 / 运行失败。
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SR = 16000
EVAL_DIR = os.path.join(ROOT, "eval")
CORPUS_PATH = os.path.join(EVAL_DIR, "corpus.json")
AUDIO_DIR = os.path.join(EVAL_DIR, "audio")
REC_DIR = os.path.join(EVAL_DIR, "recordings")
REPORT_DIR = os.path.join(EVAL_DIR, "reports")
BASELINE_PATH = os.path.join(EVAL_DIR, "baseline.json")

# 与 bench_pipeline 相同的节奏：正常停顿 / 快语速 / 抢话轮换
GAPS = [0.9, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 1.2]


# ---------------------------------------------------------------- 语料
def _corpus_texts() -> list:
    """37 句：沿用 `.runtime/bench_results/tr_diag.json` 的固定文本（标题/普通句/碎片句）。"""
    path = os.path.join(ROOT, ".runtime", "bench_results", "tr_diag.json")
    if not os.path.exists(path):
        raise SystemExit("缺少 37 句语料来源：%s" % path)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [r["src"] for r in rows]


def _corrected(text: str) -> str:
    """语料的「意图文本」：把项目里已人工确认的错词/标题纠正应用上去。

    音频仍是用历史 ASR 文本合成的（错的就真读错），参考文本写正确写法，
    CER 才能区分「修好了」和「没修」。这不是循环论证：纠正条目来自
    改进记录里逐条人工确认的结果，不是用当前输出反推的。
    """
    from core.asr import WhisperASR
    from core.namefix import apply_names, merge_roster

    fixed = WhisperASR.apply_builtin_fixes(text)
    return apply_names(fixed, merge_roster(""))


def build_corpus() -> dict:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    from latency_test import tts

    items = []
    for i, text in enumerate(_corpus_texts()):
        rel = "audio/%03d.mp3" % i
        dst = os.path.join(EVAL_DIR, rel)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
            shutil.copyfile(tts(text), dst)
        items.append({"i": i, "text": text, "ref": _corrected(text), "file": rel})
    corpus = {"version": 2, "sr": SR, "gaps": GAPS, "items": items}
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return corpus


def load_corpus() -> dict:
    if not os.path.exists(CORPUS_PATH):
        raise SystemExit("缺少 eval/corpus.json，先运行：eval.py --build-corpus")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def corpus_sha1(corpus: dict) -> str:
    raw = json.dumps([it.get("ref") or it["text"] for it in corpus["items"]],
                     ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_material(corpus: dict) -> np.ndarray:
    from core.transcribe import decode_audio

    parts = []
    for i, it in enumerate(corpus["items"]):
        audio = decode_audio(os.path.join(EVAL_DIR, it["file"]))
        parts.append(audio.astype(np.float32))
        parts.append(np.zeros(int(corpus["gaps"][i % len(corpus["gaps"])] * SR),
                              dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------- 录制/回放
def audio_hash(audio: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    return hashlib.sha1(a.tobytes()).hexdigest()[:20]


class Recorder:
    """把识别原始输出与通道响应落成 jsonl；回放时按键查表。"""

    def __init__(self, tag: str, ablate: list):
        self.tag = tag
        self.ablate = ablate
        self.dir = os.path.join(REC_DIR, tag)
        self.asr_path = os.path.join(self.dir, "asr.jsonl")
        self.tr_path = os.path.join(
            self.dir, "tr%s.jsonl" % (("__" + "-".join(ablate)) if ablate else ""))
        self.misses: list = []
        self.asr_rows: dict = {}
        self.tr_rows: list = []
        self._tr_index: dict = {}
        self._tr_cursor: dict = {}
        self._lock = threading.Lock()
        self._seq = 0

    # -------- 写
    def add_asr(self, h: str, raw: str, audio_s: float, ms: float) -> None:
        with self._lock:
            if h not in self.asr_rows:
                self.asr_rows[h] = {"h": h, "raw": raw or "", "audio_s": round(audio_s, 3),
                                    "ms": round(ms, 1)}
            self._dirty_asr = True

    def add_tr(self, ch: str, text: str, src: str, out, err: str, ms: float) -> None:
        with self._lock:
            self._seq += 1
            self.tr_rows.append({"seq": self._seq, "ch": ch, "text": text, "src": src,
                                 "out": out, "err": err, "ms": round(ms, 1)})

    def flush(self, meta: dict) -> None:
        os.makedirs(self.dir, exist_ok=True)
        if getattr(self, "_dirty_asr", False):
            with open(self.asr_path, "w", encoding="utf-8") as f:
                for row in self.asr_rows.values():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.tr_path, "w", encoding="utf-8") as f:
            for row in self.tr_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta_path = os.path.join(
            self.dir, "meta%s.json" % (("__" + "-".join(self.ablate)) if self.ablate else ""))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
            f.write("\n")

    # -------- 读
    def load_asr(self) -> None:
        if not os.path.exists(self.asr_path):
            raise SystemExit("回放缺少 %s（先 --record --tag %s）" % (self.asr_path, self.tag))
        with open(self.asr_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    self.asr_rows[row["h"]] = row

    def load_tr(self) -> None:
        """回放时优先用基线录制。

        带 ablation 的录制只用来补「请求文本变了」的新键：同一句同样的请求，
        必须回放基线那一次的真实响应，否则网络抖动会混进 A/B 结论。
        """
        base_path = os.path.join(self.dir, "tr.jsonl")
        paths = [base_path]
        if os.path.abspath(self.tr_path) != os.path.abspath(base_path):
            paths.append(self.tr_path)
        for p in paths:
            if not os.path.exists(p):
                if p == base_path:
                    raise SystemExit("回放缺少 %s（先 --record --tag %s）" % (p, self.tag))
                raise SystemExit("回放缺少 %s（先 --record --tag %s%s）"
                                 % (p, self.tag,
                                    (" --ablate " + ",".join(self.ablate)) if self.ablate else ""))
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = (row["ch"], row["text"], row["src"])
                    # 基线已有的键不再接收 ablation 录制里的同名请求
                    if key in self._tr_index:
                        continue
                    self.tr_rows.append(row)
                    self._tr_index.setdefault(key, []).append(row)
        for lst in self._tr_index.values():
            lst.sort(key=lambda r: r["seq"])

    def lookup_asr(self, h: str):
        row = self.asr_rows.get(h)
        if row is None:
            with self._lock:
                self.misses.append("识别回放缺失：hash=%s" % h)
            return None
        return row

    def lookup_tr(self, ch: str, text: str, src: str):
        key = (ch, text, src)
        lst = self._tr_index.get(key)
        if not lst:
            with self._lock:
                self.misses.append("翻译回放缺失：ch=%s src=%s text=%r"
                                   % (ch, src, text[:48]))
            return None
        i = self._tr_cursor.get(key, 0)
        self._tr_cursor[key] = i + 1
        return lst[min(i, len(lst) - 1)]


class FakeCapture:
    """按音频时间轴吐块；回放可关闭节奏，纯按样本推进。"""

    def __init__(self, audio: np.ndarray, paced: bool, chunk: int = 320):
        self.audio = audio
        self.chunk = chunk
        self.paced = paced
        self.pos = 0
        self.running = False
        self.error = ""
        self.ever_data = False
        self.last_data_ts = 0.0
        self.device = type("Dev", (), {"name": "__eval__", "is_default": False})()

    def start(self) -> bool:
        self.running = True
        self._t0 = time.perf_counter()
        self.last_data_ts = time.time()
        return True

    def stop(self) -> None:
        self.running = False

    def read(self, timeout: float = 0.3):
        if not self.running or self.pos >= len(self.audio):
            time.sleep(min(timeout, 0.02) if self.paced else 0.001)
            return None
        if self.paced:
            target = self._t0 + self.pos / SR
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        data = self.audio[self.pos:self.pos + self.chunk]
        self.pos += len(data)
        self.last_data_ts = time.time()
        self.ever_data = True
        return data


# ---------------------------------------------------------------- 指标
def _norm(text: str) -> str:
    keep = []
    for ch in (text or ""):
        if ch.isalnum() or "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff":
            keep.append(ch)
    return "".join(keep)


def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _pct(vals: list) -> dict:
    if not vals:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {"p50": round(float(np.percentile(arr, 50)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "p99": round(float(np.percentile(arr, 99)), 1)}


def cer_report(refs: list, finals: list) -> dict:
    """按参考句配对后算 CER，并单独给覆盖率。

    VAD 会把碎片句并到邻句、也会把长句切开，所以不要求一句对一条；
    每条参考句取相似度最高的上屏句（≥0.4）配对。
    未配对的参考句不计入 CER，而是体现在 coverage 上，避免分段差异冒充识别错误。
    """
    used = set()
    pairs = []
    for ri, ref in enumerate(refs):
        best, best_i = 0.0, -1
        for fi, item in enumerate(finals):
            if fi in used:
                continue
            if not item["src"]:
                continue
            ratio = difflib.SequenceMatcher(None, _norm(ref), _norm(item["src"])).ratio()
            if ratio > best:
                best, best_i = ratio, fi
        if best_i >= 0 and best >= 0.4:
            used.add(best_i)
            pairs.append((ri, best_i))
    edits = sum(_lev(_norm(refs[ri]), _norm(finals[fi]["src"])) for ri, fi in pairs)
    chars = sum(len(_norm(refs[ri])) for ri, _ in pairs)
    return {
        "cer": round(edits / max(chars, 1), 4),
        "coverage": round(len(pairs) / max(len(refs), 1), 4),
    }


# ---------------------------------------------------------------- 跑一次
def run_once(args, corpus: dict, recorder: Recorder, replay: bool) -> dict:
    from core import config as cfg_mod
    from core.asr import WhisperASR
    from core.pipeline import Pipeline
    from core import translator as tr_mod

    cfg = cfg_mod.AppConfig.load()
    for item in args.set:
        if "=" not in item:
            raise SystemExit("--set 需要 k=v：%s" % item)
        k, v = item.split("=", 1)
        cur = getattr(cfg, k, None)
        if isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(v)
        elif isinstance(cur, float):
            val = float(v)
        else:
            val = v
        setattr(cfg, k, val)

    ablate = set(args.ablate)
    mods = []
    if "zh_rules" in ablate:
        cfg.enable_zh_rules = False
    if "glossary" in ablate:
        cfg.enable_translate_glossary = False
    if "glossary_on" in ablate:
        # 该功能默认关，A/B 需要显式打开
        cfg.enable_translate_glossary = True
    if "titles" in ablate:
        cfg.enable_namefix = False
    if "title_glossary" in ablate:
        cfg.enable_title_glossary = False
    if "zh_polish" in ablate:
        cfg.enable_zh_polish = False
    if "asrfix" in ablate:
        mods.append("asrfix")
    if "phrases" in ablate:
        mods.append("phrases")

    # 诊断脚本铁律：绝不写回配置
    cfg.save = lambda *a, **k: None
    cfg.auto_proxy = False

    pipe = Pipeline(cfg)
    tr = pipe.translator
    tr.probe = lambda *a, **k: None
    tr.hedge_mode = "single"          # 关掉双发：回放才能和录制走同一通道
    tr.apply_proxy = lambda *a, **k: None
    if mods:
        if "asrfix" in mods:
            WhisperASR.apply_builtin_fixes = staticmethod(lambda text: text)
        if "phrases" in mods:
            tr_mod.phrase_lookup = lambda text: None

    # ---- 识别侧钩子
    if replay:
        pipe.asr.load = lambda *a, **k: None
        pipe.asr.dev_info = "replay"

        def dec(audio, hotwords=""):
            row = recorder.lookup_asr(audio_hash(audio))
            return None if row is None else row["raw"]

        pipe.asr._decode = dec
    else:
        raw_decode = pipe.asr._decode

        def dec(audio, hotwords=""):
            t0 = time.perf_counter()
            text = raw_decode(audio, hotwords=hotwords)
            recorder.add_asr(audio_hash(audio), text, len(audio) / SR,
                             (time.perf_counter() - t0) * 1000)
            return text

        pipe.asr._decode = dec

    # ---- 翻译侧钩子
    if replay:
        def fire(ch, text, src="ja", timeout=4.0, context=""):
            row = recorder.lookup_tr(ch, text, src)
            if row is None:
                tr._tls.eval_ms = 0.0
                return None
            tr._tls.eval_ms = row["ms"]
            if row.get("err"):
                raise RuntimeError(row["err"])
            return row.get("out")

        tr._fire = fire
    else:
        raw_fire = tr._fire

        def fire(ch, text, src="ja", timeout=4.0, context=""):
            t0 = time.perf_counter()
            out, err = None, ""
            try:
                out = raw_fire(ch, text, src, timeout, context)
                return out
            except Exception as exc:
                err = type(exc).__name__
                raise
            finally:
                ms = (time.perf_counter() - t0) * 1000
                if err:
                    recorder.add_tr(ch, text, src, None, err, ms)
                else:
                    recorder.add_tr(ch, text, src, out, "", ms)
                    if out and out.strip():
                        tr._tls.eval_ms = ms

        tr._fire = fire

    # ---- 采集替身
    audio = build_material(corpus)
    cap = FakeCapture(audio, paced=not replay)
    pipe._new_capture = lambda dev: cap
    import core.pipeline as pipe_mod
    pipe_mod.pick_default_device = lambda: None

    finals = []
    calls = []
    lock = threading.Lock()

    def on_result(seq, src, tgt):
        with lock:
            finals.append({"seq": seq, "src": src, "tgt": tgt})

    orig_translate = tr.translate

    def translate(text, context=()):
        tr._tls.eval_ms = None
        out = orig_translate(text, context)
        with lock:
            calls.append({"text": text, "out": out,
                          "ch": getattr(tr._tls, "channel", "") or "",
                          "ms": tr._tls.eval_ms})
        return out

    pipe.on_result = on_result
    pipe.on_status = lambda s: None
    tr.translate = translate

    pipe.start()
    t0 = time.perf_counter()
    deadline = t0 + len(audio) / SR + 30
    while time.perf_counter() < deadline:
        if cap.pos >= len(audio):
            break
        time.sleep(0.05)
    # 尾巴：等最后一个句子断完、译完
    last_n, last_ts = -1, time.time()
    while time.perf_counter() < deadline + 40:
        time.sleep(0.2)
        with lock:
            n = len(finals)
        if n != last_n:
            last_n, last_ts = n, time.time()
        if time.time() - last_ts > args.tail:
            break
    pipe.stop()
    wall = time.perf_counter() - t0

    # ---- 组报告：每句上屏配一条翻译调用记录
    pool = {}
    for c in calls:
        pool.setdefault(c["out"], []).append(c)
    for lst in pool.values():
        lst.sort(key=lambda c: c["ch"])
    items = []
    finals.sort(key=lambda f: f["seq"])
    for f in finals:
        lst = pool.get(f["tgt"]) or []
        c = lst.pop(0) if lst else {}
        items.append({
            "seq": f["seq"],
            "src": f["src"] or "",
            "tgt": f["tgt"] or "",
            "ch": c.get("ch", ""),
            "net_ms": (round(c["ms"], 1) if c.get("ms") is not None else None),
        })

    refs = [it.get("ref") or it["text"] for it in corpus["items"]]
    cer = cer_report(refs, items)
    net = [it["net_ms"] for it in items if it["net_ms"] is not None]
    chans = {}
    for it in items:
        key = it["ch"] or "local"
        chans[key] = chans.get(key, 0) + 1

    report = {
        "tag": args.tag,
        "mode": "replay" if replay else "record",
        "ablate": sorted(ablate),
        "set": sorted(args.set),
        "corpus": {"version": corpus["version"], "sha1": corpus_sha1(corpus),
                   "refs": len(refs)},
        "config": {
            "model_size": cfg.model_size,
            "beam_size": cfg.beam_size,
            "min_speech_ms": cfg.min_speech_ms,
            "min_silence_ms": cfg.min_silence_ms,
            "max_speech_ms": cfg.max_speech_ms,
            "enable_hotwords": bool(cfg.enable_hotwords),
            "builtin_titles": bool(cfg.builtin_titles),
            "enable_namefix": bool(cfg.enable_namefix),
            "enable_title_glossary": bool(cfg.enable_title_glossary),
            "enable_translate_glossary": bool(cfg.enable_translate_glossary),
            "enable_zh_rules": bool(cfg.enable_zh_rules),
            "enable_zh_polish": bool(cfg.enable_zh_polish),
        },
        "summary": {
            "finals": len(items),
            "with_tgt": sum(1 for it in items if it["tgt"]),
            "cer": cer["cer"],
            "coverage": cer["coverage"],
            "channels": chans,
            "net_ms": _pct(net),
        },
        "items": items,
    }
    print("  用时 %.1fs，上屏 %d 条（有译文 %d），CER %.4f，缺失 %d"
          % (wall, len(items), report["summary"]["with_tgt"], cer["cer"], len(recorder.misses)))
    return report


# ---------------------------------------------------------------- 比较
def diff_against(base: dict, cur: dict) -> list:
    out = []
    a = base["items"]
    b = cur["items"]
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else {}
        y = b[i] if i < len(b) else {}
        if x.get("src") != y.get("src"):
            out.append({"i": i, "field": "src", "base": x.get("src"), "cur": y.get("src")})
        if x.get("tgt") != y.get("tgt"):
            out.append({"i": i, "field": "tgt", "base": x.get("tgt"), "cur": y.get("tgt")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="固定语料评测（录制/回放/基线）")
    ap.add_argument("--build-corpus", action="store_true", help="从 tts_cache 生成 eval/corpus.json")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--ablate", default="", help="逗号分隔的开关名")
    ap.add_argument("--set", action="append", default=[], help="配置覆盖 k=v，可重复")
    ap.add_argument("--tail", type=float, default=4.0, help="无新句子多少秒后收尾")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.ablate = [x for x in args.ablate.split(",") if x]

    if args.build_corpus:
        corpus = build_corpus()
        print("语料已生成：%d 句 → %s" % (len(corpus["items"]), CORPUS_PATH))
        return 0
    if not (args.record or args.replay):
        ap.error("需要 --record 或 --replay")

    corpus = load_corpus()
    recorder = Recorder(args.tag, args.ablate)
    if args.replay:
        recorder.load_asr()
        recorder.load_tr()
    else:
        if os.path.exists(recorder.tr_path) and not args.force:
            raise SystemExit("录制已存在：%s（要覆盖加 --force）" % recorder.tr_path)

    print("语料 %d 句，模式 %s，tag=%s%s"
          % (len(corpus["items"]), "回放" if args.replay else "录制", args.tag,
             ("，ablate=" + ",".join(args.ablate)) if args.ablate else ""))
    report = run_once(args, corpus, recorder, args.replay)

    if recorder.misses:
        for msg in recorder.misses[:20]:
            print("  [缺失] " + msg, file=sys.stderr)
        print("回放命中不足（%d 条缺失），已中止；不会回落到网络。" % len(recorder.misses),
              file=sys.stderr)
        return 2

    if not args.replay:
        recorder.flush({
            "corpus_sha1": corpus_sha1(corpus),
            "ablate": sorted(args.ablate),
            "config": report["config"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    os.makedirs(REPORT_DIR, exist_ok=True)
    name = args.tag + ("__" + "-".join(args.ablate) if args.ablate else "")
    out_path = os.path.join(REPORT_DIR, name + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("  报告：%s" % out_path)

    if args.save_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("  基线：%s" % BASELINE_PATH)

    if args.no_compare or args.save_baseline or not os.path.exists(BASELINE_PATH):
        return 0

    with open(BASELINE_PATH, encoding="utf-8") as f:
        base = json.load(f)
    diffs = diff_against(base, report)
    s0, s1 = base["summary"], report["summary"]
    print("\n与基线对比")
    print("  CER %.4f → %.4f   条数 %d → %d   有译文 %d → %d"
          % (s0["cer"], s1["cer"], s0["finals"], s1["finals"],
             s0["with_tgt"], s1["with_tgt"]))
    print("  net_ms P50/P90/P99 %.1f/%.1f/%.1f → %.1f/%.1f/%.1f"
          % (s0["net_ms"]["p50"], s0["net_ms"]["p90"], s0["net_ms"]["p99"],
             s1["net_ms"]["p50"], s1["net_ms"]["p90"], s1["net_ms"]["p99"]))
    if not diffs:
        print("  逐句：完全一致")
        return 0
    print("  逐句差异 %d 处：" % len(diffs))
    for d in diffs:
        print("   #%02d %s\n      base: %s\n      cur : %s"
              % (d["i"], d["field"], d["base"], d["cur"]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
