# -*- coding: utf-8 -*-
"""专有名词（游戏 / 番剧 / 人名标题）识别测试：裸识别 vs 热词偏置 vs 二次解码。

现象（用户报告）：主播说「黒神話：悟空」，被识别成「黒色」——错在**识别源头**，
翻译层再准也救不回来（识别错 → 译文错是错误主因）。

本脚本把合成语音**直接喂给 WhisperASR**（不经扬声器与采集链路，排除播放/采集变量），
对同一段音频跑三种解码设置并对比：

  ① 裸识别（现状）
  ② 热词偏置：把标题下发给解码器（`hotwords`）
  ③ 二次解码：先裸识别，命中「近似串」后再带热词重解一次（本轮新增的救援路径）

用法：.runtime\\venv\\Scripts\\python.exe title_asr_test.py
"""
from __future__ import annotations

import difflib
import hashlib
import os
import sys
import time
import urllib.parse

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.asr import WhisperASR              # noqa: E402
from core.config import AppConfig            # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".runtime", "tts_cache")
os.makedirs(CACHE, exist_ok=True)


def tts(text: str) -> str:
    """百度日语 TTS（与 latency_test 同一实现，缓存共用）。"""
    name = hashlib.md5(text.encode("utf-8")).hexdigest()[:16] + ".mp3"
    path = os.path.join(CACHE, name)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        url = ("https://fanyi.baidu.com/gettts?lan=jp&text=%s&spd=5&source=web"
               % urllib.parse.quote(text))
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def decode(path: str) -> np.ndarray:
    """mp3 → 16k 单声道 float32。

    `faster_whisper.audio.decode_audio` 在本机 PyAV 版本上不可用
    （`av.open(metadata_errors=...)` TypeError），故自己走 AudioResampler。
    """
    import av  # 延迟导入：只有本脚本用得到

    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):      # 冲刷缓冲
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# (标题展示写法, 标题读音（给解码器用的假名）, 含标题的一句自然日语)
CASES = [
    ("黒神話：悟空", "こくしんわごくう", "黒神話：悟空が今すごく話題になってますよね"),
    ("ゼルダの伝説", "ぜるだのでんせつ", "今日はゼルダの伝説を最初からやっていきます"),
    ("エルデンリング", "えるでんりんぐ", "エルデンリングのDLCを買っちゃいました"),
    ("モンハンワイルズ", "もんはんわいるず", "モンハンワイルズの発売日が決まったみたいです"),
    ("原神", "げんしん", "原神の新しいイベントが始まりました"),
    ("スプラトゥーン", "すぷらとぅーん", "スプラトゥーンを久しぶりに起動しました"),
]


def cer(ref: str, hyp: str) -> float:
    """字符错误率（编辑距离 / 参考长度）。空格与标点不参与，只看假名与汉字。"""
    drop = set(" 　、。！？!?：:・「」『』，,")
    r = [c for c in ref if c not in drop]
    h = [c for c in hyp if c not in drop]
    if not r:
        return 0.0 if not h else 1.0
    sm = difflib.SequenceMatcher(None, r, h)
    same = sum(b.size for b in sm.get_matching_blocks())
    return 1.0 - same / len(r)


def hit(title: str, hyp: str) -> bool:
    """标题是否基本被正确识别：标题里去标点后，有 ≥60% 的字符按顺序出现。"""
    t = [c for c in title if c not in " ：:・"]
    h = hyp.replace(" ", "")
    if not t:
        return False
    sm = difflib.SequenceMatcher(None, t, h)
    return sum(b.size for b in sm.get_matching_blocks()) / len(t) >= 0.6


# 误伤扫描用：**不含任何标题**的普通直播句
PLAIN = [
    "こんにちは、みなさん！今日も元気に配信していきます",
    "えっと、まず最初に設定をちょっと確認しておきますね",
    "ありがとうございます、すごく嬉しいです",
    "ちょっと待ってください、今あとで説明します",
    "このゲームは本当に難しくて、何回も死んじゃいました",
    "じゃあ、そろそろ始めていきましょうか",
]

# 热词表：把上面的标题都塞进去，看普通句会不会被「凭空插入」标题
HOT_BIG = "、".join(c[0] for c in CASES)


def scan_false_positive(asr, roster_hot: str) -> None:
    """普通句 × 大热词表：① 是否注入标题　② CER 是否变差。"""
    print("\n" + "=" * 64)
    print("误伤扫描：%d 句普通直播句（不含任何标题）× 热词表（%d 个标题）"
          % (len(PLAIN), len(CASES)))
    print("热词表：%s" % roster_hot)
    import re as _re

    base_cer = hot_cer = 0.0
    injected = []
    for s in PLAIN:
        audio = decode(tts(s))
        a = asr.transcribe(audio, roster="") or ""
        b = asr.transcribe(audio, roster="", hotwords=roster_hot) or ""
        base_cer += cer(s, a)
        hot_cer += cer(s, b)
        for title, _rd, _sent in CASES:
            key = [c for c in title if _re.match(r"[\u4e00-\u9fff\u30a1-\u30f6]", c)]
            if key and all(c in b for c in key) and not all(c in a for c in key):
                injected.append((s, title, a, b))
        print("  原：%s" % a)
        print("  热：%s" % b)
    n = len(PLAIN)
    print("\n平均 CER：裸 %.1f%%　带热词 %.1f%%" % (base_cer / n * 100, hot_cer / n * 100))
    print("凭空插入标题：%d 处 %s" % (len(injected), injected if injected else "（无）"))


def run_e2e(hotwords_on: bool) -> None:
    """端到端（真实链路）：播到扬声器 → WASAPI 回环抓 → VAD → Whisper → 字幕。

    验证的是「接线」：配置开关有没有真的把热词送进解码器、复述护栏会不会误杀正常句。
    """
    from latency_test import decode_mp3, play_wasapi
    from core.pipeline import Pipeline

    cfg = AppConfig.load()
    orig = cfg.enable_hotwords
    cfg.enable_hotwords = hotwords_on
    print("\n" + "=" * 64)
    print("端到端模式：enable_hotwords=%s（内置标题=%s，用户词表 %d 字）"
          % (hotwords_on, cfg.builtin_titles, len(cfg.name_roster or "")))

    outs = []
    levels = []
    pipe = Pipeline(cfg)
    pipe.on_result = lambda seq, src, out: outs.append((src, out))
    pipe.on_partial = lambda seq, src: None
    pipe.on_status = lambda s: print("  [状态] " + s, flush=True)
    pipe.on_level = lambda v: levels.append(v)
    dev = cfg.device or {}
    print("  设备：%s（index=%s, %s）" % (dev.get("name"), dev.get("index"), dev.get("kind")))
    pipe.start()
    import time as _t
    for _ in range(120):
        if pipe.state in ("running", "error"):
            break
        _t.sleep(0.5)
    if pipe.state != "running":
        print("启动失败，放弃", flush=True)
        return
    _t.sleep(1.0)
    loop_name = (dev.get("name") or "")
    for _title, _reading, line in CASES:
        audio = decode_mp3(tts(line))
        play_wasapi(audio, 16000, loop_name)
        _t.sleep(len(audio) / 16000.0 + 0.6)
    _t.sleep(9.0)
    print("  采集电平：收到 %d 次回调，峰值 %.3f %s"
          % (len(levels), max(levels) if levels else 0.0,
             "（有声音流入）" if levels and max(levels) > 0.005 else "⚠ 几乎没声音，采集可能没抓到"))
    pipe.stop()
    print("字幕（原文 → 译文）：")
    for src, out in outs:
        print("  %s" % src)
        print("    → %s" % out)
    hit_n = sum(1 for t, _r, _s in CASES
                if any(hit(t, src) for src, _o in outs))
    print("标题命中：%d/%d" % (hit_n, len(CASES)))
    # 测试只该改内存里的开关：`_restart_capture` 会把 cfg 落盘，
    # 不还原就会把用户真实的配置（enable_hotwords）改掉——实测踩过。
    cfg.enable_hotwords = orig
    cfg.save()


def main() -> int:
    cfg = AppConfig.load()
    asr = WhisperASR(cfg.model_size, cfg.asr_device, cfg.compute_type, cfg.source_lang)
    print("加载模型 %s …" % cfg.model_size, flush=True)
    asr.load()
    print("设备：%s\n" % asr.dev_info, flush=True)

    rows = []
    for title, reading, sentence in CASES:
        audio = decode(tts(sentence))
        results = []
        # ① 现状：裸识别
        t0 = time.time()
        base = asr.transcribe(audio, roster="") or ""
        results.append(("① 裸识别", base, time.time() - t0))
        # ② 热词 = 标题的展示写法（汉字原样）
        t0 = time.time()
        hot_cjk = asr.transcribe(audio, roster="", hotwords=title) or ""
        results.append(("② 热词·汉字", hot_cjk, time.time() - t0))
        # ③ 热词 = 标题的假名读音（模块文档声称解码器更易利用）
        t0 = time.time()
        hot_kana = asr.transcribe(audio, roster="", hotwords=reading) or ""
        results.append(("③ 热词·假名", hot_kana, time.time() - t0))

        print("【%s】　读音 = %s" % (title, reading))
        print("  参考 ：%s" % sentence)
        for name, text, dt in results:
            print("  %-11s：%s   (CER %.1f%%, %.2fs, 命中=%s)"
                  % (name, text, cer(sentence, text) * 100, dt, hit(title, text)))
        print(flush=True)
        rows.append((title, sentence, results))

    n = len(rows)
    print("=" * 64)
    for idx, label in enumerate(["① 裸识别", "② 热词·汉字", "③ 热词·假名"]):
        h = sum(hit(r[0], r[2][idx][1]) for r in rows)
        c = sum(cer(r[1], r[2][idx][1]) for r in rows) / n
        print("%-11s 标题命中 %d/%d　平均 CER %.1f%%" % (label, h, n, c * 100))

    if "--fp" in sys.argv or os.environ.get("TITLE_TEST_FP") == "1":
        scan_false_positive(asr, HOT_BIG)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if "--e2e" in sys.argv:
        run_e2e(hotwords_on=True)
        run_e2e(hotwords_on=False)
        sys.exit(0)
    sys.exit(main())
