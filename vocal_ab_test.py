# -*- coding: utf-8 -*-
"""人声提取（中置提取）A/B 复测：普通降混 vs 中置提取，四种 BGM 条件。

**为什么要重测**：上一轮报的「重 BGM 下 CER 100% → 4.6%」是临时脚本跑出来的，
而 2026-10-13 排查时发现 `WhisperASR._accept` 有一条*跨调用*的「连续相同文本 = 幻觉」
护栏——同一段音频连喂三遍，第三遍必被判幻觉丢成空串。临时脚本有没有踩到它无法回溯，
所以这里用干净口径（每次转写前 reset_repeat_guard）重跑一遍，把数字钉死。

BGM 用两种声场：
  - 反相宽声场（L=+b, R=-b）：中置提取能压掉 100dB 级，这是提取的**上界**；
  - 部分去相关（R 是 b 的延迟 0.3s 版本）：真实混音的常见形态，提取只有 ~3dB，这是**实际收益**。

用法：.runtime\\venv\\Scripts\\python.exe vocal_ab_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.asr import WhisperASR                 # noqa: E402
from core.audio import resample, TARGET_SR      # noqa: E402
from core.config import AppConfig               # noqa: E402
from core.denoise import Denoise                # noqa: E402
from core.vocals import CenterExtractor         # noqa: E402
from title_asr_test import cer, decode, tts     # noqa: E402

LINES = [
    "こんにちは、みなさん！今日も元気に配信していきます",
    "黒神話：悟空が今すごく話題になってますよね",
    "このゲームは本当に難しくて、何回も死んじゃいました",
    "ありがとうございます、すごく嬉しいです",
]
SR48 = 48000
STRENGTH = 0.8


def bgm_pair(n: int, kind: str, seed: int = 7) -> tuple:
    """生成 BGM 立体声对（音乐感：几个音 + 粉噪）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR48
    b = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 330 * t)
    b = b + rng.standard_normal(n) * 0.15
    b = b.astype(np.float32)
    if kind == "反相":
        return b, -b
    # 部分去相关：右声道延迟 0.3s（真实混响/立体声加宽常见形态）
    d = int(0.3 * SR48)
    r = np.concatenate([np.zeros(d, dtype=np.float32), b[:n - d]])
    return b, r


def path(audio48: np.ndarray, mode: str) -> np.ndarray:
    """mode: 'downmix' 普通降混 | 'center' 中置提取。返回 16k 单声道。"""
    if mode == "center":
        ex = CenterExtractor(strength=STRENGTH)
        x = ex.process(audio48[:, 0], audio48[:, 1])
    else:
        x = audio48.mean(axis=1)
    return resample(x, SR48, TARGET_SR)


def db(a: np.ndarray, b: np.ndarray) -> float:
    ra = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    rb = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
    return 20 * np.log10((ra + 1e-12) / (rb + 1e-12))


def main() -> int:
    cfg = AppConfig.load()
    asr = WhisperASR(cfg.model_size, cfg.asr_device, cfg.compute_type, cfg.source_lang)
    print("加载模型 %s …" % cfg.model_size, flush=True)
    asr.load()
    print("设备：%s\n" % asr.dev_info, flush=True)

    def rec(audio: np.ndarray) -> str:
        asr.reset_repeat_guard()
        return asr.transcribe(audio, roster="") or ""

    # BGM 各条件下的抑制量（只喂 BGM，不含人声）
    print("── BGM 抑制量（只喂 BGM）──")
    n0 = SR48 * 4
    for kind in ("反相", "部分去相关"):
        bl, br = bgm_pair(n0, kind)
        st = np.stack([bl, br], axis=1)
        d_in = path(st, "downmix")
        d_out = path(st, "center")
        print("  %-6s 立体声对：降混 %.2f → 中置 %.2f　抑制 %.1f dB"
              % (kind, np.sqrt((d_in.astype(np.float64) ** 2).mean()),
                 np.sqrt((d_out.astype(np.float64) ** 2).mean()), db(d_in, d_out)))

    print("\n── 识别 A/B（每格 = 4 句平均 CER）──")
    header = "%-18s %10s %10s" % ("条件", "① 普通降混", "② 中置提取")
    print(header)
    for level, kind, label in ((0.0, None, "无 BGM"),
                               (0.7, "部分去相关", "轻 BGM·部分去相关"),
                               (1.6, "部分去相关", "重 BGM·部分去相关"),
                               (1.6, "反相", "重 BGM·反相宽声场")):
        cers = {"downmix": 0.0, "center": 0.0}
        outs = {"downmix": [], "center": []}
        for line in LINES:
            v48 = resample(decode(tts(line)), TARGET_SR, SR48).astype(np.float32)
            if level > 0:
                bl, br = bgm_pair(len(v48), kind)
                scale = level * float(np.sqrt(np.mean(v48.astype(np.float64) ** 2))) / \
                    (float(np.sqrt(np.mean(bl.astype(np.float64) ** 2))) + 1e-12)
                bl, br = bl * scale, br * scale
                st = np.stack([v48 + bl, v48 + br], axis=1)
            else:
                st = np.stack([v48, v48], axis=1)
            for mode in ("downmix", "center"):
                txt = rec(path(st, mode))
                cers[mode] += cer(line, txt)
                outs[mode].append(txt)
        n = len(LINES)
        print("%-18s %9.1f%% %9.1f%%" % (label, cers["downmix"] / n * 100, cers["center"] / n * 100))
        if cers["center"] != cers["downmix"]:
            print("    ① 降混：%s" % " ／ ".join(outs["downmix"]))
            print("    ② 提取：%s" % " ／ ".join(outs["center"]))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
