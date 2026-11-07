# -*- coding: utf-8 -*-
"""本轮修复的回归自测（离线运行，不联网、不加载模型、不碰用户模型缓存）。

覆盖：
    - 离线转写语种归一（kotoba 强制 ja）
    - CPU + 自定义档的降级链
    - 解码阶段取消
    - 翻译缓存 LRU
    - 模型仓库映射 / 准备模型.py --check
    - 两个 .bat 的编码与坏 venv 判定
    - 面板：卸载判断、工程 JSON 元数据、模型缓存互斥、min_speech 提示
    - VAD 句首块重复（回归）
用法：.runtime\\venv\\Scripts\\python.exe fix_verify_test.py
"""
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import wave
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".runtime", "hf"))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + ((" | " + detail) if detail else ""))


def main() -> int:
    from core.asr import WhisperASR
    from core.transcribe import decode_audio, transcribe_media
    from core.translator import Translator
    from core import model_store

    tmp = tempfile.mkdtemp(prefix="fix_verify_")
    wav = os.path.join(tmp, "a.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)

    class FakeModel:
        def __init__(self):
            self.kw = None

        def transcribe(self, audio, **kw):
            self.kw = kw
            return iter([]), types.SimpleNamespace(duration=len(audio) / 16000.0)

    class FakeASR:
        def __init__(self):
            self.model = FakeModel()
            self.lock = threading.Lock()
            self.batch_size = 1

    # ---------------------------------------------------- 语种归一（kotoba）
    kotoba = WhisperASR("kotoba-whisper-v2.0", "cpu", "int8", "auto")
    vals = [kotoba.resolve_language(x) for x in ("auto", "", "en", None)]
    check("kotoba 语种归一恒为 ja", vals == ["ja"] * 4, str(vals))
    normal = WhisperASR("small", "cpu", "int8", "ja")
    check("普通档不改变调用方语种",
          normal.resolve_language("en") == "en"
          and normal.resolve_language("auto") is None
          and normal.resolve_language("") is None)
    fake = FakeASR()
    transcribe_media(fake, wav, language=kotoba.resolve_language("auto"),
                     word_timestamps=False)
    check("离线转写下发的 language 是 ja", fake.model.kw.get("language") == "ja",
          str(fake.model.kw.get("language")))

    # ---------------------------------------------------- CPU 降级链
    plan = WhisperASR("kotoba-whisper-v2.0", "cpu", "auto")._plan("cpu")
    check("CPU + kotoba 有 small 兜底",
          [c[0] for c in plan] == ["kotoba-whisper-v2.0", "small"], str(plan))
    turbo = WhisperASR("large-v3-turbo", "cpu", "auto")
    check("CPU + large 系直接降 small",
          turbo._plan("cpu") == [("small", "cpu", "int8")], str(turbo._plan("cpu")))
    check("CPU + small 不重复追加",
          WhisperASR("small", "cpu", "auto")._plan("cpu") == [("small", "cpu", "int8")])

    # ---------------------------------------------------- 解码阶段取消
    calls = {"n": 0}

    def stop_after_2():
        calls["n"] += 1
        return calls["n"] >= 2

    arr = decode_audio(wav, should_stop=stop_after_2)
    check("decode_audio 中途取消返回空数组", arr.size == 0 and calls["n"] >= 2,
          f"size={arr.size}")
    fake2 = FakeASR()
    calls2 = {"n": 0}

    def stop2():
        calls2["n"] += 1
        return calls2["n"] >= 2

    segs, total = transcribe_media(fake2, wav, language="ja",
                                   word_timestamps=False, should_stop=stop2)
    check("解码阶段取消后不再进入识别",
          segs == [] and total == 0.0 and fake2.model.kw is None)

    # ---------------------------------------------------- 翻译缓存 LRU
    from collections import OrderedDict
    t = Translator.__new__(Translator)
    t._cache = OrderedDict()
    t.lock = threading.Lock()
    t._tls = threading.local()
    t._cache["あ"] = "A"
    t._cache["い"] = "I"
    out = t.translate("あ")
    check("缓存命中后移到队尾（LRU）",
          out == "A" and list(t._cache)[-1] == "あ", str(list(t._cache)))

    # ---------------------------------------------------- 仓库映射 / --check
    check("repo_id 覆盖 turbo / kotoba",
          model_store.repo_id("large-v3-turbo").startswith("mobiuslabsgmbh/")
          and model_store.repo_id("kotoba-whisper-v2.0").startswith("kotoba-tech/"))
    check("缓存目录不再是 Systran 前缀",
          "Systran" not in model_store.model_dir("large-v3-turbo"),
          os.path.basename(model_store.model_dir("large-v3-turbo")))

    hf = tempfile.mkdtemp(prefix="hf_check_")
    repo = model_store.repo_id("kotoba-whisper-v2.0")
    snap = os.path.join(hf, "hub", "models--" + repo.replace("/", "--"),
                        "snapshots", "rev1")
    os.makedirs(snap)
    for f in ("model.bin", "config.json", "tokenizer.json"):
        open(os.path.join(snap, f), "w").write("x")
    env = dict(os.environ, HF_HOME=hf)
    cmd = [sys.executable, os.path.join(ROOT, "准备模型.py"),
           "--check", "kotoba-whisper-v2.0"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    check("已缓存：--check 返回 0", r.returncode == 0, f"rc={r.returncode}")
    os.remove(os.path.join(snap, "model.bin"))
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    check("缓存不完整：--check 返回 1", r.returncode == 1, f"rc={r.returncode}")

    spec = importlib.util.spec_from_file_location(
        "prep_model", os.path.join(ROOT, "准备模型.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    reqs = []
    fake_hf = types.ModuleType("huggingface_hub")

    def _snapshot_download(repo_id, **kw):
        reqs.append(repo_id)
        raise RuntimeError("fix_verify: stop before real download")

    fake_hf.snapshot_download = _snapshot_download
    sys.modules["huggingface_hub"] = fake_hf
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="hf_dl_")
    argv = sys.argv
    sys.argv = ["准备模型.py", "large-v3-turbo"]
    try:
        with redirect_stdout(io.StringIO()):
            prep.main()
    finally:
        sys.argv = argv
    check("准备模型.py 请求 mobius 仓库而不是 Systran",
          reqs and reqs[-1] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo", str(reqs))

    bat_start = open(os.path.join(ROOT, "启动翻译器.bat"), "rb").read().decode("gbk")
    check("启动脚本改用 --check，不再用 Systran 前缀判定",
          "--check" in bat_start
          and 'dir /b "%MODELDIR%\\models--Systran' not in bat_start)

    # ---------------------------------------------------- .bat 字节格式 / 坏 venv
    for name in ("启动翻译器.bat", "安装依赖.bat"):
        raw = open(os.path.join(ROOT, name), "rb").read()
        cr, lf = raw.count(b"\r"), raw.count(b"\n")
        try:
            raw.decode("utf-8")
            u8_fail = False
        except UnicodeDecodeError:
            u8_fail = True
        check(f"{name} GBK+CRLF+无BOM+非UTF8",
              cr == lf and not raw.startswith(b"\xef\xbb\xbf")
              and u8_fail and bool(raw.decode("gbk")),
              f"CR={cr}, LF={lf}")

    fake_venv = tempfile.mkdtemp(prefix="badvenv_")
    os.makedirs(os.path.join(fake_venv, "Scripts"))
    shutil.copy(r"C:\Windows\System32\where.exe",
                os.path.join(fake_venv, "Scripts", "python.exe"))
    rc_bad = subprocess.run([os.path.join(fake_venv, "Scripts", "python.exe"),
                             "-c", "import sys"], capture_output=True).returncode
    check("坏 venv 的健康检查返回非 0（触发重建）", rc_bad != 0, f"rc={rc_bad}")
    rc_ok = subprocess.run([sys.executable, "-c", "import sys"],
                           capture_output=True).returncode
    check("正常 venv 返回 0（不会误删）", rc_ok == 0, f"rc={rc_ok}")
    bat_ins = open(os.path.join(ROOT, "安装依赖.bat"), "rb").read().decode("gbk")
    check("安装脚本含「坏 venv 检测 + 重建」分支",
          '-c "import sys"' in bat_ins and 'rmdir /s /q "%VENV%"' in bat_ins)

    # ---------------------------------------------------- VAD 句首重复（回归）
    import numpy as np
    from core.vad import SpeechSegmenter
    seg = SpeechSegmenter(sample_rate=16000, sensitivity=0.7, min_speech_ms=250,
                          min_silence_ms=400, max_speech_ms=4000, pre_roll_ms=300)
    zeros = np.zeros(1024, dtype="float32")
    ones = np.ones(1024, dtype="float32") * 0.05
    got = []
    for b in [zeros] * 5 + [ones] * 5 + [zeros] * 8:
        out, _lvl = seg.feed(b)
        if out is not None:
            got.append(out)
    voiced_samples = sum(int(np.sum(o > 0.02)) for o in got)
    check("VAD 句首块不再重复（有声音的采样数 = 5 块）",
          voiced_samples == 5 * 1024 and len(got) == 1,
          f"voiced={voiced_samples}, segs={len(got)}")

    # ---------------------------------------------------- 面板层
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    import ui.panel as P
    from core.config import AppConfig
    from core.transcribe import Cue

    class StubVAD:
        last_level = 0.02
        threshold = 0.01

    class StubASR:
        def __init__(self):
            self.model_size = "large-v3"
            self._loaded_size = "small"
            self.dev_info = "cpu/int8"
            self._force_ja = False
            self.model = object()
            self.unloaded = 0

        def unload(self):
            self.unloaded += 1
            self.model = None

    asr = StubASR()
    pipeline = types.SimpleNamespace(vad=StubVAD(), asr=asr)

    class StubCtrl:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    StubCtrl.pipeline = pipeline

    cfg = AppConfig()
    cfg.source_lang = "auto"
    panel = P.ControlPanel(cfg, StubCtrl())

    panel._release_loaded("small")
    check("卸载按 _loaded_size 判断（旧代码漏卸）", asr.unloaded == 1)
    asr.model = object()
    panel._release_loaded("large-v3")
    check("不误卸仍在内存里的其他档（旧代码误卸）", asr.unloaded == 1)

    asr._force_ja = True
    asr._loaded_size = "kotoba-whisper-v2.0"
    asr.dev_info = "cpu/int8"
    panel._trs_segments = [Cue(0.0, 1.5, "こんにちは", [], "你好")]
    panel._trs_path = os.path.join(tmp, "media.wav")
    panel._trs_total = 12.5
    out_path = os.path.join(tmp, "proj.json")
    orig = P.QFileDialog.getSaveFileName
    P.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_path, ""))
    try:
        panel._export_trs("json")
    finally:
        P.QFileDialog.getSaveFileName = orig
    data = json.load(open(out_path, encoding="utf-8"))
    check("工程 JSON：duration_s 用解码总时长",
          data["media"]["duration_s"] == 12.5, str(data["media"]["duration_s"]))
    check("工程 JSON：model/device/language 用实际值",
          data["asr"]["model"] == "kotoba-whisper-v2.0"
          and data["asr"]["device"] == "cpu/int8"
          and data["media"]["language"] == "ja",
          str(data["asr"]) + " / " + str(data["media"]["language"]))

    logs = []
    panel._log = lambda t: logs.append(t)
    panel._mdl_busy = None
    panel._running = True
    panel._on_mdl_click("small")
    check("实时运行中拒绝下载/卸载", panel._mdl_busy is None, str(logs[-1:]))
    panel._running = False
    panel._transcribing = True
    panel._on_mdl_click("small")
    check("转写期间拒绝下载/卸载", panel._mdl_busy is None)

    panel._transcribing = False
    orig_dl = P.model_store.is_downloaded
    P.model_store.is_downloaded = lambda s: False
    panel._mdl_task = lambda *a: None
    try:
        panel._on_mdl_click("small")
        check("下载期间开始/换模型/换设备全部禁用",
              not panel.btn_start.isEnabled()
              and not panel.btn_trs_start.isEnabled()
              and not panel.cmb_model.isEnabled()
              and not panel.cmb_dev.isEnabled())
        panel._on_mdl_done()
        check("任务结束后按钮恢复",
              panel.btn_start.isEnabled() and panel.btn_trs_start.isEnabled()
              and panel.cmb_model.isEnabled() and panel.cmb_dev.isEnabled())
    finally:
        P.model_store.is_downloaded = orig_dl
        panel._mdl_busy = None

    cfg.min_speech_ms = 100
    panel._update_diag()
    text = panel.lbl_diag.text()
    check("min_speech_ms<200 有可见提示", "低于 200ms" in text, text[:60])
    cfg.min_speech_ms = 250
    panel._update_diag()
    check("默认 250ms 不误报", "低于 200ms" not in panel.lbl_diag.text())
    panel.close()

    bad = [n for n, ok, _ in results if not ok]
    print()
    print(f"合计 {len(results)} 项，失败 {len(bad)} 项"
          + (("：" + "; ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
