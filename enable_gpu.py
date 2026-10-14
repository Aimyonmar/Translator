# -*- coding: utf-8 -*-
"""把 cublas 运行库放进 ctranslate2 能找到的位置，并验证 GPU 推理是否可用。

用法（装完 nvidia-cublas-cu12 后运行）：
    .runtime\\venv\\Scripts\\python.exe enable_gpu.py

原理：pip 把 cublas64_12.dll 装在 site-packages\\nvidia\\cublas\\bin，
但 Windows 加载 DLL 时不搜这个目录，所以要复制到 ctranslate2 自己的目录
（它已经自带 cudnn64_9.dll，同级目录一定能被找到）。
"""
import os
import shutil
import sys

VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    ".runtime", "venv", "Lib", "site-packages")
CT_DIR = os.path.join(VENV, "ctranslate2")
NVIDIA_DIR = os.path.join(VENV, "nvidia")


def find_cublas():
    hits = []
    for root, _dirs, files in os.walk(NVIDIA_DIR):
        for f in files:
            if f.lower().startswith("cublas") and f.lower().endswith(".dll"):
                hits.append(os.path.join(root, f))
    return hits


def main():
    if not os.path.isdir(NVIDIA_DIR):
        print("未找到 nvidia 运行库目录，请先执行：")
        print("  .runtime\\venv\\Scripts\\python.exe -m pip install "
              "-i https://mirrors.aliyun.com/pypi/simple/ nvidia-cublas-cu12")
        return 1

    dlls = find_cublas()
    if not dlls:
        print("nvidia 目录下没有 cublas DLL，安装可能不完整。")
        return 1

    print("找到运行库：")
    for d in dlls:
        print("   ", d)
        dst = os.path.join(CT_DIR, os.path.basename(d))
        shutil.copy2(d, dst)
        print("    已复制到 ctranslate2 目录 ->", os.path.basename(dst))

    print("\n验证 GPU 推理…")
    try:
        import numpy as np

        # 优先用全局模型缓存；有本地缓存就别联网
        # （本机系统代理会把联网检查拦成 ProxyError 502）
        cands = [
            os.environ.get("HF_HOME"),
            # 兼容「模型集中放在项目上一级 settings/hf」的布局
            os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "settings", "hf")),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime", "hf"),
        ]
        hf = next((c for c in cands if c and os.path.isdir(os.path.join(c, "hub"))),
                  cands[1])
        os.environ["HF_HOME"] = hf
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("模型缓存:", hf)

        from faster_whisper import WhisperModel
        m = WhisperModel("small", device="cuda", compute_type="float16")
        segs, _ = m.transcribe(np.zeros(1600, dtype=np.float32),
                               language="ja", beam_size=1, vad_filter=False)
        list(segs)
        print("GPU 推理成功！")
    except Exception as exc:
        print("GPU 仍不可用：", type(exc).__name__, exc)
        print("可先继续用 CPU（软件会自动回退，功能不受影响）。")
        return 2

    # 让软件下次启动时敢用 GPU
    import json
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["gpu_ok"] = True
        json.dump(cfg, open(cfg_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("config.json 的 gpu_ok 已设为 True，下次启动将使用 GPU。")
    except Exception as exc:
        print("写 config.json 失败（可手动把 gpu_ok 改成 true）：", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
