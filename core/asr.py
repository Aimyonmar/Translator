# -*- coding: utf-8 -*-
"""日语语音识别：基于 faster-whisper（CTranslate2），CPU 即可实时跑 small/base 模型。"""
from __future__ import annotations

import os
import re
import subprocess
import threading
from typing import Callable, List, Optional

import numpy as np

# 模型缓存放在项目内的 .runtime/hf，不占用系统盘（C:\Users\xxx\.cache 默认在 C 盘）
# 模型动辄几百 MB，放在项目里还能随项目一起搬走
_RUNTIME_HF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runtime", "hf"
)
try:
    os.makedirs(_RUNTIME_HF, exist_ok=True)
    os.environ.setdefault("HF_HOME", _RUNTIME_HF)
except Exception:
    pass

# 国内默认走 HF 镜像；镜像的 xet 传输常报 401，关掉改走普通 HTTP 下载
# 若用户自己设过这些变量，则尊重用户的设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Windows 默认无权创建符号链接，hub 会留下 0 字节的空模型文件，直接改成复制
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def _gpu_free_gb() -> Optional[float]:
    """当前显卡「空闲」显存（GB）；查不到返回 None。

    关键场景：开着游戏看直播时，6GB 的 4050 可能只剩 1~2GB，
    large-v3 int8（约 2.4GB）根本放不进去 —— 硬上只会 OOM 然后掉到
    CPU 跑 large（实测 RTF≈4，一句话要 15 秒）。所以加载前先看余量。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.decode("utf-8", "ignore").splitlines():
            line = line.strip()
            if line:
                return float(line.split()[0]) / 1024.0
    except Exception:
        pass
    return None


_BATCH_SUPPORTED = None


def _batch_supported() -> bool:
    """faster-whisper 的 batch_size 参数从 0.9.0 才开始支持。用户环境里可能装了更老的版本
    （实测报 unexpected keyword argument 'batch_size'），这里探测一次并缓存，
    旧版本就不传，避免每句识别都抛 TypeError 导致字幕全空白。"""
    global _BATCH_SUPPORTED
    if _BATCH_SUPPORTED is None:
        try:
            import inspect
            from faster_whisper import WhisperModel
            _BATCH_SUPPORTED = "batch_size" in inspect.signature(WhisperModel.transcribe).parameters
        except Exception:
            _BATCH_SUPPORTED = False
    return _BATCH_SUPPORTED


_HOTWORDS_SUPPORTED = None


def _hotwords_supported() -> bool:
    """hotwords 参数 1.1.0 起才有；老版本传了会 TypeError，先探测再决定是否下发。"""
    global _HOTWORDS_SUPPORTED
    if _HOTWORDS_SUPPORTED is None:
        try:
            import inspect
            from faster_whisper import WhisperModel
            _HOTWORDS_SUPPORTED = "hotwords" in inspect.signature(WhisperModel.transcribe).parameters
        except Exception:
            _HOTWORDS_SUPPORTED = False
    return _HOTWORDS_SUPPORTED


# Whisper 在静音/噪音上会凭空吐出的固定句子（真正的幻觉）。
# 注意：只放「整段音频没声音时才会出现的模板句」，千万不能放日常用语——
# 「ありがとうございます」这类致谢在 VTuber 直播里高频出现，一旦列进来会被整句吞掉，
# 表现为「明明在说话但字幕不出字」。
HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "字幕の作成は",
    "字幕の作成は、",
    "ご視聴ありがとうございます",
    # 旧版 initial_prompt 的原文会被 Whisper 原样复述出来，留着兜底
    "日本語の自然な会話です",
    "日本語の自然な会話です。",
    "日本語の自然な会話です。固有名詞はカタカナで。",
    "固有名詞はカタカナで",
}

# 关于「热词/提示词」：
# 早期版本把热词当 initial_prompt 传给 Whisper，实测三种写法全会被原样复述进字幕
# （完整句、词表、长音频都中过招），表现为「明明在说话却冒出莫名其妙的一句」。
# 所以**绝不再用 initial_prompt 传热词**。改用更安全的两套机制：
#   1. word_map（纠错替换表）：识别完成后做确定性字符串替换，零回声风险，
#      用来修掉 Whisper 把人名/梗听错的部分（如「うさだぺこら」→「兎田ぺこら」）。
#   2. 真要提升专有名词命中率，faster-whisper 1.2 的 hotwords= 参数虽同走 prompt，
#      但本机实测在静音上不会复述；考虑到风险仍不默认开启，仅作内部保留。


class WhisperASR:
    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "auto", language: str = "ja"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None
        self.lock = threading.Lock()
        self._last_text = ""
        self._repeat = 0
        self.dev_info = ""        # 实际生效的设备，如 "cuda/float16" 或 "cpu/int8"
        self.last_error = ""      # 最近一次识别异常，供界面提示
        # 解码宽度：beam 越小越快、越大越准。实时字幕默认 2（比 5 快约 2~3 倍，准确度几乎不掉）；
        # 面板可拖到 1（最快）或 3~5（更准）。温度回退只在低置信度时触发，不计常速。
        self.beam_size = 2
        self.batch_size = 1       # GPU 上一次并行解码多个语音块，提升吞吐（CPU 无意义）

    # -- 加载
    def _plan(self, dev: str) -> List[tuple]:
        """按「显存余量 + 失败自动降级」给出加载链（前面的组合失败就试下一个）。

        实测（RTX 4050 6GB）：cuda/int8 的 large-v3 识别 5.3s 音频只要 1.0s；
        但掉到 cpu/int8 后同一句要 14.7s（RTF≈4）—— 这正是「字幕依旧很慢」的根源。
        所以原则是：GPU 放得下就坚决上 GPU；放不下就换小一号模型留在 GPU，
        而不是掉到 CPU 跑大模型；只剩 CPU 时 large 直接换 small 保实时。"""
        size = self.model_size
        big = size.startswith("large")
        user_ct = self.compute_type if self.compute_type != "auto" else None
        if dev != "cuda":
            # CPU 上 large-v3 毫无可用性（一句话 15 秒），直接换 small 保实时
            return [("small" if big else size, "cpu", user_ct or "int8")]
        free = _gpu_free_gb()
        chain: List[tuple] = []
        if big:
            if free is None or free >= 2.8:
                chain.append((size, "cuda", user_ct or "int8"))
            if free is None or free >= 1.5:
                chain.append(("medium", "cuda", "int8"))
            chain.append(("small", "cuda", "int8"))
        else:
            if user_ct:
                chain.append((size, "cuda", user_ct))
            else:
                chain.append((size, "cuda",
                              "float16" if (free is None or free >= 1.8) else "int8"))
            chain.append((size, "cuda", "int8"))
            chain.append(("small", "cuda", "int8"))
        seen, out = set(), []
        for c in chain:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> None:
        # 加锁：预热线程 / 开始翻译 / 手动换模型可能并发进来，
        # 不加锁会同时加载两份模型，显存直接翻倍。
        with self.lock:
            if self.model is not None and self.model_size == getattr(self, "_loaded_size", None):
                return
            from faster_whisper import WhisperModel

            # 模型已经在本地缓存了就别联网：本机有系统代理，联网检查常被拦成
            # ProxyError 502，反而让"明明有模型"也加载失败。
            if self._cached():
                os.environ["HF_HUB_OFFLINE"] = "1"

            dev = self.device
            if dev == "auto":
                try:
                    import ctranslate2
                    dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except Exception:
                    dev = "cpu"

            asked = self.model_size
            chain = self._plan(dev)
            if on_progress and chain[0][0] != asked:
                free = _gpu_free_gb()
                on_progress(
                    f"显存空闲仅 {free:.1f}GB，{asked} 放不下 —— 先用 {chain[0][0]} 保速度"
                    f"（关掉占显存的程序后重启可回到 {asked}）")
            last_exc: Optional[Exception] = None
            for size, device, ct in chain:
                if on_progress:
                    on_progress(f"正在加载 Whisper 模型 {size}（{device}/{ct}）…")
                try:
                    self.model = WhisperModel(size, device=device, compute_type=ct)
                    self._smoke()          # 真的跑一次推理，别让"缺 DLL 的 GPU"蒙混过关
                    self.model_size = size
                    self._loaded_size = size
                    self.dev_info = f"{device}/{ct}"
                    # GPU 上一次把多个语音块并行送进解码器（batch_size），能显著压短
                    # 长段/长句的识别耗时（仅新版 faster-whisper 的 transcribe 支持）；
                    # CPU 没有意义，保持 1。
                    self.batch_size = 8 if device == "cuda" else 1
                    if on_progress:
                        note = "" if size == asked else \
                            f"（原选 {asked}，因显存/设备限制自动降级；重启程序可重试原模型）"
                        on_progress(f"模型 {size} 就绪（{self.dev_info}）{note}")
                    return
                except Exception as exc:
                    last_exc = exc
                    self.model = None
                    if on_progress:
                        on_progress(f"{size}（{device}）不可用：{type(exc).__name__}，换下一档…")
            raise RuntimeError(f"所有加载组合都失败：{last_exc}")

    def _cached(self) -> bool:
        """本地是否已有该模型的缓存。"""
        home = os.environ.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface")
        hub = os.path.join(home, "hub", f"models--Systran--faster-whisper-{self.model_size}")
        return os.path.isdir(hub)

    def _smoke(self) -> None:
        """用极短静音跑一次推理：GPU 缺运行库时会在这里就暴露，而不是等到真正识别时。"""
        segs, _ = self.model.transcribe(
            np.zeros(1600, dtype=np.float32), language=self.language,
            beam_size=1, vad_filter=False,
        )
        list(segs)

    def unload(self) -> None:
        self.model = None
        self._loaded_size = None

    def _retry_on_oom(self, audio: np.ndarray, word_map: str = "",
                      roster: str = "", hotwords: str = "") -> Optional[str]:
        """运行期才发现 GPU 用不了（最典型：直播中途开了游戏，显存被占爆）。

        自救顺序：先试「小一号模型仍留在 GPU」（medium int8 约 0.9GB，
        游戏再占也常放得下），再试「small 落 CPU」。
        绝不再用 CPU 跑 large —— 实测一句话要 15 秒，字幕直接报废。
        每次会话只自救一次，避免反复重载模型。"""
        if getattr(self, "_oom_retried", False):
            return None
        self._oom_retried = True
        from faster_whisper import WhisperModel
        for size, device, ct in (("medium", "cuda", "int8"), ("small", "cpu", "int8")):
            try:
                self.model = WhisperModel(size, device=device, compute_type=ct)
                self._smoke()
                self.model_size = size
                self._loaded_size = size
                self.dev_info = f"{device}/{ct}（运行期自动降级）"
                self.batch_size = 8 if device == "cuda" else 1
                self.last_error = ""
                return self.transcribe(audio, word_map=word_map,
                                       roster=roster, hotwords=hotwords)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        return None

    # -- 识别
    @staticmethod
    def _normalize_gain(audio: np.ndarray) -> np.ndarray:
        """音量归一化：只提不降。

        真实场景里系统音量常常不高（或回环采集电平偏低），此时 Whisper 会把整段
        判成「没人说话」→ 一个字都不出。实测把音频压到 15% 音量后，12 句里有 7 句
        完全无输出（平均错误率 69.5%）；先做响度归一化后降到 21.5%，
        而正常音量的音频**不加增益、保持原样**（实测误加增益反而轻微变差）。
        """
        if audio is None or len(audio) == 0:
            return audio
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < 1e-5:                       # 几乎是纯静音，别放大噪声
            return audio
        gain = min(0.055 / rms, 12.0)        # 目标 RMS≈0.055（约 -25dBFS）
        if gain <= 1.05:                     # 本来就不小，原样不动
            return audio
        out = audio * gain
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 0.98:                      # 放大后削峰，防爆音
            out = out / peak * 0.98
        return out.astype(np.float32)

    def transcribe(self, audio: np.ndarray, word_map: str = "",
                   roster: str = "", hotwords: str = "") -> Optional[str]:
        if self.model is None or audio is None or len(audio) < 1600:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        audio = self._normalize_gain(audio)
        audio = np.clip(audio, -1.0, 1.0)
        # 去掉首尾可能的爆音
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio))) or 1.0
        if peak > 0.98:
            audio = audio / peak * 0.98

        try:
            with self.lock:
                # batch_size 只在 faster-whisper >= 0.9 才支持；老版本没有这个参数，直接传会
                # TypeError 导致每句都识别失败。按版本能力决定要不要加（探测一次后缓存）。
                kwargs = dict(
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    # 口音 / 含混语音：0.0 解不出时自动回退 0.4 / 0.6（faster-whisper 的温度回退机制），
                    # 带口音、背景嘈杂时的识别率明显提升；干净音频不受影响（仅在回退时才多花时间）。
                    temperature=[0.0, 0.4, 0.6],
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                )
                if _batch_supported():
                    kwargs["batch_size"] = self.batch_size
                # 热词：把词表下发给解码器，从「识别源头」提高人名命中率（default 关闭，
                # 见 config.enable_hotwords；同走 prompt，实测静音不复述但仍有风险）。
                if hotwords and _hotwords_supported():
                    kwargs["hotwords"] = hotwords
                segments, _info = self.model.transcribe(audio, **kwargs)
                parts: List[str] = []
                for seg in segments:
                    try:
                        # 这几个门槛别再往下压了：试过 -0.95 / 2.0，会把正常句子一起杀掉，
                        # 实测直接变成「一句字幕都不出」。现在的取值是能保住正常句子的边界。
                        if getattr(seg, "no_speech_prob", 0.0) > 0.7:
                            continue
                        if getattr(seg, "avg_logprob", 0.0) < -1.2:
                            continue
                        if getattr(seg, "compression_ratio", 1.0) > 2.4:
                            continue
                    except Exception:
                        pass
                    t = (seg.text or "").strip()
                    if t:
                        parts.append(t)
        except Exception as exc:
            # 绝不静默吞掉：记下来，并按「GPU 小模型 → CPU 小模型」自救一次
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self._retry_on_oom(audio, word_map, roster, hotwords)

        text = " ".join(parts).strip()
        text = self._clean(text)
        if word_map:
            # 1) 精确替换表：用户写死的「误=正」，优先级最高
            text = self.apply_word_map(text, word_map)
        if roster:
            # 2) 录音近修正：只录正确人名，按读音把听错的假名片段归一（零模型开销）
            from .namefix import apply_names
            text = apply_names(text, roster)
        if not self._accept(text):
            return None
        return text

    @staticmethod
    def parse_word_map(raw: str) -> "dict":
        """把「错=对」多行文本解析成 {错误写法: 正确写法}。

        支持每行 `误=正` 或 `误=正 # 注释`，空行/无等号行忽略。
        返回只读字典，供 apply_word_map 做确定性替换。
        """
        out = {}
        for line in (raw or "").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            wrong, right = line.split("=", 1)
            wrong, right = wrong.strip(), right.strip()
            if wrong and right:
                # 长的放前面匹配，避免短前缀先抢走子串（如「ペコラ」先于「兎田ぺこら」）
                out[wrong] = right
        return dict(sorted(out.items(), key=lambda kv: -len(kv[0])))

    @staticmethod
    def apply_word_map(text: str, raw: str) -> str:
        """识别完成后修正听错的人名/梗：纯字符串替换，绝不进模型、不会回声。"""
        if not raw or not text:
            return text
        for wrong, right in WhisperASR.parse_word_map(raw).items():
            if wrong in text:
                text = text.replace(wrong, right)
        return text

    # -- 清洗与过滤
    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace(" ", "").replace("　", "").strip()
        # 去掉 Whisper 常见的句内重复
        for _ in range(2):
            n = len(text)
            if n >= 8 and n % 2 == 0 and text[: n // 2] == text[n // 2:]:
                text = text[: n // 2]
        # 拖长音/哼唱会识别成一长串同一个字（すいちゅうううううううう），压回去
        text = re.sub(r"(.)\1{3,}", r"\1", text)
        # ASR 偶发把同一句吐两遍（「〜ください、〜ください」）：折叠近似重复子句，
        # 否则译文会跟着重复一遍（实测截图里出现过「拜托你相当一下嘛，拜托你相当一下嘛」）
        from .ja_norm import dedupe_clauses
        text = dedupe_clauses(text)
        return text

    @staticmethod
    def _looks_not_japanese(text: str) -> bool:
        """中日混合乱码检测：Whisper 跑偏到中文分支时会吐出一串汉字、几乎不带假名
        （实测出现过「さあ所有人開啟進步甩帽子」「ウエルラ結論ブラック」）。
        日语口语里几乎不可能汉字多到这个程度还没假名，所以按「汉字够多 + 假名太少」判。
        不用字形黑名单——幻觉的汉字繁简都有，列不全。
        """
        kana = sum(1 for c in text if "\u3040" <= c <= "\u30ff")      # 平假名 + 片假名
        kanji = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if kanji < 2 or not text:
            return False
        return kana / len(text) < 0.2

    def reset_repeat_guard(self) -> None:
        """清掉「连续相同文本」幻觉护栏的状态。

        这个护栏是**跨调用**的状态（`_last_text` / `_repeat`）：连续两次吐出同一句时
        第三次会被判成幻觉丢掉。对直播是合理的（Whisper 爱循环复读），但**离线评测**
        里把同一段音频喂三遍就会误判——2026-10-13 排查采集链路时正是被它坑了一次，
        表现为「第三遍永远是空字符串」。评测脚本换样本/换设置前先调它。
        """
        self._last_text = ""
        self._repeat = 0

    def _accept(self, text: str) -> bool:
        if len(text) < 2:
            return False
        if text in HALLUCINATIONS:
            return False
        if self.language == "ja" and self._looks_not_japanese(text):
            return False
        if text == self._last_text:
            self._repeat += 1
            if self._repeat >= 2:      # 连说两次同样的短句有可能是真的，第三次才认为是幻觉
                self._repeat = 0
                return False
        else:
            self._repeat = 0
        self._last_text = text
        # 单字符高频重复（あああああ）
        if len(set(text)) <= 1 and len(text) > 2:
            return False
        return True
