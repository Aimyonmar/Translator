# -*- coding: utf-8 -*-
"""单视频 URL 取音频 / 取字幕（可选依赖 yt-dlp）。

边界：只处理用户本人提供的一个视频链接，用于观看辅助、转写与评测；
不做批量抓取，不绕过登录墙/DRM，不下载会员专属清晰度。没有登录态时只提示
用户可自备 cookies（放到 .runtime/cookies.txt），程序不伪造身份。
"""
from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".runtime")

INSTALL_HINT = ("未安装 yt-dlp。安装命令：\n"
                ".runtime\\venv\\Scripts\\python.exe -m pip install yt-dlp")


class UrlError(Exception):
    """kind: missing/url/login/geo/http/network/cancel/subs/other"""

    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint


class Cancelled(Exception):
    pass


def cookies_file() -> str:
    """固定路径的 cookies：存在就用，不存在就当游客。"""
    p = os.path.join(RUNTIME_DIR, "cookies.txt")
    return p if os.path.isfile(p) else ""


def have_ytdlp() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False


def _yt():
    try:
        import yt_dlp
        return yt_dlp
    except Exception:
        raise UrlError("missing", "缺少可选依赖 yt-dlp", INSTALL_HINT)


def _base_opts(proxy: str = "") -> Dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
    }
    if proxy:
        opts["proxy"] = proxy
    ck = cookies_file()
    if ck:
        opts["cookiefile"] = ck
    return opts


def classify(exc: Exception, cancelled: bool = False) -> UrlError:
    if cancelled:
        return UrlError("cancel", "已取消", "")
    text = str(exc)
    low = text.lower()
    if "unsupported url" in low or "is not a valid url" in low:
        return UrlError("url", "链接无效或站点不受支持", "检查链接是否完整（以 http/https 开头）。")
    if "login" in low or "cookies" in low or "private" in low or "会员" in text:
        return UrlError("login", "该内容需要登录才能读取",
                        "可把自己的 cookies 导出为 .runtime/cookies.txt 后再试；"
                        "程序不会绕过登录墙。")
    if "not available in your country" in low or "geo" in low or "地区" in text:
        return UrlError("geo", "该视频在当前地区不可用", "地区限制无法通过程序绕过。")
    if "http error 404" in low or "video unavailable" in low or "不存在" in text:
        return UrlError("http", "视频不存在或已删除", "确认链接在浏览器里能正常打开。")
    if "timed out" in low or "connection" in low or "network" in low or "resolve" in low:
        return UrlError("network", "网络连接失败", "检查网络；国内站点可直接连，境外站点可能需要代理。")
    return UrlError("other", "获取失败：%s" % text[:200], "")


def probe(url: str, proxy: str = "") -> Dict:
    """只读探测：标题、时长、可用的字幕/自动字幕语言。"""
    yt = _yt()
    try:
        with yt.YoutubeDL(_base_opts(proxy)) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise classify(exc)
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    def langs(d):
        return [{"lang": k, "formats": [f.get("ext") for f in v][:4]}
                for k, v in d.items()]
    return {
        "title": info.get("title") or "",
        "duration": float(info.get("duration") or 0.0),
        "id": info.get("id") or "",
        "webpage_url": info.get("webpage_url") or url,
        "uploader": info.get("uploader") or "",
        "subtitles": langs(subs),
        "automatic": langs(auto),
        "need_login": not bool(subs or auto),
    }


def download_audio(url: str, folder: str, proxy: str = "",
                   on_progress: Optional[Callable[[int, str], None]] = None,
                   should_stop: Optional[Callable[[], bool]] = None
                   ) -> Tuple[str, Dict]:
    """只取音频流（bestaudio），返回 (文件路径, info)。取消时抛 Cancelled。"""
    yt = _yt()
    os.makedirs(folder, exist_ok=True)

    def hook(d):
        if should_stop and should_stop():
            raise Cancelled()
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            pct = int(got * 100 / total) if total else 0
            on_progress(min(99, pct), d.get("_percent_str", "").strip())

    opts = _base_opts(proxy)
    opts.update({
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder, "%(id)s.%(ext)s"),
        "progress_hooks": [hook],
        "overwrites": True,
    })
    try:
        with yt.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
    except Exception as exc:
        raise classify(exc, cancelled=bool(should_stop and should_stop()))
    if not os.path.isfile(path):
        raise UrlError("other", "下载完成但找不到音频文件", "检查临时目录权限。")
    if on_progress:
        on_progress(100, "100%")
    return path, info


def download_subtitles(url: str, folder: str, proxy: str = "",
                       should_stop: Optional[Callable[[], bool]] = None
                       ) -> List[str]:
    """下载平台字幕（人工优先，其次自动字幕），返回本地 .srt 路径列表。"""
    yt = _yt()
    os.makedirs(folder, exist_ok=True)
    opts = _base_opts(proxy)
    opts.update({
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "srt/best",
        "subtitleslangs": ["zh-Hans", "zh-CN", "zh", "zh-Hant", "en", "ja"],
        "outtmpl": os.path.join(folder, "%(id)s.%(ext)s"),
    })
    try:
        with yt.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise classify(exc, cancelled=bool(should_stop and should_stop()))
    if should_stop and should_stop():
        raise UrlError("cancel", "已取消", "")
    files = []
    for name in os.listdir(folder):
        if name.startswith(info.get("id") or "") and name.endswith((".srt", ".vtt")):
            files.append(os.path.join(folder, name))
    if not files:
        raise UrlError("subs", "该视频没有可读取的人工/AI 字幕",
                       "可改用「下载并转写」走语音识别；若浏览器里能看到字幕，"
                       "多半需要登录，可提供 .runtime/cookies.txt。")
    return sorted(files)


_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
                 r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def _sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def read_srt(path: str) -> List[Tuple[float, float, str]]:
    """SRT/VTT → [(起, 止, 文本)]；VTT 只取带时间轴的行。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().replace("\r\n", "\n")
    out = []
    cur = None
    buf: List[str] = []
    for line in raw.split("\n"):
        m = _TS.search(line)
        if m:
            if cur and buf:
                text = " ".join(x.strip() for x in buf if x.strip())
                if text:
                    out.append((cur[0], cur[1], text))
            g = m.groups()
            cur = (_sec(*g[:4]), _sec(*g[4:]))
            buf = []
        elif not line.strip():
            if cur and buf:
                text = " ".join(x.strip() for x in buf if x.strip())
                if text:
                    out.append((cur[0], cur[1], text))
            cur, buf = None, []
        elif cur is not None:
            buf.append(line)
    if cur and buf:
        text = " ".join(x.strip() for x in buf if x.strip())
        if text:
            out.append((cur[0], cur[1], text))
    return out
