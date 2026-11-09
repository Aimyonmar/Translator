# -*- coding: utf-8 -*-
"""HTTP 会话与代理：建 Session、套用代理、扫描本机常见代理端口。"""
from __future__ import annotations

import socket
import urllib.parse
from typing import Optional

import requests

from .constants import PROXY_PORTS, UA


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def apply_proxy(session: requests.Session, proxy: str) -> None:
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    else:
        session.proxies.clear()


def port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def detect_local_proxy(session: requests.Session) -> Optional[str]:
    """扫描本机常见代理端口，能用就返回代理地址。"""
    saved = dict(session.proxies)
    try:
        for port in PROXY_PORTS:
            if not port_open(port):
                continue
            url = f"http://127.0.0.1:{port}"
            try:
                r = requests.get(
                    "https://translate.googleapis.com/translate_a/single"
                    "?client=gtx&sl=ja&tl=zh-CN&dt=t&q=" + urllib.parse.quote("こんにちは"),
                    headers={"User-Agent": UA}, proxies={"http": url, "https": url}, timeout=7)
                if r.status_code == 200 and r.json():
                    return url
            except Exception:
                continue
    finally:
        session.proxies.clear()
        session.proxies.update(saved)
    return None
