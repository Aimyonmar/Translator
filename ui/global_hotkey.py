# -*- coding: utf-8 -*-
"""Windows 全局热键：ctypes 调 RegisterHotKey，Qt 原生事件过滤器收 WM_HOTKEY。

为什么不用第三方库：需求只覆盖 Windows，user32 两个函数足够，避免新增依赖。
线程模型：RegisterHotKey 在 GUI 线程调用，WM_HOTKEY 投递到该线程的消息队列，
QAbstractNativeEventFilter 在 Qt 事件循环里收到后再发 Qt 信号，由面板以
QueuedConnection 接回主线程；回调路径不直接碰任何控件。
"""
from __future__ import annotations

import sys
from typing import Tuple

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409

DEFAULT_HOTKEY = "Ctrl+Alt+M"
_HOTKEY_ID = 0x4D31

_SPECIAL_VK = {
    Qt.Key_Space: 0x20, Qt.Key_Tab: 0x09, Qt.Key_Backspace: 0x08,
    Qt.Key_Return: 0x0D, Qt.Key_Enter: 0x0D, Qt.Key_Escape: 0x1B,
    Qt.Key_Insert: 0x2D, Qt.Key_Delete: 0x2E,
    Qt.Key_Home: 0x24, Qt.Key_End: 0x23,
    Qt.Key_PageUp: 0x21, Qt.Key_PageDown: 0x22,
    Qt.Key_Left: 0x25, Qt.Key_Up: 0x26, Qt.Key_Right: 0x27, Qt.Key_Down: 0x28,
    Qt.Key_Print: 0x2C, Qt.Key_Pause: 0x13,
}


def parse_hotkey(text: str) -> Tuple[int, int]:
    """把 "Ctrl+Alt+M" 解析成 (Win32 修饰键, 虚拟键码)。无效时抛 ValueError。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("热键为空")
    seq = QKeySequence(text)
    if seq.isEmpty():
        raise ValueError("热键格式无效：%s" % text)
    combo = seq[0]
    key = combo.key()
    mods = combo.keyboardModifiers()
    win_mods = 0
    if mods & Qt.ControlModifier:
        win_mods |= MOD_CONTROL
    if mods & Qt.AltModifier:
        win_mods |= MOD_ALT
    if mods & Qt.ShiftModifier:
        win_mods |= MOD_SHIFT
    if mods & Qt.MetaModifier:
        win_mods |= MOD_WIN
    # 只有 Shift 或没有修饰键会把普通输入吞掉，禁止注册
    if not (win_mods & (MOD_CONTROL | MOD_ALT | MOD_WIN)):
        raise ValueError("全局热键至少要含 Ctrl、Alt 或 Win")

    if Qt.Key_A <= key <= Qt.Key_Z:
        vk = ord("A") + (key - Qt.Key_A)
    elif Qt.Key_0 <= key <= Qt.Key_9:
        vk = ord("0") + (key - Qt.Key_0)
    elif Qt.Key_F1 <= key <= Qt.Key_F24:
        vk = 0x70 + (key - Qt.Key_F1)
    elif key in _SPECIAL_VK:
        vk = _SPECIAL_VK[key]
    else:
        raise ValueError("不支持这个按键：%s" % QKeySequence(key).toString())
    return win_mods, vk


class _HotkeyEventFilter(QAbstractNativeEventFilter):
    def __init__(self, owner: "GlobalHotkey"):
        super().__init__()
        self.owner = owner

    def nativeEventFilter(self, event_type, message):
        if sys.platform == "win32" and self.owner.registered:
            try:
                import ctypes
                from ctypes import wintypes
                msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
                if msg.message == WM_HOTKEY and int(msg.wParam) == _HOTKEY_ID:
                    # 只发信号，不碰控件；面板用 QueuedConnection 接回主线程
                    self.owner.sig_activated.emit()
                    return True, 0
            except Exception:
                pass
        return False, 0


class GlobalHotkey(QObject):
    """注册失败要能说清原因；unregister() 可重复调用。"""

    sig_activated = Signal()
    sig_status = Signal(bool, str)      # (是否可用, 面向用户的原因)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.registered = False
        self.last_error = 0
        self.current = ""
        self._filter = _HotkeyEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installNativeEventFilter(self._filter)
        self._user32 = None
        if sys.platform == "win32":
            try:
                import ctypes
                self._user32 = ctypes.WinDLL("user32", use_last_error=True)
                self._user32.RegisterHotKey.argtypes = [
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
                self._user32.RegisterHotKey.restype = ctypes.c_int
                self._user32.UnregisterHotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
                self._user32.UnregisterHotKey.restype = ctypes.c_int
            except Exception:
                self._user32 = None

    def register(self, text: str) -> Tuple[bool, str]:
        """先注销旧的再注册新的；返回 (成功, 原因)，原因直接可上屏。"""
        self.unregister()
        self.current = (text or "").strip()
        if sys.platform != "win32":
            msg = "仅 Windows 支持全局热键，当前只保留应用内快捷键"
            self.sig_status.emit(False, msg)
            return False, msg
        if self._user32 is None:
            msg = "user32 不可用，全局热键未注册"
            self.sig_status.emit(False, msg)
            return False, msg
        try:
            import ctypes
            mods, vk = parse_hotkey(self.current)
            ok = self._user32.RegisterHotKey(None, _HOTKEY_ID, mods | MOD_NOREPEAT, vk)
            if not ok:
                self.last_error = int(ctypes.get_last_error())
                if self.last_error == ERROR_HOTKEY_ALREADY_REGISTERED:
                    msg = "热键被占用或无效（错误 %d），仅应用内可用" % self.last_error
                else:
                    msg = "热键注册失败（错误 %d），仅应用内可用" % self.last_error
                self.sig_status.emit(False, msg)
                return False, msg
        except ValueError as exc:
            self.last_error = 0
            msg = "%s，仅应用内可用" % exc
            self.sig_status.emit(False, msg)
            return False, msg
        except Exception as exc:
            self.last_error = -1
            msg = "热键注册异常：%s" % exc
            self.sig_status.emit(False, msg)
            return False, msg
        self.registered = True
        msg = "全局热键 %s 可用" % self.current
        self.sig_status.emit(True, msg)
        return True, msg

    def unregister(self) -> bool:
        """退出、关开关、换热键都要走这里，避免系统级残留。"""
        if not self.registered:
            return True
        ok = True
        try:
            if self._user32 is not None:
                ok = bool(self._user32.UnregisterHotKey(None, _HOTKEY_ID))
        except Exception:
            ok = False
        self.registered = False
        return ok

    def shutdown(self) -> None:
        self.unregister()
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeNativeEventFilter(self._filter)
            except Exception:
                pass
