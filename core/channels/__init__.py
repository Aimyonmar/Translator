# -*- coding: utf-8 -*-
"""翻译通道层：静态定义、HTTP 会话、免费/自定义适配器、竞速与保活。

拆分自 core/translator.py：各模块只负责单一职责，且不反向 import Translator，
避免循环导入；对外入口仍是 core.translator.Translator。
"""
