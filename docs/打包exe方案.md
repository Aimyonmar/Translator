# exe 打包方案（暂不采用，备用）

本文回答一个问题：**为什么 v0.9.0 发源码包，而不是一个双击就能跑的 exe**，
以及哪天真要打包时该怎么做。

## 一、结论

v0.9.0 **不发 exe**。源码包 + `安装依赖.bat` 已经做到「解压后双击两次」，
而 exe 的代价（体积、启动、误报、维护）在现阶段换不来相称的收益。

## 二、代价清单（按影响排序）

| 代价 | 具体表现 | 量化参考 |
| --- | --- | --- |
| **体积** | PySide6（约 200 MB）+ CTranslate2（约 90 MB）+ onnxruntime（约 15 MB，VAD 用）+ Python 运行时，单文件压缩后仍 400 MB 起 | 本机 venv 2.56 GB，打包去掉 .py 与缓存后约 450~700 MB |
| **首次启动慢** | `--onefile` 每次启动都要把内容解到 `%TEMP%\_MEIxxxx`，冷启动 5~15 秒；`--onedir` 快但目录里几十个文件 | onefile 解压 ≈ 体积 / 磁盘速度 |
| **杀软误报** | 未签名的 PyInstaller 单文件常被 Defender/360 报 `Trojan:Win32/Wacatac` 之类，需用户手动放行 | 无签名证书时高概率 |
| **模型仍要另下** | 模型 0.5~3 GB 不能塞进 exe（会到 1.5~4 GB，且更新模型要重发大包） | 与源码包方案相同 |
| **隐藏依赖多** | CTranslate2 的 CUDA/cuDNN、`pyaudiowpatch` 的 PortAudio DLL、Qt 插件（platforms/、imageformats/）都要显式收集 | 每个依赖都可能要写 hook |
| **维护成本** | 每次改代码都要重新打包、重新测安装、重新解释误报；CI 也得装完整工具链 | 发版耗时从分钟级涨到小时级 |
| **GPU 差异** | 有 GPU 的机器要用 CUDA 版 CTranslate2，无 GPU 要用 CPU 版，一个包难两头兼顾 | 要么发两个包，要么首次运行再补装 |

## 三、真要做的话（步骤草案）

1. 准备干净环境（不要在开发 venv 里打）：
   ```bat
   python -m venv .build\venv
   .build\venv\Scripts\pip install -r requirements.txt pyinstaller
   ```
2. 写 spec（比命令行参数可靠），关键点：
   - `hiddenimports`：`faster_whisper`、`ctranslate2`、`onnxruntime`、`av`、`pyaudiowpatch`、
     `sounddevice`、`PySide6.QtSvg`；
   - `datas`：`sounddevice` 的 `_sounddevice_data`（PortAudio DLL）、PySide6 的
     `plugins/platforms`、可选图标与字体；
   - 排除：`torch`（本项目未用，装了会白涨 2 GB）、`matplotlib`、`tkinter`。
3. 打包：
   ```bat
   .build\venv\Scripts\pyinstaller --noconfirm --clean --onedir --windowed ^
       --name VTuberTranslator main.py
   ```
   - **推荐 `--onedir`**：启动快、方便替换 `ui`/`core` 的 pyc；`--onefile` 只在分发便利性上占优。
4. 验证矩阵（至少要跑）：
   - 未装 Python 的干净 Windows 10/11（虚拟机）；
   - 有 GPU / 无 GPU 各一台；
   - 杀软开启状态下下载与首次启动；
   - 首次启动下载模型（exe 内仍需 `准备模型.py` 的逻辑，可做成 `--prefetch` 参数）。
5. 模型策略：exe 内**不打包模型**，首次启动仍走 `hf-mirror.com` 下载；界面里给进度。

## 四、什么时候值得改发 exe

满足任一条再考虑：

- 目标用户里「装 Python」这一步的流失率明显偏高（要数据，不靠感觉）；
- 拿到代码签名证书（消除误报）；
- 有 CI 能一条命令产出 exe 并跑完验证矩阵。

在那之前，源码包 + 脚本是性价比最高的形态。
