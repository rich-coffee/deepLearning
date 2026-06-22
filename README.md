#  基于 RNN/GRU/LSTM 的生理信号预测与实时可视化系统

> **Neural Oscilloscope // SCADA Terminal**
>
> 使用 MIT-BIH 心律失常数据库真实心电信号训练循环神经网络，并通过 WebSocket + D3.js 构建工业级实时示波器监控面板。

---

##  项目结构

```
deepLearning/
 train.py                       # 训练脚本：MIT-BIH 数据  RNN/GRU/LSTM 模型（修改 MODEL_TYPE 切换架构）
 sever.py                       # WebSocket 服务器：加载模型，实时推送预测波形
 index.html                     # 前端面板：D3.js 工业示波器 + SCADA 遥测显示
 compare_models.py              # ★ 模型横向对比工具：一键训练三模型，生成 7 张对比图表 + JSON 数据
 d3.v7.min.js                   # ★ D3.js v7 本地副本（273 KB），消除对外网 CDN 的依赖
 requirements.txt               # Python 依赖清单
 .gitignore
 100.dat / 100.hea / 100.atr    # MIT-BIH 心律失常数据库 Record 100（真实心电信号）
 MIT-BIH_data/
    weight/
       sleep_rnn_weights.pth    # RNN 预训练权重
       sleep_gru_weights.pth    # GRU 预训练权重
       sleep_lstm_weights.pth   # LSTM 预训练权重
    result/
       MIT-BIH数据RNN训练结果.png
       MIT-BIH数据GRU训练结果.png
       MIT-BIH数据LSTM训练结果.png
    comparison/                 # ★ compare_models.py 输出目录
       loss_convergence.png     # 训练损失收敛曲线对比
       training_efficiency.png  # 训练效率对比
       convergence_speed.png    # 收敛速度分析
       inference_performance.png# 推理性能对比
       cost_performance_radar.png# 成本-性能雷达图
       prediction_waveform.png  # 测试集预测波形对比
       summary_table.png        # 综合汇总表
       comparison_data.json     # 结构化原始数据
 README.md
```

---

##  快速开始

### 环境要求

- Python 3.10+
- 现代浏览器（Chrome / Edge / Firefox）

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate   # Windows

# 安装 CPU 版 PyTorch + 其他依赖
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

修改 `train.py` 第 25 行的 `MODEL_TYPE` 变量可切换训练目标：

```python
MODEL_TYPE = 'GRU'   # 可选: 'RNN' | 'GRU' | 'LSTM'
```

### 3. 启动 WebSocket 推理服务器

```bash
python sever.py
```

成功启动后会显示：

```
=============================================
  [SYS] 工业级边缘计算遥测终端已启动
  [SYS] 当前挂载计算核心: GRU 神经网络
  [SYS] 数据来源: MIT-BIH 心律失常数据库 Record 100
  [SYS] 监听端口: ws://127.0.0.1:8765
=============================================
```

### 4. 打开前端监控面板

用浏览器直接打开 `index.html`，页面将自动通过 WebSocket 连接服务器，状态从 **SYS OFFLINE** 变为 **DATA LINK ACTV**，波形和遥测数据开始实时更新。

> **注意**: 页面使用的 D3.js 已本地化（`d3.v7.min.js`），无需联网即可打开。详见下方"前端依赖本地化"章节。

---

##  模型对比分析（compare_models.py）

### 功能概述

`compare_models.py` 是模型横向对比的自动化工具，一键完成以下工作流：

1. **重新训练** RNN / GRU / LSTM 三个模型（从头开始，各 200 epoch，固定随机种子确保可复现）
2. **全量记录**每个 epoch 的 loss 值、训练耗时
3. **推理评估**在独立测试集上计算 MSE / MAE / RMSE / 推理延迟
4. **生成 7 张对比图表 + 1 份 JSON 原始数据**

### 运行方式

```bash
python compare_models.py
```

### 输出图表说明

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 损失收敛曲线 | `loss_convergence.png` | 三模型 loss vs epoch 全图 + 尾部 60 epoch 放大，纵轴对数坐标 |
| 训练效率 | `training_efficiency.png` | 三栏：总训练耗时 / 单 epoch 平均耗时 / 可训练参数量 |
| 收敛速度 | `convergence_speed.png` | 达到各损失阈值（0.5→0.01）所需 epoch 数，越少越快 |
| 推理性能 | `inference_performance.png` | 双栏：单步推理延迟（μs）/ 测试集 MSE+MAE+RMSE |
| 成本-性能雷达图 | `cost_performance_radar.png` | 5 维归一化：训练速度 / 收敛速度 / 精度 / 轻量 / 推理速度 |
| 预测波形 | `prediction_waveform.png` | 三模型在测试集上的实际 vs 预测波形 + 误差填充带 |
| 综合汇总表 | `summary_table.png` | 所有指标一览，★ 标记每行最优值 |

### 典型结论（基于 MIT-BIH Record 100, 500 点训练）

| 指标 | RNN | GRU | LSTM | 最优 |
|------|-----|-----|------|------|
| 最终训练 Loss | 0.005921 | 0.004616 | **0.003499** | LSTM |
| 训练总耗时 | 6.1s | 19.8s | **0.8s** | LSTM |
| 参数量 | **1,153** | 3,393 | 4,513 | RNN（最轻） |
| 推理延迟 | **92.0 μs** | 116.6 μs | 207.1 μs | RNN（最快） |
| 测试集 MSE | 0.011382 | 0.016654 | **0.007757** | LSTM（最准） |

> **结论**: LSTM 预测精度和训练速度均最优，但参数量大、推理慢；RNN 最轻量、推理最快，适合边缘部署；GRU 各项居中。实际选择需根据部署场景（精度优先 or 延迟/资源优先）权衡。

---

##  前端依赖本地化

### d3.v7.min.js

将 D3.js v7 完整库（273 KB）下载到项目根目录 `d3.v7.min.js`，`index.html` 改为本地引用：

```html
<script src="d3.v7.min.js"></script>
```
> `d3.v7.min.js` 已加入 `.gitignore` 豁免（或直接纳入版本管理），确保项目在任何环境 clone 后即可使用。

---

##  自定义与亮点

### 一、前端视觉 — 工业级 SCADA 示波器面板

| 改动项 | 原始状态 | 修改后 | 设计意图 |
|--------|----------|--------|----------|
| 示波器背景 | 深灰 `#0a0a0a` | **纯黑 `#000000`** | 模拟专业示波器暗场显示 |
| 网格线 | 深灰实线 `#1a1a1a` | **白色虚线** `stroke-dasharray: 2,8` | 比波形线更稀疏，不干扰信号阅读 |
| CH1 波形线 | 蓝色 `#00ffff` 实线 | **黄色 `#ffdd00` 实线** | 工业信号标准中黄色代表通道一真实值 |
| CH2 波形线 | 红色 `#ff00ff` 虚线 | **绿色 `#00ff00` 虚线** | 绿色虚线代表预测值，直观区分真实/预测 |
| CH1 面板边框 | 蓝色 | **黄色** | 与波形颜色统一 |
| CH2 面板边框 | 红色 | **绿色** | 与波形颜色统一 |
| 右上角状态灯 | 无自动重连 | **断线 3 秒自动重连** | 模拟工业设备在线/离线状态机 |
| D3.js 依赖 | CDN 外链 | **本地文件** | 消除网络依赖，离线可用 |

> 所有修改处均以 `/* === MODIFIED === */` 注释标记在代码中，方便审阅。

### 二、数据路径 — 真实数据验证

```
 常见做法: 使用 np.sin() 等模拟数据训练
 我的做法: 读取本地 MIT-BIH 原始 .dat/.hea 文件
```

- **数据来源**: MIT-BIH 心律失常数据库 Record 100（PhysioNet 标准格式） 网址： https://physionet.org/content/mitdb/1.0.0/
- **读取方式**: 使用 `wfdb` 库直接解析本地 `.dat` / `.hea` 文件，无需联网
- **数据路径**: `100.dat` + `100.hea`（放在项目根目录）
- **信号通道**: 提取 MLII 导联（最常用于心律失常分析的导联）
- **预处理**: Z-score 标准化

训练脚本 [train.py](train.py) 的 `load_mitbih_data()` 函数展示了完整的本地数据加载流程。

### 三、模型权重路径 — 独立存储，支持横向对比

| 模型 | 训练输出路径 | 推理加载路径 |
|------|-------------|-------------|
| RNN | `MIT-BIH_data/weight/sleep_rnn_weights.pth` | 同左 |
| GRU | `MIT-BIH_data/weight/sleep_gru_weights.pth` | 同左 |
| LSTM | `MIT-BIH_data/weight/sleep_lstm_weights.pth` | 同左 |

每种模型权重独立存储，不会互相覆盖，方便切换对比不同架构的预测效果。训练脚本和推理服务器的路径完全一致。

### 四、模型统一工厂模式

一个 `SleepRNNDemo` 类通过 `cell_type` 参数同时支持 RNN / GRU / LSTM 三种架构，训练脚本 (`train.py`)、推理服务器 (`sever.py`)、对比分析工具 (`compare_models.py`) 中均使用相同的模型定义（代码不重复）。

### 五、工程健壮性

1. **信号捕获机制** — 捕获 `SIGTERM` / `SIGINT`，训练被 SLURM 或其他调度器 kill 时自动保存检查点
2. **断点续训** — 下次启动自动从上次中断的 epoch 继续，不浪费算力
3. **容错降级** — 服务器未找到权重文件时，自动使用随机权重继续运行，不中断服务
4. **WebSocket 自动重连** — 前端检测到断线后每 3 秒自动尝试重连

---

##  数据真实性声明

> **本项目的训练数据 100% 来自真实生理信号。**

- 数据文件 `100.dat`、`100.hea`、`100.atr` 来自 **MIT-BIH Arrhythmia Database**（麻省理工学院心律失常数据库，数据下载网址 'https://physionet.org/content/mitdb/1.0.0/'），是生理信号处理领域最权威的公开基准数据集之一
- Record 100 包含约 30 分钟的二导联动态心电图记录，采样率 360 Hz
- 训练结果截图保存在 `MIT-BIH_data/result/` 目录下，对比分析图表保存在 `MIT-BIH_data/comparison/` 目录下

---

##  技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch | RNN / GRU / LSTM 模型训练与推理 |
| 数据读取 | wfdb | 解析 PhysioNet 标准格式的 MIT-BIH 数据 |
| 实时通信 | WebSocket (websockets 库) | 服务端→前端低延迟数据推送 |
| 波形渲染 | D3.js v7（本地副本） | 150 点滑动窗口 SVG 波形实时绘制 |
| 前端布局 | Flexbox + CSS3 | 自适应示波器 + SCADA 面板布局 |
| 信号处理 | NumPy | 数据预处理与仿真信号生成 |
| 图表生成 | Matplotlib | 对比分析工具的可视化输出 |

---

##  操作说明

### 切换模型类型

1. 修改 `sever.py` 第 24 行：
   ```python
   MODEL_TYPE = 'GRU'  # 改为 'RNN' 或 'LSTM'
   ```

2. 重启 `python sever.py`

3. 刷新浏览器页面，观察不同模型架构的预测效果差异

### 横向对比三种模型

运行对比工具即可：

```bash
python compare_models.py
```

打开 `MIT-BIH_data/comparison/` 目录查看生成的 7 张对比图表。

### 查看训练结果

- 单次训练截图：打开 `MIT-BIH_data/result/` 目录
- 横向对比图表：打开 `MIT-BIH_data/comparison/` 目录

---

##  代码改动清单

| 文件 | 行号 | 改动内容 |
|------|------|----------|
| `train.py` | 1-20 | 新增模块级文档字符串，包含项目文件说明和对比分析工具引用 |
| `train.py` | 29 | 权重路径改为 `MIT-BIH_data\weight\` 独立目录 |
| `train.py` | 56-60 | 新增 `SleepRNNDemo` 类的完整 docstring |
| `train.py` | 83-84 | 新增 `forward()` 方法 docstring |
| `train.py` | 96-133 | 重写 `load_mitbih_data()` 的注释，强调真实数据 |
| `sever.py` | 1-20 | 新增模块级文档字符串，包含项目文件说明 |
| `sever.py` | 26 | 权重路径指向 `MIT-BIH_data\weight\`，与训练脚本一致 |
| `sever.py` | 31-32 | 新增模型类 docstring |
| `sever.py` | 70-72 | 补充仿真信号构造注释 |
| `sever.py` | 88-96 | 补充 JSON 数据帧字段注释 |
| `sever.py` | 110 | 新增数据来源打印信息 |
| `index.html` | 1-22 | HTML 头部注释重写：新增依赖说明（本地 d3.v7.min.js + 本地 WebSocket） |
| `index.html` | 25-26 | D3.js 引用改为本地文件，注释说明本地化原因 |
| `index.html` | 30-31 | 示波器背景改为纯黑，注释标记 |
| `index.html` | 57-59 | 面板边框颜色改为黄/绿，注释标记 |
| `index.html` | 71-73 | 数值颜色改为黄/绿 + 辉光效果，注释标记 |
| `index.html` | 77-78 | 网格改为白色虚线（dasharray 2,8），注释标记 |
| `index.html` | 83-96 | CH1 蓝→黄实线，CH2 红→绿虚线，注释标记 |
| `index.html` | 134-139 | JS 注释新增本地依赖与 WebSocket 本机连接说明 |
| `index.html` | 179-224 | WebSocket 代码注释重写，突出 SCADA 风格 |
| **NEW** `d3.v7.min.js` | — | D3.js v7 完整库本地副本（273 KB），替代 CDN 引用，国内无需翻墙即可使用 |
| **NEW** `compare_models.py` | — | 模型横向对比工具：训练三模型 + 生成 7 张对比图表 + JSON 数据，详见项目结构章节 |
| `requirements.txt` | — | 新增 `wfdb` 依赖及说明注释 |

---

##  注意事项

- 请先启动 `sever.py`，再打开 `index.html`，否则页面会一直显示 SYS OFFLINE
- `index.html` 使用本地 `d3.v7.min.js`，**无需网络连接**即可打开（已消除 CDN 依赖）
- 训练脚本需要本地存在 MIT-BIH 数据文件（`100.dat` + `100.hea` + `100.atr`）
- `compare_models.py` 会从头训练三个模型，耗时约 30-60 秒（CPU），请耐心等待
- 推荐使用虚拟环境运行，避免依赖冲突

---

*Deep Learning Final Project — 2026*
