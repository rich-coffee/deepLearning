# 深度学习基础 — RNN / GRU / LSTM 实时生理信号预测系统

> **从理论到可视化：基于 PyTorch 实现的经典循环神经网络架构，使用真实心电数据训练，并通过浏览器端示波器实时呈现预测效果。**

---

## 💡 项目初衷

在深入学习和研究具身智能与 SLAM 系统的过程中，我对深度学习的基础框架产生了浓厚兴趣。为了从根本上理解循环神经网络的运作机制，我创建了此仓库，从零实现 RNN、GRU 和 LSTM 三种经典架构，并在真实生理信号上进行训练、对比与可视化诊断。

---

## 🔬 核心内容

### 已复现的经典模型

- **Vanilla RNN（Elman 网络）** — 最朴素的循环架构，作为理解梯度消失问题的基线模型
- **GRU（门控循环单元）** — 比 LSTM 更轻量的门控方案，在效率与记忆能力之间取得平衡
- **LSTM（长短期记忆网络）** — 序列建模的主力架构，通过遗忘门/输入门/输出门显式控制长程依赖

三种架构共用一套**统一模型工厂**（[models.py](models.py) 中的 `SleepRNNDemo`），仅需修改一个字符串参数即可切换单元类型，而所有其他超参数保持不变——从而实现严格受控的架构对比实验。

### 使用的框架与技术栈

| 层级 | 技术 | 说明 |
|---|---|---|
| **深度学习框架** | PyTorch | 动态计算图，调试直观，GPU 支持完善 |
| **数据管线** | `wfdb` + NumPy | 原生读取 PhysioNet 标准格式的 MIT-BIH 心电数据 |
| **实时推流** | WebSocket（`websockets`） | 服务端→浏览器低延迟遥测推送（~30 FPS） |
| **前端可视化** | D3.js v7（本地副本） | 高性能 SVG 渲染，零外部网络依赖 |
| **页面布局** | 原生 HTML5 / CSS3 Flexbox | 自适应示波器 + 工业 SCADA 风格遥测面板 |

### 训练特性

- ✅ 使用 MIT-BIH Record 100 全部约 65 万个真实心电采样点进行全量训练
- ✅ 75% 重叠滑窗采样策略——模型在信号的每一种相位对齐上都有充分样本
- ✅ 按时间顺序 80/20 划分训练/验证集——杜绝未来信息泄漏
- ✅ 小批量训练（兼容 GPU）+ 梯度裁剪（max norm = 1.0）
- ✅ ReduceLROnPlateau 学习率自动衰减
- ✅ 早停机制（patience 可配置）
- ✅ 断点续训——训练中断后自动从上次 epoch 恢复，适应 HPC 集群资源回收
- ✅ 按验证损失导出最佳模型，而非仅保留最后一轮权重

---

## 🖼️ 可视化

### 训练结果对比

| RNN | GRU | LSTM |
|---|---|---|
| ![RNN](MIT-BIH_data/result/MIT-BIH数据RNN训练结果.png) | ![GRU](MIT-BIH_data/result/MIT-BIH数据GRU训练结果.png) | ![LSTM](MIT-BIH_data/result/MIT-BIH数据LSTM训练结果.png) |

> *运行 `python train.py` 并将 `MODEL_TYPE` 分别设为 `'RNN'`、`'GRU'`、`'LSTM'` 进行训练，然后将对比图保存至 `MIT-BIH_data/result/` 目录即可在此处展示。*

### 示波器演示

![示波器演示](MIT-BIH_data/result/oscilloscope_demo.gif)

> *基于浏览器的示波器以约 30 FPS 的帧率并排渲染 CH1（黄色，地面真值/真实标签）与 CH2（绿色虚线，神经网络预测）。右侧的 SCADA（监视控制与数据采集）面板则实时显示误差幅值与链路延迟。*

### 训练 Loss 曲线

运行训练脚本后，控制台会逐轮输出 train/val loss。如需生成论文级图表，可在 `train.py` 末尾添加或交互式运行：

```python
import matplotlib.pyplot as plt
# （将每轮的 train_loss / val_loss 记录到列表中，然后：）
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.legend(); plt.title(f'{MODEL_TYPE} — MIT-BIH 心电预测 Loss 曲线')
plt.savefig(f'MIT-BIH_data/result/MIT-BIH_{MODEL_TYPE}_training_curve.png', dpi=150)
```

**立即体验：**
1. `python train.py` — 训练模型（或直接使用 `MIT-BIH_data/weight/` 中的预训练权重）
2. `python sever.py` — 启动 WebSocket 推理服务器
3. 用浏览器打开 `index.html` — 波形即刻开始渲染

---

## 📂 项目结构

```
deepLearning/
├── train.py                        # 训练脚本：MIT-BIH 全量数据 → RNN/GRU/LSTM
├── models.py                       # 统一模型工厂（训练与推理共用）
├── data_loader.py                  # 数据管线：信号读取 + 滑窗采样 + DataLoader
├── sever.py                        # WebSocket 推理服务器（边缘计算遥测终端）
├── index.html                      # D3.js 示波器 + SCADA 工业遥测面板
├── d3.v7.min.js                    # D3.js v7 本地副本（273 KB，无需 CDN）
├── requirements.txt                # Python 依赖清单
├── 100.dat / 100.hea / 100.atr     # MIT-BIH 心律失常数据库 Record 100
└── MIT-BIH_data/
    ├── weight/                     # 预训练权重（.pth 文件）
    │   ├── sleep_rnn_weights.pth
    │   ├── sleep_gru_weights.pth
    │   └── sleep_lstm_weights.pth
    └── result/                     # 训练曲线与架构对比图
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 现代浏览器（Chrome / Edge / Firefox）
- （可选）NVIDIA GPU + CUDA 12.4，用于加速训练

### 1. 安装依赖

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# CPU 版（无需 NVIDIA 显卡）
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# GPU 版（CUDA 12.4，推荐用于训练）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

修改 [train.py](train.py) 顶部的 `MODEL_TYPE` 切换架构：
```python
MODEL_TYPE = 'LSTM'   # 可选: 'RNN' | 'GRU' | 'LSTM'
```

全量数据训练在 CPU 上约需 3–5 分钟，GPU 环境下显著更快。

### 3. 启动推理服务器

```bash
python sever.py
```

成功启动后输出：
```
=============================================
  [SYS] 工业级边缘计算遥测终端已启动
  [SYS] 当前挂载计算核心: LSTM 神经网络
  [SYS] 数据来源: MIT-BIH 心律失常数据库 Record 100 (真实生理信号)
  [SYS] 监听端口: ws://127.0.0.1:8765
=============================================
```

### 4. 打开前端监控面板

直接用浏览器打开 `index.html`。页面自动通过 WebSocket 连接服务器，状态指示灯从 **SYS OFFLINE** 变为 **DATA LINK ACTV**，双通道波形与遥测数据开始实时更新。

> D3.js 已本地化（`d3.v7.min.js`），WebSocket 连接本机 `localhost`——**无需任何网络连接**即可打开面板。

---

## 🧪 运行单元测试

每个 Python 模块均包含 `__main__` 自检代码：

```bash
python data_loader.py   # 验证信号加载、滑窗切割、DataLoader 管线
python models.py        # 验证 RNN/GRU/LSTM 实例化与前向传播 shape
```

---

## 🎯 科研意义

### 为什么需要实时可视化

标量损失函数（MSE、MAE）用一个数字概括模型性能，却掩盖了预测误差的**动力学特征**。一个低 MSE 的模型仍可能存在：

- **相位滞后** — 预测波形正确但整体在时间轴上偏移
- **幅值漂移** — 预测包络相对于真实值逐渐衰减或发散
- **瞬态失明** — 在波形突变（如心律失常搏动）时失效，而在规则节律上表现良好

示波器面板让这些失效模式**一目了然**，为定量评估提供了不可或缺的定性调试手段。

### 为什么使用统一模型工厂

在保持所有超参数一致的前提下，仅改变循环单元类型（`RNN` → `GRU` → `LSTM`），研究者可以**分离出循环架构本身**对预测精度的影响。MLP 输出头（hidden → 16 → 1）同时作为消融实验点：将其替换为单层 `nn.Linear`，即可量化额外非线性层的贡献。

### 为什么选用真实生理数据基准

MIT-BIH 是生物医学信号处理领域引用量最高的公开数据集之一。在真实心电数据上训练——而非合成正弦波——确保学到的动态特性反映实际生理信号特征，使研究结果对异常检测、信号压缩、去噪等实际应用具有参考价值。

---

## 📚 数据来源

数据文件 `100.dat`、`100.hea`、`100.atr` 来自 **MIT-BIH 心律失常数据库**（[PhysioNet](https://physionet.org/content/mitdb/1.0.0/)），是生理信号处理领域最权威的公开基准数据集之一。Record 100 包含约 30 分钟双导联动态心电图，采样率 360 Hz，共约 65 万个采样点。本项目提取 MLII 导联并进行 Z-score 标准化。

---

## 📝 注意事项

- 请先启动 `sever.py`，再打开 `index.html`，否则页面会一直显示 **SYS OFFLINE**
- `index.html` 使用本地 `d3.v7.min.js`，**零外部网络依赖**
- 训练需要项目根目录下存在 MIT-BIH 数据文件（`100.dat`、`100.hea`、`100.atr`）
- 全量数据训练在 CPU 上约需 3–5 分钟，建议在有 GPU 的环境下运行以获得最佳体验
- 切换架构时，请确保 `train.py` 和 `sever.py` 中的 `MODEL_TYPE` 保持一致
- 推荐使用虚拟环境运行，避免依赖冲突
