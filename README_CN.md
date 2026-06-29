# 心电示波器 — 生理信号实时预测与可视化系统

> **RNN / GRU / LSTM 单步预测 + 浏览器端示波器监控**
>
> 使用 MIT-BIH 心律失常数据库 Record 100 全量数据训练循环神经网络，并通过 WebSocket + D3.js 将预测结果实时推送至浏览器端工业人机界面。

---

##  项目结构

```
deepLearning/
├── train.py                       # 训练脚本：MIT-BIH 全量数据 → RNN/GRU/LSTM 模型
├── models.py                      # 共享模型工厂：SleepRNNDemo（train.py / sever.py 共用）
├── data_loader.py                 # 数据管线：滑窗采样器 + PyTorch DataLoader
├── sever.py                       # WebSocket 推理服务器：向前端推送实时预测流
├── index.html                     # 前端面板：D3.js 示波器 + 遥测读数
├── d3.v7.min.js                   # D3.js v7 本地副本（273 KB），无需 CDN
├── requirements.txt               # Python 依赖清单
├── .gitignore
├── 100.dat / 100.hea / 100.atr    # MIT-BIH 心律失常数据库 Record 100（约 30 分钟心电）
└── MIT-BIH_data/
    ├── weight/
    │   ├── sleep_rnn_weights.pth    # RNN 预训练权重
    │   ├── sleep_gru_weights.pth    # GRU 预训练权重
    │   └── sleep_lstm_weights.pth   # LSTM 预训练权重
    └── result/
        ├── MIT-BIH数据RNN训练结果.png
        ├── MIT-BIH数据GRU训练结果.png
        └── MIT-BIH数据LSTM训练结果.png
```

---

##  快速开始

### 环境要求

- Python 3.10+
- 现代浏览器（Chrome / Edge / Firefox）

### 1. 安装依赖

```bash
# 创建并激活虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate   # Windows

# ---- CPU 版（无需 NVIDIA 显卡）----
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# ---- GPU 版（CUDA 12.4，推荐用于训练）----
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

修改 [train.py](train.py) 顶部的配置区可切换架构或调整超参数：

```python
MODEL_TYPE = 'LSTM'   # 可选: 'RNN' | 'GRU' | 'LSTM'
```

**训练特性：**

- **全量数据** — 使用 Record 100 全部约 65 万个采样点，滑窗生成数千条训练序列
- **小批量训练** — 默认 batch_size=64，兼容 GPU 并行
- **验证集评估** — 80/20 时间顺序切分，每轮评估泛化性能
- **早停机制** — 验证损失连续 20 轮不改善自动终止
- **学习率调度** — ReduceLROnPlateau 自动衰减学习率
- **梯度裁剪** — 最大范数 1.0，防止深层 RNN 梯度爆炸
- **断点续训** — 训练中断后自动从上次 epoch 恢复
- **最佳模型导出** — 按验证损失保存最优权重

### 3. 启动 WebSocket 推理服务器

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

直接用浏览器打开 `index.html`。页面自动通过 WebSocket 连接服务器，状态指示灯从 **SYS OFFLINE** 变为 **DATA LINK ACTV**，波形和遥测数据开始实时更新。

> D3.js 已本地化（`d3.v7.min.js`），无需联网即可打开面板。

---

##  使用指南

### 切换模型架构

1. 在 [sever.py](sever.py) 中设置 `MODEL_TYPE`（需与训练时一致）：
   ```python
   MODEL_TYPE = 'GRU'  # 'RNN' | 'GRU' | 'LSTM'
   ```
2. 重启 `python sever.py`。
3. 刷新浏览器页面，观察不同架构的预测效果差异。

### 调整训练超参数

修改 [train.py](train.py) 第 36–60 行的配置变量：

```python
MODEL_TYPE = 'LSTM'          # 模型架构
SEQ_LEN = 500                # 滑窗序列长度（采样点数）
BATCH_SIZE = 64              # 小批量大小
HIDDEN_SIZE = 64             # 隐藏层维度
NUM_LAYERS = 2               # 堆叠 RNN 层数
DROPOUT = 0.1                # Dropout 比率
LEARNING_RATE = 0.001        # 初始学习率（Adam）
TOTAL_EPOCHS = 300           # 最大训练轮数
EARLY_STOP_PATIENCE = 20     # 早停耐心值
```

### 运行单元测试

每个 Python 模块均包含 `__main__` 自检代码，可直接验证核心功能：

```bash
python data_loader.py   # 验证信号加载、滑窗切割、DataLoader
python models.py        # 验证 RNN/GRU/LSTM 实例化与前向传播
```

---

##  科研意义

### 为什么需要实时可视化

标量损失函数（MSE、MAE）用一个数字概括模型性能，却掩盖了预测误差的*动力学特征*。一个低 MSE 的模型可能仍然存在：

- **相位滞后** — 预测波形正确但整体在时间轴上偏移。
- **幅值漂移** — 预测包络相对于真实值逐渐衰减或发散。
- **瞬态失明** — 在波形突变（如心律失常搏动）时失效，而在规则节律上表现良好。

示波器面板使这些失效模式一目了然，为定量评估提供了互补的定性调试手段。

### 为什么需要全量数据训练

使用 Record 100 全部约 65 万个采样点（而非仅抽取几百个点）训练，模型学到的是真实的时序结构，而非对单一波形片段的死记硬背。75% 重叠的滑窗策略确保模型在信号的每一种相位对齐上都有充分样本，提升对时间偏移的鲁棒性。

### 为什么使用统一模型工厂

在保持所有超参数一致的前提下，仅改变循环单元类型（`RNN` → `GRU` → `LSTM`），研究者可以分离出循环架构本身对预测精度的影响。MLP 输出头（hidden → 16 → 1）同时作为消融实验点：将其替换为单层 `nn.Linear`，即可量化额外非线性层的贡献。

### 为什么选用真实生理数据基准

MIT-BIH 是生物医学信号处理领域引用量最高的公开数据集之一。在真实心电数据上训练——而非合成正弦波——确保学到的动态特性反映实际生理信号特征，使研究结果对异常检测、信号压缩、去噪等实际应用具有参考价值。

---

##  数据来源

> **本项目的训练数据 100% 来自真实生理信号。**

- 数据文件 `100.dat`、`100.hea`、`100.atr` 来自 **MIT-BIH 心律失常数据库**（[PhysioNet](https://physionet.org/content/mitdb/1.0.0/)），是生理信号处理领域最权威的公开基准数据集之一
- Record 100 包含约 30 分钟的双导联动态心电图，采样率 360 Hz，共约 65 万个采样点
- 训练使用全部 65 万个点；推理服务器使用一段留出的信号以展示泛化能力
- 提取 MLII 导联（最常用于心律失常分析的导联）并进行 Z-score 标准化

---

##  技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习 | PyTorch | RNN / GRU / LSTM 模型训练与推理 |
| 数据读取 | wfdb | 解析 PhysioNet 标准格式的 MIT-BIH 文件 |
| 实时通信 | WebSocket（websockets 库） | 服务端→前端低延迟数据推送 |
| 波形渲染 | D3.js v7（本地副本） | 150 点滑动窗口 SVG 实时绘制 |
| 前端布局 | Flexbox + CSS3 | 自适应示波器 + 遥测面板布局 |

---

##  注意事项

- 请先启动 `sever.py`，再打开 `index.html`，否则页面会一直显示 **SYS OFFLINE**
- `index.html` 使用本地 `d3.v7.min.js`，**无需网络连接**即可打开
- 训练需要项目根目录下存在 MIT-BIH 数据文件（`100.dat`、`100.hea`、`100.atr`）
- 全量数据训练在 CPU 上约需 3–5 分钟，建议在有 GPU 的环境下运行以获得最佳体验
- 推荐使用虚拟环境运行，避免依赖冲突
