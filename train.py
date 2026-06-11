"""
=============================================================================
 深度学习期末作业 — 基于 RNN/GRU/LSTM 的生理信号预测与实时可视化系统
 训练模块：使用 MIT-BIH 心律失常数据库（Record 100）的真实心电信号训练
=============================================================================
【自定义亮点】
  1. 真实数据源 — 读取本地 MIT-BIH 原始 .dat/.hea 文件，非模拟数据
  2. 模型统一工厂 — 一个类实现 RNN/GRU/LSTM 三种架构，便于横向对比
  3. 优雅中断机制 — 捕获 SIGTERM/SIGINT，训练被打断时自动保存检查点
  4. 断点续训 — 下次启动自动从上次中断处继续，防止资源浪费
  5. 独立权重路径 — 每种模型类型生成独立权重文件，避免互相覆盖
=============================================================================
"""
import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys

# ==========================================
# 教学配置：通过修改变量一键切换 RNN / GRU / LSTM
# 可选值: 'RNN' | 'GRU' | 'LSTM'
# ==========================================
MODEL_TYPE = 'GRU'

# 检查点与权重路径 — 每种模型独立存储，支持并行对比实验
CHECKPOINT_PATH = f'sleep_{MODEL_TYPE.lower()}_checkpoint.pth'
FINAL_WEIGHTS_PATH = f'MIT-BIH_data\\weight\\sleep_{MODEL_TYPE.lower()}_weights.pth'

# ----------------- 1. 信号捕获与安全性配置（HPC 共享集群友好） -----------------
# 当作业调度器（如 SLURM）回收资源时，捕获信号并优雅保存进度

def receive_signal(signum, frame):
    print(f"\n[警告] 收到资源回收信号 (Signal: {signum})！正在紧急保存当前进度...")
    global model, optimizer, epoch, loss
    save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)
    print(f"[退出] {MODEL_TYPE} 模型的进度已安全保存，程序优雅退出。")
    sys.exit(0)

signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

def save_checkpoint(epoch, model, optimizer, loss, path):
    """保存完整训练快照，支持断点续训"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"-> 检查点已保存至: {path}")

# ----------------- 2. 核心知识点：三类循环神经网络统一工厂模式 -----------------
class SleepRNNDemo(nn.Module):
    """
    统一模型工厂 — 一个 __init__ 实现三种架构的切换
    教学价值：直观对比 RNN（梯度消失）、GRU（门控简化）、LSTM（长程记忆）的差异
    """
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SleepRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 三种核心时序组件的 API 与内部结构对比
        if self.cell_type == 'RNN':
            # 标准 RNN：结构最简单，但存在梯度消失问题，适合短序列建模
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'GRU':
            # GRU：引入更新门 (update gate) 和重置门 (reset gate)，参数更少，训练更快
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'LSTM':
            # LSTM：引入遗忘门、输入门、输出门和独立细胞状态 (c)，擅长捕捉长程依赖
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("未知的网络类型！请选择 'RNN', 'GRU' 或 'LSTM'")

        # 统一的线性输出层：将隐藏状态映射为预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """前向传播 — LSTM 需要双隐状态 (h, c)，RNN/GRU 只需 h"""
        device = x.device
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)
        return self.fc(out)

# ----------------- 3. 真实生理数据加载（MIT-BIH 心律失常数据库） -----------------
def load_mitbih_data(file_path='100', sequence_length=500, device='cpu'):
    """
    【亮点】从本地 MIT-BIH 原始文件 (.dat/.hea) 读取真实心电信号
    使用 wfdb 库解析 PhysioNet 标准格式，提取 MLII 导联数据

    参数:
        file_path: 不含扩展名的记录名，例如 '100' → 读取 100.dat + 100.hea
        sequence_length: 单条训练序列长度（默认500个采样点）
        device: 计算设备（cpu / cuda）

    返回:
        x: Tensor (1, sequence_length, 1) — 输入序列
        y: Tensor (1, sequence_length, 1) — 目标序列（x 向后平移1步）
    """
    import wfdb

    # 读取本地 MIT-BIH 记录（.dat + .hea）
    record = wfdb.rdrecord(file_path)
    # 提取第一导联信号（MLII 导联 — 最常用于心律失常分析）
    signal = record.p_signal[:, 0]

    # Z-score 标准化，加速模型收敛
    signal = (signal - signal.mean()) / signal.std()

    total_len = len(signal)
    if total_len < sequence_length + 1:
        raise ValueError(f"信号长度 {total_len} 不足 {sequence_length+1}，请减小 sequence_length")

    # 构造监督学习对：x[i] → y[i] = x[i+1]（单步预测任务）
    x = signal[:sequence_length]
    y = signal[1:sequence_length+1]

    # 转换为模型输入格式 (batch=1, seq_len, input_size=1)
    x = torch.tensor(x, dtype=torch.float32).view(1, -1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(1, -1, 1)

    return x.to(device), y.to(device)

# ----------------- 4. 主训练逻辑 — 断点续训 + 自动导出 -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"======== 深度学习期末作业：正在训练 【{MODEL_TYPE}】 模型 ========")
    print(f"计算设备: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 实例化模型
    model = SleepRNNDemo(cell_type=MODEL_TYPE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_epoch = 0
    total_epochs = 200
    loss = torch.tensor(0.0)

    # 【亮点】断点续训 — 训练中断后自动恢复到上次保存的 epoch
    if os.path.exists(CHECKPOINT_PATH):
        print(f"发现【{MODEL_TYPE}】的历史训练记录，正在恢复...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"成功恢复！从第 {start_epoch} 个 Epoch 继续。")

    # 加载真实 MIT-BIH 生理数据
    x, y = load_mitbih_data('100', sequence_length=500, device=device)

    try:
        for epoch in range(start_epoch, total_epochs):
            model.train()
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'[{MODEL_TYPE}] Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}')
                save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n[成功] {MODEL_TYPE} 模型训练完成！")

        # 导出轻量化部署权重（仅保存 state_dict）
        torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
        print(f"部署权重已保存至: {FINAL_WEIGHTS_PATH}")
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
    except Exception as e:
        print(f"训练发生意外: {e}")
        save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)